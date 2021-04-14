#FCN & GCN Architecture

#Imports
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_scatter import scatter_add

from util_funcs import sparse_dense_mat_mul

#1. FCN 
class FCN(nn.Module):
    'Fully Connected Network'
    def __init__(self, input_dim, output_dim, hidden_dim=128, dropout=0.2):
        super(FCN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        #i. Input layer
        self.linear = nn.Linear(self.input_dim, hidden_dim)
        
        #ii. Hidden layer + final layer 
        self.hidden2label = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4), 
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim // 4, self.output_dim),
        )

    def forward(self, inputs, adj=None):
        
        if len(inputs.size())>2:
            #Average across all channels
            inputs = torch.mean(inputs, dim=-1) 

        x = F.relu(self.linear(inputs))
        x = F.dropout(x, training=self.training,p=self.dropout)
        y_pred = self.hidden2label(x)
        
        return y_pred

#*********************************************************************************
#2. Graph CNN
class GraphConv(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    math:
      Z = D(-1/2)AstarD(-1/2)X; Astar=I+A
    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        K (int): Chebyshev filter size, *i.e.* number of hops :math:`K`.
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_paramadd_self_loopseter('bias', None)
        ####initialize all layer parameters
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.kaiming_normal_(self.weight.data, a=0, mode='fan_in')
        if self.bias is not None:
            nn.init.constant_(self.bias.data, 0.0)

    def forward(self,  x, edge_index, edge_weight=None):
        batch, num_nodes = x.size(0), x.size(1)
        ##first adjust the adj matrix with diag elements
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight, 1, num_nodes) #self.
        row, col = edge_index
        
        edge_weight = edge_weight.view(-1)
        assert edge_weight.size(0) == edge_index.size(1)
        
        ###degree matrix
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        # Compute normalized and rescaled Laplacian.
        deg = deg.pow(-0.5)
        deg[torch.isinf(deg)] = 0
        lap = deg[row] * edge_weight * deg[col]
        ###Rescale the Laplacian eigenvalues in [-1, 1]
        #fill_value = 0.05  ##-0.5
        #edge_index, lap = add_self_loops(edge_index, lap, fill_value, num_nodes)

        x = torch.matmul(x, self.weight)
        out = sparse_dense_mat_mul(edge_index, lap, num_nodes, x.permute(1, 2, 0).contiguous().view((num_nodes, -1))).view((num_nodes, -1, batch)).permute(2, 0,1)  # sparse_dense_mat_mul(edge_index, lap, num_nodes, x)

        if self.bias is not None:
            out = out + self.bias

        return out
    
    #Use the old add_self_loop function from pytorch-geometric
    #def maybe_num_nodes(self, index, num_nodes=None):
        #return index.max().item() + 1 if num_nodes is None else num_nodes
    
    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

#3. Graph + k-order neighbourhood
class ChebNet(nn.Module):
    def __init__(self, nfeat, nfilters, nclass, K=2, nodes=360, nhid=128, gcn_layer=2, dropout=0, gcn_flag=False):
        super(ChebNet, self).__init__()
        self.gcn_layer = gcn_layer

        ####feature extracter
        self.graph_features = nn.ModuleList()
        if gcn_flag is True:
            print('Using GCN Layers instead')
            self.graph_features.append(GraphConv(nfeat, nfilters))
        else:
            self.graph_features.append(ChebConv(nfeat, nfilters, K))
        for i in range(gcn_layer):
            if gcn_flag is True:
                self.graph_features.append(GraphConv(nfilters, nfilters))
            else:
                self.graph_features.append(ChebConv(nfilters, nfilters, K))


        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity(dropout)

        # Define the output layer
        self.graph_nodes = nodes
        self.hidden_size = self.graph_nodes
        self.pool = nn.AdaptiveMaxPool2d((self.hidden_size,1))

        self.linear = nn.Linear(self.hidden_size, nclass)
        self.hidden2label = nn.Sequential(
            nn.Linear(self.hidden_size, nhid),
            nn.ReLU(True),
            nn.Dropout(p=0.25),
            nn.Linear(nhid, nhid // 4),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(nhid // 4, nclass),
        )

    def forward(self, inputs, adj_mat):
        edge_index = adj_mat._indices()
        edge_weight = adj_mat._values()
        batch = inputs.size(0)
        ###gcn layer
        x = inputs
        for layer in self.graph_features:
            x = F.relu(layer(x, edge_index, edge_weight))
            x = self.dropout(x)
        x = self.pool(x)
        ###linear dense layer
        # y_pred = self.linear(x.view(batch,-1))
        y_pred = self.hidden2label(x.view(batch, -1))
        return y_pred

#**********************************************************************
class ChebConv(nn.Module):
    """The chebyshev spectral graph convolutional operator
    .. math::
        \mathbf{Z}^{(0)} &= \mathbf{X}

        \mathbf{Z}^{(1)} &= \mathbf{\hat{L}} \cdot \mathbf{X}

        \mathbf{Z}^{(k)} &= 2 \cdot \mathbf{\hat{L}} \cdot
        \mathbf{Z}^{(k-1)} - \mathbf{Z}^{(k-2)}

    and :math:`\mathbf{\hat{L}}` denotes the scaled and normalized Laplacian.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        K (int): Chebyshev filter size, *i.e.* number of hops :math:`K`.
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(self, in_channels, out_channels, K, bias=True):
        super(ChebConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.FloatTensor(K+1, in_channels, out_channels))

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_channels))
        else:
            self.register_parameter('bias', None)
        ####initialize all layer parameters
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.kaiming_normal_(self.weight.data, a=0, mode='fan_in')
        if self.bias is not None:
            nn.init.constant_(self.bias.data, 0.0)

    def forward(self, x, edge_index, edge_weight=None):
        """"""
        # edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        # print(x.size(), edge_index.size())
        row, col = edge_index
        batch, num_nodes, num_edges, K = x.size(0), x.size(1), row.size(0), self.weight.size(0)
            
        edge_weight = edge_weight.view(-1)
        assert edge_weight.size(0) == edge_index.size(1)
        
        ###degree matrix
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        # Compute normalized and rescaled Laplacian.
        deg = deg.pow(-0.5)
        deg[torch.isinf(deg)] = 0
        lap = -deg[row] * edge_weight * deg[col]
        ###Rescale the Laplacian eigenvalues in [-1, 1]
        ##rescale: 2L/lmax-I; lmax=1.0
        fill_value = -0.05  ##-0.5
        edge_index, lap = add_self_loops(edge_index, lap, fill_value, num_nodes)
        lap *= 2

        ########################################
        # Perform filter operation recurrently.
        Tx_0 = x
        out = torch.matmul(Tx_0, self.weight[0])
        if K > 1:
            Tx_1 = sparse_dense_mat_mul(edge_index, lap, num_nodes, x.permute(1, 2, 0).contiguous().view((num_nodes, -1))).view((num_nodes, -1, batch)).permute(2, 0,1)  # sparse_dense_mat_mul(edge_index, lap, num_nodes, x)
            out = out + torch.matmul(Tx_1, self.weight[1])

        for k in range(2, K):
            Tx_2 = 2 * sparse_dense_mat_mul(edge_index, lap, num_nodes, x.permute(1, 2, 0).contiguous().view((num_nodes, -1))).view((num_nodes, -1, batch)).permute(2,0,1) - Tx_0
            # 2 * sparse_dense_mat_mul(edge_index, lap, num_nodes, Tx_1) - Tx_0
            out = out + torch.matmul(Tx_2, self.weight[k])
            Tx_0, Tx_1 = Tx_1, Tx_2

        if self.bias is not None:
            out = out + self.bias

        return out

    def __repr__(self):
        return '{}({}, {}, K={})'.format(self.__class__.__name__, self.in_channels, self.out_channels, self.weight.size(0)-1)
    
#********************************************************************
class Adjacency_matrix():
    
    def __init__(self, connectivity_matrix, n_neighbours = 8):
        
        self.connectivity_matrix = connectivity_matrix
        self.n_neighbours = n_neighbours 
        #self.adj_mat_sp = None #build_adjacency(self, connectivity_matrix, n_neighbours)
        #self.adj_mat_sp_torch = None #sparse_mx_to_torch_sparse_tensor(self,  self.adj_mat_sp)
        #self._orig_shape = None
    
    def get_adj_mat_sp(self):
        
        self.adj_mat_sp = self.build_adjacency(self.connectivity_matrix, self.n_neighbours)
        
        return self.adj_mat_sp
    
    def get_adj_sp_torch_tensor(self):
        
        self.adj_mat_sp = self.build_adjacency(self.connectivity_matrix, self.n_neighbours)       
        self.adj_mat_sp_torch = self.sparse_mx_to_torch_sparse_tensor(self.adj_mat_sp)
        
        return self.adj_mat_sp_torch

    def get_k_strongest_connections(self, connectivity_matrix, n_neighbours):
            
        ''' Get k greatest (in magnitude) connections
            
        Returns
        connections 
        idx_connections

        '''
        idx = np.argsort(-connectivity_matrix)[:, 1:n_neighbours + 1] #Get indices of greatest values - greatest percentage of fibres (-) -> decreasing order
        connections = np.array([connectivity_matrix[i, idx[i]] for i in range(connectivity_matrix.shape[0])]) #Gets top 8 greatest connections
        connections[connections < 0.1] = 0

        return connections, idx
    
    def sparse_mx_to_torch_sparse_tensor(self, adj_mat_sparse):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        
        sparse_mx = adj_mat_sparse.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        
        return torch.sparse.FloatTensor(indices, values, shape)

    #Adjacency matrix
    def build_adjacency(self, connectivity_matrix, n_neighbours):
        """Return the adjacency matrix of a kNN graph."""
        
        #Get k strongest connections
        connections, idx = self.get_k_strongest_connections(connectivity_matrix, n_neighbours)
        
        #Sparse matrix
        M, k = connections.shape
        assert M, k == idx.shape #M - number of vertices. k == nearest neighbours 
        assert connections.min() >= 0

        # Weights.
        sigma2 = np.mean(connections[:, -1])**2
        connections = np.exp(- connections**2 / sigma2)

        # Weight matrix.
        I = np.arange(0, M).repeat(k) #row
        J = idx.reshape(M*k) #col
        V = connections.reshape(M*k) #data 
        W = sparse.coo_matrix((V, (I, J)), shape=(M, M)) #COO is a fast format for constructing sparse matrices

        # No self-connections.
        W.setdiag(0)

        # Non-directed graph.
        bigger = W.T > W
        W = W - W.multiply(bigger) + W.T.multiply(bigger)
        return W


def add_self_loops(edge_index, edge_weight=None, fill_value=1, num_nodes=None):

    r"""Adds a self-loop :math:`(i,i) \in \mathcal{E}` to every node
    :math:`i \in \mathcal{V}` in the graph given by :attr:`edge_index`.
    In case the graph is weighted, all existent self-loops will be removed and
    replaced by weights denoted by :obj:`fill_value`.

    Reason for adding self-loop:
    The aggregated representation of a node does not include its own features.
    The representation is an aggregate of the features ofneighbor nodes, 
    so only nodes that has a self-loop will include their own features in the aggregate.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_weight (Tensor, optional): One-dimensional edge weights.
            (default: :obj:`None`)
        fill_value (int, optional): If :obj:`edge_weight` is not :obj:`None`,
            will add self-loops with edge weights of :obj:`fill_value` to the
            graph. (default: :obj:`1`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`)
"""
    num_nodes = maybe_num_nodes(edge_index, num_nodes) #self

    loop_index = torch.arange(0,
                              num_nodes,
                              dtype=torch.long,
                              device=edge_index.device)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)

    if edge_weight is not None:
        assert edge_weight.numel() == edge_index.size(1)
        loop_weight = edge_weight.new_full((num_nodes, ), fill_value)
        edge_weight = torch.cat([edge_weight, loop_weight], dim=0)

    edge_index = torch.cat([edge_index, loop_index], dim=1)

    return edge_index, edge_weight

    #Use the old add_self_loop function from pytorch-geometric
def maybe_num_nodes(index, num_nodes=None):
    return index.max().item() + 1 if num_nodes is None else num_nodes
    
    
#****************************************************************************************************
#Functions - Model Training
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

##training the model
def train(model, adj_mat, device, train_loader, optimizer,loss_func, epoch):
    model.train()

    acc = 0.
    train_loss = 0.
    total = 0
    t0 = time.time()
    for batch_idx, (data,target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        #print('******************************')
        #print('DATA SHAPE = {}'.format(data.shape)) #torch.Size([20, 360, 17]) 
        #print('TARGET SHAPE = {}'.format(target.shape)) #torch.Size([20])
        
        optimizer.zero_grad()
        out = model(data, adj_mat)
        loss = loss_func(out,target)
        pred = F.log_softmax(out, dim=1).argmax(dim=1)
        #print('LOSS SHAPE = {}'.format(loss.shape))
        
        total += target.size(0)
        train_loss += loss.sum().item()
        #Accuracy - torch.eq computes element-wise equality
        acc += pred.eq(target.view_as(pred)).sum().item()
        
        loss.backward()
        optimizer.step()


    print("\nEpoch {}: \nTime Usage:{:4f} | Training Loss {:4f} | Acc {:4f}".format(epoch,time.time()-t0,train_loss/total,acc/total))
    return train_loss/total, acc/total

def test(model, adj_mat, device, test_loader, n_labels, loss_func):
    model.eval()
    test_loss=0.
    test_acc = 0.
    total = 0
    #Include n_classes *********
    confusion_matrix = torch.zeros(n_labels, n_labels)
    ##no gradient desend for testing
    with torch.no_grad():
        for data, target_classes in test_loader:
            data, target_classes = data.to(device), target_classes.to(device)
            out = model(data, adj_mat)
            
            loss = loss_func(out, target_classes)
            test_loss += loss.sum().item()
            predictions = F.log_softmax(out, dim=1).argmax(dim=1) #log of softmax. Get the index with the greatest probability 
            #pred = out.argmax(dim=1,keepdim=True) # get the index of the max log-probability
            total += target_classes.size(0)
            #Accuracy - torch.eq computes element-wise equality
            test_acc += predictions.eq(target_classes.view_as(predictions)).sum().item() #.item gets actual sum value (rather then tensor object), like array[0]

            #Confusion matrix
            for target_class, pred in zip(target_classes.view(-1), predictions.view(-1)): #Traverse the lists in parallel
                confusion_matrix[t.long(), p.long()] += 1 #Inrease number at that point in confusion matrix
            
            #Inspect
            print(f'Total = {total}')
    
    test_loss /= total
    test_acc /= total

    print('Test Loss {:4f} | Acc {:4f}'.format(test_loss,test_acc))
    return test_loss, test_acc, confusion_matrix

def model_fit_evaluate(model,adj_mat,device,train_loader, test_loader, n_labels, optimizer,loss_func,num_epochs=100):
    best_acc = 0 
    best_confusion_matrix = 0
    model_history={}
    model_history['train_loss']=[];
    model_history['train_acc']=[];
    model_history['test_loss']=[];
    model_history['test_acc']=[];  
    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, adj_mat, device, train_loader, optimizer,loss_func, epoch)
        model_history['train_loss'].append(train_loss)
        model_history['train_acc'].append(train_acc)

        test_loss, test_acc, confusion_matrix = test(model, adj_mat, device, test_loader, n_labels, loss_func)
        model_history['test_loss'].append(test_loss)
        model_history['test_acc'].append(test_acc)
        if test_acc > best_acc:
            best_acc = test_acc
            best_confusion_matrix = confusion_matrix
            print("Model updated: Best-Acc = {:4f}".format(best_acc))

    print("Best Testing accuarcy:",best_acc)

    print('\n Confusion Matrix:')

    plot_history(model_history)
   
    return best_acc, best_confusion_matrix

def plot_history(model_history):
    plt.figure(figsize=(10,4))
    plt.subplot(121)
    plt.plot(model_history['train_acc'], color='r')
    plt.plot(model_history['test_acc'], color='b')
    plt.xlabel('Epochs')
    plt.ylabel('Prediction Accuracy')
    plt.legend(['Training', 'Validation'])

    plt.subplot(122)
    plt.plot(model_history['train_loss'], color='r')
    plt.plot(model_history['test_loss'], color='b')
    plt.xlabel('Epochs')
    plt.ylabel('Loss Function')
    plt.legend(['Training', 'Validation'])
    plt.show()

#Inspect multiclass classification
class MulticlassClassification(nn.Module):
    def __init__(self, num_feature, num_class):
        super(MulticlassClassification, self).__init__()
        
        self.layer_1 = nn.Linear(num_feature, 512)
        self.layer_2 = nn.Linear(512, 128)
        self.layer_3 = nn.Linear(128, 64)
        self.layer_out = nn.Linear(64, num_class) 
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.batchnorm1 = nn.BatchNorm1d(512)
        self.batchnorm2 = nn.BatchNorm1d(128)
        self.batchnorm3 = nn.BatchNorm1d(64)
        
    def forward(self, x):
        x = self.layer_1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        
        x = self.layer_2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.layer_3(x)
        x = self.batchnorm3(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.layer_out(x)
        
        return x

def check_it_updates():
    print('It updates :)')