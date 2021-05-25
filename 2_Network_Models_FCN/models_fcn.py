#FCN Architecture

#Imports
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def forward(self, inputs):
        
        if len(inputs.size())>2:
            #Average across all channels
            inputs = torch.mean(inputs, dim=-1) 

        x = F.relu(self.linear(inputs))
        x = F.dropout(x, training=self.training,p=self.dropout)
        y_pred = self.hidden2label(x)
        
        return y_pred

#**************************************************************************
#Functions - Model Training

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

##training the model
def train(model, device, train_loader, optimizer,loss_func, epoch):
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
        out = model(data)
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

def test(model, device, test_loader, n_labels, loss_func):
    model.eval()
    test_loss=0.; test_acc = 0.
    count = 0; total = 0
    prop_equal = torch.zeros([n_labels], dtype=torch.int32)
    
    #Include n_classes *********
    confusion_matrix = torch.zeros(n_labels, n_labels)
    ##no gradient desend for testing
    with torch.no_grad():
        for data, target_classes in test_loader:
            data, target_classes = data.to(device), target_classes.to(device)
            #Shapes
            #print('************')
            #print(f'\n Data shape ={data.shape}, Target class shape = {target_classes.shape}')
            
            out = model(data)  
            loss = loss_func(out, target_classes)
            test_loss += loss.sum().item()
            predictions = F.log_softmax(out, dim=1).argmax(dim=1) #log of softmax. Get the index/class with the greatest probability 
            #pred = out.argmax(dim=1,keepdim=True) # get the index of the max log-probability
            total += target_classes.size(0)

            #Accuracy - torch.eq computes element-wise equality
            test_acc += predictions.eq(target_classes.view_as(predictions)).sum().item() #.item gets actual sum value (rather then tensor object), like array[0]
            prop_equal += predictions.eq(target_classes).view_as(predictions)*1 #Convert boolean to 1,0 integer   

            #Confusion matrix
            for target_class, pred in zip(target_classes.view(-1), predictions.view(-1)): #Traverse the lists in parallel
                confusion_matrix[target_class.long(), predictions.long()] += 1 #Inrease number at that point in confusion matrix

            count += 1
    
    test_loss /= total
    test_acc /= total

    print(f'\nTOTAL ={total}')
    print('Test Loss {:4f} | Acc {:4f}'.format(test_loss,test_acc))
    return test_loss, test_acc, confusion_matrix, predictions, target_classes, prop_equal, count

def model_fit_evaluate(model, device, train_loader, test_loader, n_labels, optimizer, loss_func, num_epochs=100):
    best_acc = 0 
    best_confusion_matrix = 0; best_count = 0
    best_predictions = 0; best_target_classes = 0
    print('n_labels = {}'.format(n_labels))
    best_prop = torch.zeros([n_labels], dtype=torch.int32)
    model_history={}
    model_history['train_loss']=[]; model_history['train_acc']=[];
    model_history['test_loss']=[];  model_history['test_acc']=[];  
    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, device, train_loader, optimizer,loss_func, epoch)
        model_history['train_loss'].append(train_loss)
        model_history['train_acc'].append(train_acc)
        #Test accuracy for each epoch
        test_loss, test_acc, confusion_matrix, predictions, target_classes, prop_equal, count = test(model, device, test_loader, n_labels, loss_func)
        model_history['test_loss'].append(test_loss)
        model_history['test_acc'].append(test_acc)
        if test_acc > best_acc:
            best_acc = test_acc
            best_confusion_matrix = confusion_matrix
            best_predictions = predictions; best_target_classes = target_classes
            best_prop = prop_equal; best_count = count
            print("Model updated: Best-Acc = {:4f}".format(best_acc))

    #Performance
    print("Best Testing accuarcy:",best_acc)
    plot_history(model_history)
   
    return best_acc, best_confusion_matrix, best_predictions, best_target_classes, best_prop, best_count

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


def check_it_updates():
    print('It updates :) x4')
    