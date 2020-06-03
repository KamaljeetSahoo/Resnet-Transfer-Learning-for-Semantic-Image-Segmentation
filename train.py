# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 01:46:28 2020

@author: Kamaljeet
"""

from dataloader import DatasetDesign
from torch.utils.data import DataLoader
from transfer_model import createDeepLabv3
import torch.optim as optim
import torch
import sys
import time

trainset = DatasetDesign(image_folder='dataset', seed=42, subset='train', train_per=0.8)
print("Dataset train allocated")
train_loader = DataLoader(trainset, batch_size = 4, shuffle=True)
print("Trainloader Done")

valset = DatasetDesign(image_folder='dataset', seed=42, subset='validation', train_per=0.8)
print("Dataset test'or'val allocated")
val_loader = DataLoader(valset, batch_size = 4, shuffle=False)
print("Testloader Done")

model = createDeepLabv3()
model = model.cuda()


lr = 1e-4
m = 0.9
Nepoch = 10
step_train = 0
step_val = 0
criterion = torch.nn.MSELoss(reduction='mean')
optimizer = optim.Adam(model.parameters(), lr = lr)
min_loss = sys.maxsize


#=============Training=============

for epoch in range(Nepoch):
    train_loss = 0.0
    tot_train_loss = 0.0
    
    model.train()

  

    start = time.time()

    for i, data in enumerate(train_loader, 0):
        step_train += 1
        image = data[0].cuda()
        mask = data[1].cuda()

        optimizer.zero_grad()
        
        with torch.set_grad_enabled(True):
            pred_mask = model(image)
            loss = criterion(pred_mask, mask)
    
            loss.backward()
            optimizer.step()      
            
        #always note that during training gradient descent is enabled
        '''
        #The following code would not work
        pred_mask = model(image)
        loss = criterion(pred_mask, mask/256)
        loss.backward()
        optimizer.step()
        '''

        train_loss += loss.item()
        tot_train_loss += loss.item()

        if(i%4 == 3):
            print('Training Loss after ' + str(epoch + 1) + ' epochs and step ' + str(i + 1) + ' is ' + str(train_loss/4))

            train_loss = 0.0

    print('\n')
    print('Total raining loss after ' + str(epoch + 1) + ' is ' + str(tot_train_loss/(i+1)))
    print('\n')

    
#===========validation==============
    
    #always note that during validation gradient descent is disabled
    with torch.no_grad():
        val_loss = 0.0
        model.eval()

        for i, data in enumerate(val_loader, 0):
           step_val += 1
           image = data[0].cuda()
           mask = data[1].cuda()
           pred_mask = model(image)
           loss = criterion(pred_mask, mask)
           val_loss += loss.item()

        val_loss = val_loss/(i + 1)
        print('Validation loss after ' + str(epoch + 1) + ' epochs is ' + str(val_loss))

        if val_loss < min_loss:
            
            min_loss = val_loss
            print('Improvement..........................................')
            torch.save(model.state_dict(), 'weight/' + 'epoch_' + str(epoch + 1) + '_loss_' + str(val_loss) + '.pt')
        
        else:
            
            print('No improvement>>>>>>>>>>>>>>>>>>>>>')
            
    end = time.time()


if(torch.cuda.is_available()):
    torch.cuda.empty_cache()
