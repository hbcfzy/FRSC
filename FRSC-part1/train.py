# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 16:42:12 2021

@author: Zsh
"""

import librosa
import glob
import os
import pandas as pd
import random
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch
import torch.nn.functional as F
import torch.optim as optim
import pickle
import se_resnet

# -------------------Data Preprocessing-------------------

train=pd.read_csv("dataset\\train_list1.csv",sep=",")
test=pd.read_csv("dataset\\test_list.csv",sep=",")

# Randomly plot an audio
# i=random.choice(train.index)
# audio_name=train.filename[i]
# path=str(audio_name)
# print("Label: ",train.label[i])
# x, sr=librosa.load(str(audio_name))
# plt.figure(figsize=(12,4))
# librosa.display.waveplot(x,sr=16000)

# -------------------Feature Extraction-------------------
# def parser(row):
#     file_name=row.filename
#     try:
#         x, sr=librosa.load(file_name)
#         mfccs=np.mean(librosa.feature.mfcc(y=x, sr=sr, n_mfcc=90),axis=1)
#         stfts = np.abs(librosa.stft(x,n_fft=112))
#         # chromas = np.mean(librosa.feature.chroma_stft(S=stfts, sr=sr).T)
#         # mels = np.mean(librosa.feature.melspectrogram(x, sr=sr),axis=1)
#         # contrasts = np.mean(librosa.feature.spectral_contrast(S=stfts, sr=sr).T)
#         print("processing")
        
#     except Exception as e:
#         print("error")
#         return None,None
#     feature_mfcc=mfccs
#     feature_stft=np.mean(stfts,axis=1)
#     # feature_mel=mels
#     feature_both=np.concatenate((feature_mfcc,feature_stft),axis=0)
#     # feature_chroma=chromas
#     # feature_contrast=np.mean(contrasts)
    
#     label=row.label
#     return [feature_both,label]

# temp_train=train.apply(parser,axis=1,result_type="expand")
# temp_test=test.apply(parser,axis=1,result_type="expand")

# Store temp data in plk file
# with open("data_train_BOTH.plk",'wb') as f:
#     pickle.dump(temp_train,f)
# with open("data_test_BOTH.plk",'wb') as f:
#     pickle.dump(temp_test,f)

# # Save processed data to csv file
# with open("data_train_THREE.plk",'rb') as f:
#     temp_train=pickle.load(f)
# temp_train.to_csv('test_THREE.csv', sep=',',header=False,index=False)
# with open("data_test_BOTH.plk",'rb') as f:
#     temp_test=pickle.load(f)
# temp_test.to_csv('test_BOTH.csv', sep=',',header=False,index=False)

class Dataset(Dataset):
    def __init__(self,xy,use_gpu):
        xy=xy.values
        self.x=xy[:,[0]]
        self.y=xy[:,[1]]
        list_x=[]
        for row in self.x:
            list_x.append(row.tolist()[0])
        self.x=torch.tensor(list_x)
        
        list_y=[]
        for row in self.y:
            list_y.append(row.tolist()[0])
        self.y=torch.tensor(list_y)
        
        self.len=xy.shape[0]
        if use_gpu:
            self.x=self.x.to("cuda")
            self.y=self.y.to("cuda")
        
    def __getitem__(self,index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len

with open("data_train_SC.plk",'rb') as f:
    temp=pickle.load(f)
# print(len(temp[0][0]))
train_dataset=Dataset(temp,torch.cuda.is_available())

with open("data_test_SC.plk",'rb') as f2:
    temp2=pickle.load(f2)
test_dataset=Dataset(temp2,torch.cuda.is_available())

train_loader=DataLoader(dataset=train_dataset,batch_size=64,shuffle=True,num_workers=2)
test_loader=DataLoader(dataset=test_dataset,batch_size=64,shuffle=False,num_workers=2)

# Channel Encoder (Classifier)
# class Model(torch.nn.Module):
#     def __init__(self):
#         super(Model,self).__init__()
#         self.fc1=torch.nn.Linear(384,320)
#         self.fc2=torch.nn.Linear(320,256)
#         self.fc3=torch.nn.Linear(256,192)
#         self.fc4=torch.nn.Linear(192,128)
#         self.fc5=torch.nn.Linear(128,96)
#         self.fc6=torch.nn.Linear(96,80)
#         self.fc7=torch.nn.Linear(80,64)
#         self.fc8=torch.nn.Linear(64,50)
        
#     def forward(self,x):
#         x-x.view(-1,384)
#         x=F.relu(self.fc1(x))
#         # x=self.dp(x)
#         x=F.relu(self.fc2(x))
#         x=F.relu(self.fc3(x))
#         x=F.relu(self.fc4(x))
#         x=F.relu(self.fc5(x))
#         x=F.relu(self.fc6(x))
#         x=F.relu(self.fc7(x))
#         return self.fc8(x)

model = getattr(se_resnet,"se_resnet_50")(num_classes = 50)
criterion=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters())

def test():
    correct=0
    total=0

    with torch.no_grad():
        for data in test_loader:
            x,y_test=data
            x=x.view(-1,3,7,7)
            y_test_pred=model(x)
            _,predicted=torch.max(y_test_pred.data,dim=1)
            
            y_test_list=y_test.data.numpy()
            predict_list=predicted.data.numpy()
            
            total+=len(predict_list)
            for i in range(len(predict_list)):
                if y_test_list[i]==predict_list[i]:
                    correct+=1
    acc=1.0*correct/total
    print('Accuracy: %.5f' % acc)
    return acc

loss_value=[]   
acc_value=[]
if __name__=='__main__':
    if torch.cuda.is_available():
        model=model.cuda()
        criterion=criterion.cuda()
    for epoch in range(100):
        batch_loss=0
        for batch_idx,data in enumerate(train_loader,0):
            inputs,labels=data
            inputs=inputs.view(-1,3,7,7)
            optimizer.zero_grad()
            y_pred=model(inputs)
            loss=criterion(y_pred,labels)
            
            batch_loss+=loss.item()
            
            loss.backward()
            optimizer.step()
        loss_value.append(batch_loss/64.0)
        print("epoch {}, Loss: {}".format(epoch+1,batch_loss))
        acc=test()
        acc_value.append(acc)
                
    plt.plot(range(len(loss_value)),loss_value)
    plt.show()
    plt.plot(range(len(acc_value)),acc_value)
    plt.show()
