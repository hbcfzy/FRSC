# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 16:11:59 2021

@author: Zsh
"""

import os
import csv
from shutil import copyfile
from sys import exit
def mkdir(path):
    isExist=os.path.exists(path)
    if not isExist:
        os.makedirs(path)
        #print("Make dir successful")
    
def move_file(file_name,target_name):
    try:
        copyfile(file_name,target_name)
    except:
        print('Failed')
        exit(1)
    print("Succeed")

with open('esc50_meta.csv','r') as f:
    reader=csv.reader(f)
    for row in reader:
        file_name='dataset\\audio\\'+row[0]
        dir_name='dataset\\train_data\\'+row[2]
        mkdir(dir_name)
        
        target_name=dir_name+'\\'+row[0]
        move_file(file_name,target_name)
        