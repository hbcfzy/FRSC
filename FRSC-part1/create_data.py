# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 16:18:45 2021

@author: Zsh
"""

import librosa
import os
def getData(audio_path,list_path):
    sound_sum=0
    audios=os.listdir(audio_path)
    
    train_set=open(os.path.join(list_path,'train_list.csv'),'w',encoding="utf-8")
    test_set=open(os.path.join(list_path,'test_list.csv'),'w',encoding="utf-8")
    for i in range(len(audios)):
        sounds=os.listdir(os.path.join(audio_path,audios[i]))
        for sound in sounds:
            sound_path=os.path.join(audio_path,audios[i],sound)
            t=librosa.get_duration(filename=sound_path)
            # filter out audios with length < 2s
            if t>=2.0:
                if sound_sum % 30 ==0:
                    test_set.write("%s,%d\n" % (sound_path,i))
                else:
                    train_set.write("%s,%d\n" % (sound_path,i))
                sound_sum+=1
        print("Audio: %d/%d" % (i+1,len(audios)))
    
    train_set.close()
    test_set.close()
    
if __name__=='__main__':
    getData('dataset\\train_data','dataset')