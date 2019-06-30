'''
Created on Oct 17, 2018
Modified on Abr 25, 2019

@author: deckyal
@modified by: marc alcaraz
'''

import numpy as np
import re
from PIL import Image,ImageFilter
import torch
from torch.utils import data
import FileWalker


class ImageDatasets(data.Dataset):
    
    def __init__(self, data_list = ["300VW-Train"],dir_gt = None, blurLevel = None, onlyFace = True,step = 1, transform = None, image_size =224):

        #Read initalizes the list of file path and possibliy label as well. 
       
        self.blurLevel  = blurLevel
        self.transform = transform
        self.onlyFace = onlyFace
        
        self.imageHeight = image_size
        self.imageWidth = image_size
        
        list_gt = []
        list_labels_t = []
        
        counter_image = 0
        annot_name = 'annot'
        
        if dir_gt is not None : 
            annot_name = dir_gt
            
        for data in data_list : 
            print(("Opening "+data))
            for f in file_walker.walk("../" + "images/"+data+"/"):                
                if f.isDirectory: # Check if object is directory
                    print((f.name, f.full_path)) # Name is without extension
                    for sub_f in f.walk():
                        if sub_f.isDirectory: # Check if object is directory
                            list_dta = []
                            print (sub_f.name)
                            for sub_sub_f in sub_f.walk(): #this is the data
                                if(".npy" not in sub_sub_f.full_path):
                                    list_dta.append(sub_sub_f.full_path)
                            
                            if(sub_f.name == annot_name) : #If that's annot, add to labels_t 
                                list_labels_t.append(sorted(list_dta))
                            elif(sub_f.name == 'img'): #Else it is the image
                                list_gt.append(sorted(list_dta))
                                counter_image+=len(list_dta)
        
        self.length = counter_image 
        print(counter_image)
        print("Now opening keylabels")
        
        list_labels = []     
        for lbl in list_labels_t :
            lbl_68 = [] #Per folder
            counter = 0
            for lbl_sub in lbl :
                counter=counter+1
                if ('pts' in lbl_sub) : 
                    x = []
                    with open(lbl_sub) as file:
                        data2 = [re.split(r'\t+',l.strip()) for l in file]
                    for i in range(len(data2)) :
                        if(i not in [0,1,2,len(data2)-1]):
                            x.append([ float(j) for j in data2[i][0].split()] )
                    print(counter)
                    lbl_68.append(np.array(x).flatten('F')) #1 record
            list_labels.append(lbl_68)
            
        list_images = []
        list_ground_truth = np.zeros([counter_image,25])
        
        #Flatten it to one list
        indexer = 0
        for i in range(0,len(list_gt)): #For each dataset
            for j in range(0,len(list_gt[i]),step): #for number of data #n_skip is usefull for video data
                list_images.append(list_gt[i][j])
                
                list_ground_truth[indexer] = np.array(list_labels[i][j]).flatten('F')
                indexer += 1
                
        self.l_imgs = list_images
        self.l_gt = list_ground_truth

    def __getitem__(self,index):
        
        #Read all data, transform etc.
        #In video, the output will be : [batch_size, sequence_size, channel, width, height] 
        #Im image : [batch_size, channel, width, height]
         
        x,label  = self.l_imgs[index],self.l_gt[index].copy()
        tImage = Image.open(x).convert("RGB")
        tImageB = None
        
        if self.transform is not None:
            img = self.transform(tImage)
        
        return img,torch.FloatTensor(label)
    
    def __len__(self):
        
        return len(self.l_imgs)


