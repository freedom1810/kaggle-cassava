import cv2
import torch
import numpy as np
import random
import sys

#has aug 
# https://github.com/kkanshul/Hide-and-Seek/blob/b4c85181cdfeafafe890266245284c22fd6176ec/hide_patch.py#L8

def hide_patch(img):
    # get width and height of the image
    s = img.shape
    wd = s[0]
    ht = s[1]

    # possible grid size, 0 means no hiding
    grid_sizes=[0,16,32,44,56]

    # hiding probability
    hide_prob = 0.5
 
    # randomly choose one grid size
    grid_size= grid_sizes[random.randint(0,len(grid_sizes)-1)]

    # hide the patches
    if(grid_size!=0):
         for x in range(0,wd,grid_size):
             for y in range(0,ht,grid_size):
                 x_end = min(wd, x+grid_size)  
                 y_end = min(ht, y+grid_size)
                 if(random.random() <=  hide_prob):
                       img[x:x_end,y:y_end,:]=0

    return img


class CassaDataset(torch.utils.data.Dataset):
    def __init__(self, df, image_folder, image_transform):
        self.df = df
        self.image_folder = image_folder
        self.image_transform = image_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        label = item["label"].astype("float32")
        #print(label)
        label_encoded = np.zeros((label.size, 5))
        label_encoded[np.arange(label.size),int(label)] = 1
        image_path = self.image_folder + item['image_id']
        image = cv2.imread(image_path)
        # image = cv2.cvtColor(image, code=cv2.COLOR_BGR2RGB)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        has_prob = 0.5
        if(random.random() <=  has_prob):
            image = hide_patch(image)

        image = self.image_transform(image=image)["image"]

        return image, label