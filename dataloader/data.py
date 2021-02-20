import cv2
import torch
import numpy as np
import random
import sys

from .has_aug import hide_patch

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
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # has_prob = 0.5
        # if(random.random() <=  has_prob):
        #     image = hide_patch(image)

        image = self.image_transform(image=image)["image"]

        return image, label