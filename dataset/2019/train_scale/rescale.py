import os
import cv2

root_link = '/home/hana/sonnh/kaggle-cassava/dataset/2019/train/'
scale_link = '/home/hana/sonnh/kaggle-cassava/dataset/2019/train_scale/'
for categorical in os.listdir(root_link):
    link = os.path.join(root_link, categorical)
    for image_name in os.listdir(link):
        image = cv2.imread(os.path.join(root_link, categorical, image_name))
        height, width = image.shape[:2]
        if height > width:
            new_height = int(height/width*600)
            new_image = cv2.resize(image, (600, new_height), interpolation=cv2.INTER_CUBIC)
        else:
            new_width = int(width/height*600)
            new_image = cv2.resize(image, (new_width, 600), interpolation=cv2.INTER_CUBIC)

        cv2.imwrite(os.path.join(scale_link, categorical) + '/' + image_name, new_image)