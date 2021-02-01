import tensorflow as tf
import numpy as np
import cv2
import os

image_feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'target': tf.io.FixedLenFeature([], tf.int64),
        'image_name': tf.io.FixedLenFeature([], tf.string) }

def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.cast(image, tf.float32)
                      
    image = tf.image.resize(image, [512, 512])
    image = tf.reshape(image, [512, 512, 3])
    return image

def _parse_image_function(example_proto):
    return tf.io.parse_single_example(example_proto, image_feature_description)
for i in os.listdir('/home/hana/sonnh/kaggle-cassava/dataset/train_mix/tfrec/'):
    raw_image_dataset  = tf.data.TFRecordDataset('/home/hana/sonnh/kaggle-cassava/dataset/train_mix/tfrec/' + i)
    print(i)
    parsed_image_dataset = raw_image_dataset.map(_parse_image_function)
    print(parsed_image_dataset)
    for image_features in parsed_image_dataset:
        #image_raw = image_features['image'].numpy()
        #print(image_features['target'].numpy(),image_features['image_name'].numpy())
        #image_target = np.frombuffer(image_features['target'].numpy(), np.string)

        name = image_features['image_name'].numpy().decode()
        img = decode_image(image_features['image']).numpy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #print(img.shape, name)
        #print(type(img))
        cv2.imwrite('/home/hana/sonnh/kaggle-cassava/dataset/train_mix/image/' + name, img)
        #image_raw.shape = (600, 600, 3)
        #print(image_raw)