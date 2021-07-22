import json
from io import BytesIO
import cv2
import os
from os.path import join, isdir, isfile
from PIL import Image

import streamlit as st
import pandas as pd
import numpy as np
import random

import tensorflow as tf

@st.cache()
def load_model(path='models/v1/fine_tuned_model'):
    """Retrieves the trained model"""
    return tf.keras.models.load_model(path)

@st.cache()
def get_class_names_and_shape(path='models/v1/data.json'):
    """Retrieves and formats the index to class label lookup dictionary needed to 
    make sense of the predictions. When loaded in, the keys are strings, this also
    processes those keys to integers."""
    
    with open(path, 'r') as fl:
        data = json.load(fl)

    class_names = [i.replace('_', ' ').title() for i in data['class_names']]
    IMAGE_SHAPE = data['image_shape']

    return [i.title() for i in class_names], IMAGE_SHAPE

@st.cache()
def predict(img, class_names, model, top=5):
    pred = model.predict(tf.expand_dims(img, axis=0)).squeeze()
    pred_df = pd.DataFrame()
    pred_df['Flower Name'] = class_names
    pred_df['Confident Level'] = np.round(pred*100, 2)
    pred_df.sort_values('Confident Level', ascending=False, inplace=True)
    pred_df = pred_df.iloc[:top]
    pred_df['Confident Level'] = ["{:.2f}%".format(i) for i in pred_df['Confident Level']]
    pred_df = pred_df.reset_index(drop=True)
    pred_df.index = pred_df.index + 1
    return pred_df

def image_to_numpy(file_dir, image_size=None):
    img =  cv2.cvtColor(cv2.imread(file_dir), cv2.COLOR_BGR2RGB)
    return cv2.resize(img, image_size) if image_size else img

def get_dirs(path):
    return [name for name in os.listdir(path) if isdir(join(path, name))]

def get_files(path):
    return [name for name in os.listdir(path) if isfile(join(path, name))]

def get_random_imgs(data_dir, rand_imgs=5, equal_img_per_class=None, rand_classes=None):
    data_dir = str(data_dir)
    class_names = get_dirs(data_dir)

    if rand_classes:
        for class_name in rand_classes:
            if class_name not in class_names:
                raise ValueError(f'"{class_name}" not found in "{data_dir}""')
    else:
        rand_classes = class_names

    if equal_img_per_class:
        rand_list = {class_name : equal_img_per_class for class_name in rand_classes}
    else:
        rand_list = {class_name : 0 for class_name in rand_classes}
        for class_name in random.choices(rand_classes, k=rand_imgs):
            rand_list[class_name] += 1

    rand = []
    for class_name, rand_img_num in rand_list.items():
        if rand_img_num:
            class_dir = join(data_dir, class_name)
            rand_images = random.choices(get_files(class_dir), k=rand_img_num)
            for i in rand_images:
                rand.append(join(class_dir, i))
    return rand

if __name__ == '__main__':
    model = load_model()
    class_names, IMAGE_SHAPE = get_class_names_and_shape()
    num_classes = len(class_names)

    file = st.file_uploader('Upload An Image of Food')
    
    if file:
        img = Image.open(file)
        prediction = predict(np.array(img.resize(IMAGE_SHAPE)), class_names, model)
        img = np.array(img)

        col1, col2 = st.beta_columns(2)
        
        col1.header("Here is the image you've selected")
        col1.image(img, use_column_width=True)
      
        col2.header("Here are the five most likely foods")
        col2.write(prediction.to_html(escape=False), unsafe_allow_html=True)
    
    st.title('The FOOD101 Blog!')
    instructions = f"""
        Here, you can classify {num_classes} types of Foods.
        These are : {', '.join(sorted(class_names))}
        """
    st.write(instructions)
    