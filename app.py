from PIL import Image
import numpy as np 
import streamlit as st 
import pandas as pd
import numpy as np
from keras.preprocessing import image
from tensorflow.keras.models import load_model
import tensorflow as tf



import torch
import pandas as pd
import shutil
import io
import numpy as np
import ast
import cv2
import os
from tqdm.auto import tqdm
import shutil as sh
import matplotlib.pyplot as plt

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tqdm import tqdm, trange
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import argparse
import os
import sys
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

#from common import DetectMultiBackend
from models.common import DetectMultiBackend

from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
import detect as det
from types import SimpleNamespace

uploaded_files = st.file_uploader("Загрузите изображения с других камер для того, чтобы узнать где есть мусор", 
       accept_multiple_files=True)
for uploaded_file in uploaded_files:
    im = Image.open(uploaded_file)
    st.image(im)
    st.write(uploaded_file.name)

    opt = SimpleNamespace(agnostic_nms=False, augment=False, classes=None, conf_thres=0.5, device='CPU', dnn=False, exist_ok=False,\
                           half=False, hide_conf=False, hide_labels=False, imgsz=[640, 640], iou_thres=0.45, line_thickness=3,\
                           max_det=1000, name='exp', nosave=False, project='runs/detect', save_conf=False, save_crop=False,\
                           save_txt=False,source=uploaded_file.name, update=False, view_img=False, visualize=False, weights=['best.pt'])


    det.main(opt)
    imgg = Image.open('runs/detect/exp37/234.png')
    st.image(imgg)
    st.write("Успешно")

# Function to Read and Manupilate Images
def load_img(img):
    #im = image.load_img(img, target_size=(224, 224))
    im = Image.open(img)
    imm = im.resize((224, 224), Image.ANTIALIAS)
    imgg = np.array(imm)
    return imgg






    #Uploading the File to the Page
   # uploaded_files = st.file_uploader("Загрузите изображения с других камер для того, чтобы узнать где есть мусор", 
   #     accept_multiple_files=True)
   # files = []
   # labels = []

   # model = load_model('models/baseline.h5') # load model


    #for uploaded_file in uploaded_files:
        #bytes_data = uploaded_file.read()
        #st.write("filename:", uploaded_file.name)
        #st.write(bytes_data)
        # Perform your Manupilations (In my Case applying Filters)
   #     img = load_img(uploaded_file)
    #    st.image(img)
    #    st.write("Изображение успешно загружено")
    #    files.append(uploaded_file.name)
#
 #       x = np.expand_dims(img, axis=0)
  #      img_preprocessed = tf.keras.applications.resnet_v2.preprocess_input(x)
   #     #images = np.vstack([x])
    #    classes = model.predict(img_preprocessed)
     #   target = np.argmax(classes, axis = 1)
      #  labels.append(target)

    #x1 = pd.DataFrame(files, columns = ["Filename"])
    #x2 = pd.DataFrame(labels, columns = ["Класс"])
    #x2["Класс"] = x2["Класс"].map({0:"Нет мусора", 1: "Есть мусор"})
    #df = pd.concat([x1,x2], axis = 1)
    #st.write(df)

