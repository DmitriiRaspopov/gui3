from PIL import Image
import numpy as np 
import streamlit as st 
import pandas as pd
import numpy as np
from keras.preprocessing import image
from tensorflow.keras.models import load_model
import tensorflow as tf

data = pd.read_csv("geo_cam_done.csv")

# Function to Read and Manupilate Images
def load_img(img):
    #im = image.load_img(img, target_size=(224, 224))
    im = Image.open(img)
    imm = im.resize((224, 224), Image.ANTIALIAS)
    imgg = np.array(imm)
    return imgg


# Create the title for the web app
st.title("Система видеомониторинга и прогнозирования событий")

# Настройка боковой панели
st.sidebar.title("События для прогнозирования")

add_selectbox1 = st.sidebar.multiselect(
    'События категории "ТКО"',
    ["Обнаружение мусора на улице и переполненности мусорных контейнеров"], "Обнаружение мусора на улице и переполненности мусорных контейнеров"
)

add_selectbox2 = st.sidebar.multiselect(
    'События категории "Безопасный город"',
    ("Обнаружение ДТП", "Обнаружение диких животных (в том числе собак)", 
        "Обнаружение пробок и оценка загруженности перекрестка", 
"Обнаружение драк"," Обнаружение оружия, терактов и подозрительных людей",
"Оценка аварийности перекрёстка исходя из текущих условий",
"Оценка инфраструктуры: дороги, знаки дорожные, ремонтные работы",
"Обнаружение несанционированных митингов"), "Обнаружение ДТП"
)


add_selectbox3 = st.sidebar.multiselect(
    'События категории "Безопасный двор"',
    ( "Обнаружение кровли перегруженной снегом",
"Проверка работоспособности комендантского часа",
"Работы ЖКХ", "Работы по засыпке от снега реагентом, открытые люки",
"Обнаружение отключений электрической энергии на территории"), "Обнаружение отключений электрической энергии на территории"   
)




if "Обнаружение мусора на улице и переполненности мусорных контейнеров" in add_selectbox1:
    st.header("Обнаружение мусора")
    data2 = data[data["Type"] == "ТКО"]
    selected_cams = st.multiselect("Выберите камеру(ы)", data2['Адрес установки камеры'].unique(), data2['Адрес установки камеры'].unique())
    #st.write("Выбранные камеры:", selected_cams)
    if len(selected_cams) == 1:
        select_data = data2.loc[data2["Адрес установки камеры"].astype(str) == selected_cams[0]]
        st.subheader("Карта с камерами")
        st.map(select_data)
        st.write("Изображение с камеры")
        immag = Image.open("tko/1.png")
        st.image(immag)
    else:
        select_data = data2.loc[data2["Адрес установки камеры"].isin(selected_cams)]
        st.subheader("Карта с камерами")
        st.map(select_data)
        with st.expander("Изображения, полученные с камер"):
            for i in range(1,len(select_data)-1):
                immag = Image.open(f"tko/{i}.png")
                st.image(immag)



    # Uploading the File to the Page
    #uploaded_files = st.file_uploader("Загрузите изображения с других камер для того, чтобы узнать где есть мусор", 
    #    accept_multiple_files=True)
    #files = []
    #labels = []

    #model = load_model('models/baseline.h5') # load model


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

if "Обнаружение ДТП" in add_selectbox2:
    st.header("Обнаружение ДТП")
    data2 = data[data["Type"] == "БГ"]
    selected_cams = st.multiselect("Выберите камеру(ы)", data2['Адрес установки камеры'].unique(), data2['Адрес установки камеры'].unique())
    #st.write("Выбранные камеры:", selected_cams)
    if len(selected_cams) == 1:
        select_data = data2.loc[data2["Адрес установки камеры"].astype(str) == selected_cams[0]]
        st.subheader("Карта с камерами")
        st.map(select_data)
        st.write("Изображение с камеры")
        immag = Image.open("tko/1.png")
        st.image(immag)
    else:
        select_data = data2.loc[data2["Адрес установки камеры"].isin(selected_cams)]
        st.subheader("Карта с камерами")
        st.map(select_data)
        with st.expander("Изображения, полученные с камер"):
            for i in range(1,15):
                immag = Image.open(f"tko/{i}.png")
                st.image(immag)


if "Обнаружение отключений электрической энергии на территории" in add_selectbox3:
    st.header("Отключение электрической энергии")
    data2 = data[data["Type"] == "БД"]
    selected_cams = st.multiselect("Выберите камеру(ы)", data2['Адрес установки камеры'].unique(), data2['Адрес установки камеры'].unique())
    #st.write("Выбранные камеры:", selected_cams)
    if len(selected_cams) == 1:
        select_data = data2.loc[data2["Адрес установки камеры"].astype(str) == selected_cams[0]]
        st.subheader("Карта с камерами")
        st.map(select_data)
        st.write("Изображение с камеры")
        immag = Image.open("tko/1.png")
        st.image(immag)
    else:
        select_data = data2.loc[data2["Адрес установки камеры"].isin(selected_cams)]
        st.subheader("Карта с камерами")
        st.map(select_data)
        with st.expander("Изображения, полученные с камер"):
            for i in range(1,15):
                immag = Image.open(f"tko/{i}.png")
                st.image(immag)