from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose,\
                                    BatchNormalization, Activation, concatenate
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import utils
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import LambdaCallback
from sklearn.model_selection import train_test_split
from keras.callbacks import ReduceLROnPlateau

import matplotlib.pyplot as plt
import time
import random
import os
import numpy as np
from PIL import Image
import cv2

img_width = 256
img_height = 512
num_classes = 16

def color2index(color):
    index=-1
    if   (color[0]==0)   and (color[1]==255)  and (color[2]==0)  : index=0 # Тенисный мяч
    elif (color[0]==255)    and (color[1]==255)    and (color[2]==0)  : index=1 # Волейбольный мяч
    elif (color[0]==255)    and (color[1]==0)  and (color[2]==0)    : index=2 # Попрыгунчик
    elif (color[0]==0)  and (color[1]==0)    and (color[2]==255)    : index=3 # Пластиковый синий мяч
    elif (color[0]==0)    and (color[1]==255)  and (color[2]==255)  : index=4 # Герметик
    elif (color[0]==130)  and (color[1]==130)    and (color[2]==0)  : index=5 # Кронштейн
    elif (color[0]==0)  and (color[1]==130)  and (color[2]==130)    : index=6 # Манипулятор
    elif (color[0]==255) and (color[1]==0) and (color[2]==255) : index=7 # Куб с чёрными наклейками
    elif (color[0]==255)    and (color[1]==130)   and (color[2]==30) : index=8 # Ваза
    elif (color[0]==155)    and (color[1]==25) and (color[2]==25)    : index=9 # Контейнер со стикером
    elif (color[0]==155) and (color[1]==155)    and (color[2]==255)    : index=10 # Ферма
    elif (color[0]==255)    and (color[1]==240) and (color[2]==135) : index=11 # Палет с микросхемой
    elif (color[0]==255) and (color[1]==75)    and (color[2]==0) : index=12 # Бабина
    elif (color[0]==0) and (color[1]==145) and (color[2]==15)    : index=13 # Гусь
    elif (color[0]==90)    and (color[1]==0)  and (color[2]==90) : index=14 # Трактор
    else: index= 15
    return index 

def index2color(index2):
    index = np.argmax(index2) # Получаем индекс максимального элемента
    color=[]
    if   index == 0: color = [0, 255, 0]  # Тенисный мяч
    elif index == 1: color = [255, 255, 0]      # Волейбольный мяч
    elif index == 2: color = [255, 0, 0]      # Попрыгунчик
    elif index == 3: color = [0, 0, 255]      # Пластиковый синий мяч
    elif index == 4: color = [0, 255, 255]    # Герметик
    elif index == 5: color = [130, 130, 0]    # Кронштейн
    elif index == 6: color = [0, 130, 130]        # Манипулятор
    elif index == 7: color = [255, 0, 255]        # Куб с чёрными наклейками
    elif index == 8: color = [255, 130, 30]        # Ваза
    elif index == 9: color = [155, 25, 25]       # Контейнер со стикером
    elif index == 10: color = [155, 155, 255]       # Ферма
    elif index == 11: color = [255, 240, 135]       # Палет с микросхемой
    elif index == 12: color = [255, 75, 0]       # Бабина
    elif index == 13: color = [0, 145, 15]       # Гусь
    elif index == 14: color = [90, 0, 90]       # Трактор
    elif index == 15: color = [0, 0, 0]       # фон
    return color # Возвращаем цвет пикслея

def rgbToohe(y, num_classes): 
    y_shape = y.shape # Запоминаем форму массива для решейпа
    y = y.reshape(y.shape[0] * y.shape[1], 3) # Решейпим в двумерный массив
    yt = [] # Создаем пустой лист
    for i in range(len(y)): # Проходим по всем трем канала изображения
        yt.append(utils.to_categorical(color2index(y[i]), num_classes=num_classes)) # Переводим пиксели в индексы и преобразуем в OHE
    yt = np.array(yt) # Преобразуем в numpy
    yt = yt.reshape(y_shape[0], y_shape[1], num_classes) # Решейпим к исходныму размеру
    return yt # Возвращаем сформированный массив

def processImage(model, count = 1, n_classes = 6):
    indexes = np.random.randint(0, len(xTrain), count) # Получаем count случайных индексов
    fig, axs = plt.subplots(3, count, figsize=(25, 5)) #Создаем полотно из n графиков
    for i,idx in enumerate(indexes): # Проходим по всем сгенерированным индексам
        predict = np.array(model.predict(xTrain[idx].reshape(1, img_width, img_height, 3))) # Предиктим картику
        pr = predict[0] # Берем нулевой элемент из перидкта
        pr1 = [] # Пустой лист под сегментированную картинку из predicta
        pr2 = [] # Пустой лист под сегменитрованную картинку из yVal
        pr = pr.reshape(-1, n_classes) # Решейпим предикт
        yr = yTrain[idx].reshape(-1, n_classes) # Решейпим yVal
        for k in range(len(pr)): # Проходим по всем уровням (количесвто классов)
            pr1.append(index2color(pr[k])) # Переводим индекс в писксель
            pr2.append(index2color(yr[k])) # Переводим индекс в писксель
        pr1 = np.array(pr1) # Преобразуем в numpy
        pr1 = pr1.reshape(img_width, img_height,3) # Решейпим к размеру изображения
        pr2 = np.array(pr2) # Преобразуем в numpy
        pr2 = pr2.reshape(img_width, img_height,3) # Решейпим к размеру изображения
        img = Image.fromarray(pr1.astype('uint8')) # Получаем картику из предикта
        axs[0,i].imshow(img.convert('RGBA')) # Отображаем на графике в первой линии
        axs[1,i].imshow(Image.fromarray(pr2.astype('uint8'))) # Отображаем на графике во второй линии сегментированное изображение из yVal
        axs[2,i].imshow(Image.fromarray(xTrain[idx].astype('uint8'))) # Отображаем на графике в третьей линии оригинальное изображение        
    plt.show()

def dice_coef(y_true, y_pred):
    return (2. * K.sum(y_true * y_pred) + 1.) / (K.sum(y_true) + K.sum(y_pred) + 1.)

'''
  Функция создания сети
    Входные параметры:
    - num_classes - количество классов
    - input_shape - размерность карты сегментации
'''
def unet2(num_classes = 2, input_shape= (176, 320, 3)):
    img_input = Input(input_shape)                                         # Создаем входной слой с размерностью input_shape

    # Block 1
    x = Conv2D(16, (3, 3), padding='same', name='block1_conv1')(img_input) # Добавляем Conv2D-слой с 64-нейронами
    x = BatchNormalization()(x)                                            # Добавляем слой BatchNormalization
    x = Activation('relu')(x)                                              # Добавляем слой Activation

    x = Conv2D(16, (3, 3), padding='same', name='block1_conv2')(x)         # Добавляем Conv2D-слой с 64-нейронами
    x = BatchNormalization()(x)                                            # Добавляем слой BatchNormalization
    block_1_out = Activation('relu')(x)                                    # Добавляем слой Activation и запоминаем в переменной block_1_out

    x = MaxPooling2D()(block_1_out)                                        # Добавляем слой MaxPooling2D

    # Block 2
    x = Conv2D(32, (3, 3), padding='same', name='block2_conv1')(x)        # Добавляем Conv2D-слой с 128-нейронами
    x = BatchNormalization()(x)                                            # Добавляем слой BatchNormalization
    x = Activation('relu')(x)                                              # Добавляем слой Activation

    x = Conv2D(32, (3, 3), padding='same', name='block2_conv2')(x)        # Добавляем Conv2D-слой с 128-нейронами
    x = BatchNormalization()(x)                                            # Добавляем слой BatchNormalization
    block_2_out = Activation('relu')(x)                                    # Добавляем слой Activation и запоминаем в переменной block_2_out

    x = MaxPooling2D()(block_2_out)                                        # Добавляем слой MaxPooling2D

    # Block 3
    x = Conv2D(64, (3, 3), padding='same', name='block3_conv1')(x)        # Добавляем Conv2D-слой с 256-нейронами
    x = BatchNormalization()(x)                                            # Добавляем слой BatchNormalization
    x = Activation('relu')(x)                                              # Добавляем слой Activation

    x = Conv2D(64, (3, 3), padding='same', name='block3_conv2')(x)        # Добавляем Conv2D-слой с 256-нейронами
    x = BatchNormalization()(x)                                            # Добавляем слой BatchNormalization
    x = Activation('relu')(x)                                              # Добавляем слой Activation

    x = Conv2D(64, (3, 3), padding='same', name='block3_conv3')(x)        # Добавляем Conv2D-слой с 256-нейронами
    x = BatchNormalization()(x)                                            # Добавляем слой BatchNormalization
    block_3_out = Activation('relu')(x)                                    # Добавляем слой Activation и запоминаем в переменной block_3_out

    x = MaxPooling2D()(block_3_out)                                        # Добавляем слой MaxPooling2D

    # Block 4
    x = Conv2D(128, (3, 3), padding='same', name='block4_conv1')(x)        # Добавляем Conv2D-слой с 512-нейронами
    x = BatchNormalization()(x)                                            # Добавляем слой BatchNormalization
    x = Activation('relu')(x)                                              # Добавляем слой Activation

    x = Conv2D(128, (3, 3), padding='same', name='block4_conv2')(x)        # Добавляем Conv2D-слой с 256-нейронами
    x = BatchNormalization()(x)                                            # Добавляем слой BatchNormalization
    x = Activation('relu')(x)                                              # Добавляем слой Activation

    x = Conv2D(128, (3, 3), padding='same', name='block4_conv3')(x)        # Добавляем Conv2D-слой с 256-нейронами
    x = BatchNormalization()(x)                                            # Добавляем слой BatchNormalization
    block_4_out = Activation('relu')(x)                                    # Добавляем слой Activation и запоминаем в переменной block_4_out
    x = block_4_out 

    # UP 2
    x = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(x)    # Добавляем слой Conv2DTranspose с 256 нейронами
    x = BatchNormalization()(x)                                            # Добавляем слой BatchNormalization
    x = Activation('relu')(x)                                              # Добавляем слой Activation

    x = concatenate([x, block_3_out])                                      # Объединем текущий слой со слоем block_3_out
    x = Conv2D(64, (3, 3), padding='same')(x)                             # Добавляем слой Conv2D с 256 нейронами
    x = BatchNormalization()(x)                                            # Добавляем слой BatchNormalization
    x = Activation('relu')(x)                                              # Добавляем слой Activation

    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)                                            # Добавляем слой BatchNormalization
    x = Activation('relu')(x)                                              # Добавляем слой Activation

    # UP 3
    x = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(x)    # Добавляем слой Conv2DTranspose с 128 нейронами
    x = BatchNormalization()(x)                                            # Добавляем слой BatchNormalization
    x = Activation('relu')(x)                                              # Добавляем слой Activation

    x = concatenate([x, block_2_out])                                      # Объединем текущий слой со слоем block_2_out
    x = Conv2D(32, (3, 3), padding='same')(x)                             # Добавляем слой Conv2D с 128 нейронами
    x = BatchNormalization()(x)                                            # Добавляем слой BatchNormalization
    x = Activation('relu')(x)                                              # Добавляем слой Activation

    x = Conv2D(32, (3, 3), padding='same')(x) # Добавляем слой Conv2D с 128 нейронами
    x = BatchNormalization()(x) # Добавляем слой BatchNormalization
    x = Activation('relu')(x) # Добавляем слой Activation

    # UP 4
    x = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(x) # Добавляем слой Conv2DTranspose с 64 нейронами
    x = BatchNormalization()(x) # Добавляем слой BatchNormalization
    x = Activation('relu')(x) # Добавляем слой Activation

    x = concatenate([x, block_1_out])  # Объединем текущий слой со слоем block_1_out
    x = Conv2D(16, (3, 3), padding='same')(x) # Добавляем слой Conv2D с 64 нейронами
    x = BatchNormalization()(x) # Добавляем слой BatchNormalization
    x = Activation('relu')(x) # Добавляем слой Activation

    x = Conv2D(16, (3, 3), padding='same')(x) # Добавляем слой Conv2D с 64 нейронами
    x = BatchNormalization()(x) # Добавляем слой BatchNormalization
    x = Activation('relu')(x) # Добавляем слой Activation

    x = Conv2D(num_classes, (3, 3), activation='softmax', padding='same')(x)  # Добавляем Conv2D-Слой с softmax-активацией на num_classes-нейронов

    model = Model(img_input, x) # Создаем модель с входом 'img_input' и выходом 'x'

    # Компилируем модель 
    model.compile(optimizer=Adam(),
                  loss='categorical_crossentropy',
                  metrics=[dice_coef])
    # model.summary()
    return model # Возвращаем сформированную модель


def unet(num_classes = 2, input_shape= (176, 320, 3)):
    img_input = Input(input_shape)                                         # Создаем входной слой с размерностью input_shape

    # Block 1
    x = Conv2D(16, (3, 3), padding='same', name='block1_conv1')(img_input) # Добавляем Conv2D-слой с 64-нейронами
    x = BatchNormalization()(x)                                            # Добавляем слой BatchNormalization
    # x = Activation('relu')(x)                                              # Добавляем слой Activation

    # x = Conv2D(16, (3, 3), padding='same', name='block1_conv2')(x)         # Добавляем Conv2D-слой с 64-нейронами
    # x = BatchNormalization()(x)                                            # Добавляем слой BatchNormalization
    block_1_out = Activation('relu')(x)                                    # Добавляем слой Activation и запоминаем в переменной block_1_out

    x = MaxPooling2D()(block_1_out)                                        # Добавляем слой MaxPooling2D

    # Block 2
    x = Conv2D(32, (3, 3), padding='same', name='block2_conv1')(x)        # Добавляем Conv2D-слой с 128-нейронами
    # x = BatchNormalization()(x)                                            # Добавляем слой BatchNormalization
    # x = Activation('relu')(x)                                              # Добавляем слой Activation

    # x = Conv2D(32, (3, 3), padding='same', name='block2_conv2')(x)        # Добавляем Conv2D-слой с 128-нейронами
    x = BatchNormalization()(x)                                            # Добавляем слой BatchNormalization
    block_2_out = Activation('relu')(x)                                    # Добавляем слой Activation и запоминаем в переменной block_2_out

    x = MaxPooling2D()(block_2_out)                                        # Добавляем слой MaxPooling2D

    # Block 3
    x = Conv2D(64, (3, 3), padding='same', name='block3_conv1')(x)        # Добавляем Conv2D-слой с 256-нейронами
    # x = BatchNormalization()(x)                                            # Добавляем слой BatchNormalization
    # x = Activation('relu')(x)                                              # Добавляем слой Activation

    # x = Conv2D(64, (3, 3), padding='same', name='block3_conv2')(x)        # Добавляем Conv2D-слой с 256-нейронами
    # x = BatchNormalization()(x)                                            # Добавляем слой BatchNormalization
     #x = Activation('relu')(x)                                              # Добавляем слой Activation

    # x = Conv2D(64, (3, 3), padding='same', name='block3_conv3')(x)        # Добавляем Conv2D-слой с 256-нейронами
    x = BatchNormalization()(x)                                            # Добавляем слой BatchNormalization
    block_3_out = Activation('relu')(x)                                  # Добавляем слой Activation и запоминаем в переменной block_3_out
    x = block_3_out
    # x = MaxPooling2D()(block_3_out)                                        # Добавляем слой MaxPooling2D

    # Block 4
    # x = Conv2D(128, (3, 3), padding='same', name='block4_conv1')(x)        # Добавляем Conv2D-слой с 512-нейронами
    # x = BatchNormalization()(x)                                            # Добавляем слой BatchNormalization
    # x = Activation('relu')(x)                                              # Добавляем слой Activation

    # x = Conv2D(128, (3, 3), padding='same', name='block4_conv2')(x)        # Добавляем Conv2D-слой с 256-нейронами
    # x = BatchNormalization()(x)                                            # Добавляем слой BatchNormalization
    # x = Activation('relu')(x)                                              # Добавляем слой Activation

    # x = Conv2D(128, (3, 3), padding='same', name='block4_conv3')(x)        # Добавляем Conv2D-слой с 256-нейронами
    # x = BatchNormalization()(x)                                            # Добавляем слой BatchNormalization
    # block_4_out = Activation('relu')(x)                                    # Добавляем слой Activation и запоминаем в переменной block_4_out
    # x = block_4_out

    # UP 2
    # x = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(x)    # Добавляем слой Conv2DTranspose с 256 нейронами
    # x = BatchNormalization()(x)                                            # Добавляем слой BatchNormalization
    # x = Activation('relu')(x)                                              # Добавляем слой Activation

    # x = concatenate([x, block_3_out])                                      # Объединем текущий слой со слоем block_3_out
    # x = Conv2D(64, (3, 3), padding='same')(x)                             # Добавляем слой Conv2D с 256 нейронами
    # x = BatchNormalization()(x)                                            # Добавляем слой BatchNormalization
    # x = Activation('relu')(x)                                              # Добавляем слой Activation

    # x = Conv2D(64, (3, 3), padding='same')(x)
    # x = BatchNormalization()(x)                                            # Добавляем слой BatchNormalization
    # x = Activation('relu')(x)                                              # Добавляем слой Activation

    # UP 3
    x = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(x)    # Добавляем слой Conv2DTranspose с 128 нейронами
    x = BatchNormalization()(x)                                            # Добавляем слой BatchNormalization
    x = Activation('relu')(x)                                              # Добавляем слой Activation

    x = concatenate([x, block_2_out])                                      # Объединем текущий слой со слоем block_2_out
    x = Conv2D(32, (3, 3), padding='same')(x)                             # Добавляем слой Conv2D с 128 нейронами
    x = BatchNormalization()(x)                                            # Добавляем слой BatchNormalization
    x = Activation('relu')(x)                                              # Добавляем слой Activation

    # x = Conv2D(32, (3, 3), padding='same')(x) # Добавляем слой Conv2D с 128 нейронами
    # x = BatchNormalization()(x) # Добавляем слой BatchNormalization
    # x = Activation('relu')(x) # Добавляем слой Activation

    # UP 4
    x = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(x) # Добавляем слой Conv2DTranspose с 64 нейронами
    x = BatchNormalization()(x) # Добавляем слой BatchNormalization
    x = Activation('relu')(x) # Добавляем слой Activation

    x = concatenate([x, block_1_out])  # Объединем текущий слой со слоем block_1_out
    # x = Conv2D(16, (3, 3), padding='same')(x) # Добавляем слой Conv2D с 64 нейронами
    # x = BatchNormalization()(x) # Добавляем слой BatchNormalization
    # x = Activation('relu')(x) # Добавляем слой Activation

    x = Conv2D(16, (3, 3), padding='same')(x) # Добавляем слой Conv2D с 64 нейронами
    x = BatchNormalization()(x) # Добавляем слой BatchNormalization
    x = Activation('relu')(x) # Добавляем слой Activation

    x = Conv2D(num_classes, (3, 3), activation='softmax', padding='same')(x)  # Добавляем Conv2D-Слой с softmax-активацией на num_classes-нейронов

    model = Model(img_input, x) # Создаем модель с входом 'img_input' и выходом 'x'

    # Компилируем модель
    model.compile(optimizer=Adam(),
                  loss='categorical_crossentropy',
                  metrics=[dice_coef])
    # model.summary()
    return model # Возвращаем сформированную модель
modelWarehouse = unet(16, (img_width,img_height, 3))
# weights_path = get_file(
#             None,
#             origin='https://drive.google.com/uc?export=download&id=1l1bmU6wAtwbgefu66Uk2GxLvp0YuV-dS', extract=True)
weights_path = 'C:/Users/zyrik/Downloads/WarehouseWeights_test3.h5'
modelWarehouse.load_weights(weights_path)



# airplane = image.load_img("C:/Users/zyrik/Downloads/(1).jpg") #,target_size=(img_width, img_height)
#
# airplane = image.img_to_array(airplane)
# airplane = np.array(airplane)
# shape = airplane.shape
# airplane = cv2.resize(airplane, (img_width, img_height), interpolation= cv2.INTER_LINEAR)

def find(r1, g1, b1, r2, g2, b2, bl, tr1, tr2, m):
    deep_copy = copy.copy()  # .copy()
    # output = cv2.resize(deep_copy, dsize)
    # cv2.imshow('result', output)
    # cv2.waitKey(0)
    thresh = cv2.cvtColor(deep_copy, cv2.COLOR_BGR2RGB)
    # формируем начальный и конечный цвет фильтра
    h_min = np.array((r1, g1, b1), np.uint8)
    h_max = np.array((r2, g2, b2), np.uint8)
    thresh = cv2.inRange(thresh, h_min, h_max)
    # image_gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    thresh = cv2.medianBlur(thresh, 1 + bl * 2)
    ret, thresh = cv2.threshold(thresh, tr1, tr2, cv2.THRESH_BINARY)
    thresh = 255 - thresh
    thresh = cv2.bitwise_not(thresh)
    # output = cv2.resize(thresh, dsize)
    # # # print(r1, g1, b1, r2, g2, b2, bl, tr1, tr2)
    # cv2.imshow('result', output)
    # cv2.waitKey(0)
    shapes, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    for cnt in shapes:
        x, y, w, h = cv2.boundingRect(cnt)
        # print(x, y, w, h)
        if x == 0 and y == 0 and h == deep_copy.shape[1] and w == deep_copy.shape[0]:
            continue
        if w < 15 or h < 15:
            continue
        if h * w < 150:
            continue
        else:
            rects.append({"x": x, "y": y, "w": w, "h": h, "m": m})
            cv2.rectangle(copy_out, (x, y), (x + w, y + h), (0, 255, 0), 10)


camera = cv2.VideoCapture(0)

while True:
    return_value, cam = camera.read()
    cv2.imwrite('rare.jpg', cam)
    airplane = image.load_img("rare.jpg",target_size=(img_width, img_height))
    airplane = image.img_to_array(airplane)
    airplane = np.array(airplane)
    predict = np.array(modelWarehouse.predict(airplane.reshape(1, img_width, img_height, 3), verbose =0))
    pr = predict[0]
    pr1 = []
    pr = pr.reshape(-1, 16)
    for k in range(len(pr)):
      pr1.append(index2color(pr[k]))
    pr1 = np.array(pr1, np.uint8)
    pr1 = pr1.reshape(img_width, img_height, 3)[:, :, ::-1]
    # print (shape)
    # pr1 = cv2.resize(pr1, (shape[1],shape[0] ), interpolation = cv2.INTER_AREA)
    # cv2.imshow("ww", pr1)
    # cv2.waitKey(0)

    # image = cv2.imread(input())  # input()"" "K:\Downloads\(10).jpg"
    scale_percent = 150  # calculate the 50 percent of original dimensions
    copy = pr1.copy()
    copy_out = pr1.copy()
    width = int(copy.shape[1] * scale_percent / 100)
    height = int(copy.shape[0] * scale_percent / 100)
    dsize = (width, height)
    rects = []




    find(r1=0, g1=255, b1=0, r2=0, g2=255, b2=0, bl=8, tr1=0, tr2=255, m=1) #Тенисный мяч
    find(r1=255, g1=255, b1=0, r2=255, g2=255, b2=0, bl=8, tr1=0, tr2=255, m=1) #Волейбольный мяч
    find(r1=255, g1=0, b1=0, r2=255, g2=0, b2=0, bl=8, tr1=0, tr2=255, m=1) #Попрыгунчик
    find(r1=0, g1=0, b1=255, r2=0, g2=0, b2=255, bl=8, tr1=0, tr2=255, m=1) #Пластиковый синий мяч
    find(r1=0, g1=255, b1=255, r2=0, g2=255, b2=255, bl=8, tr1=0, tr2=255, m=1) #Герметик
    find(r1=130, g1=130, b1=0, r2=130, g2=130, b2=0, bl=8, tr1=0, tr2=255, m=1) #Кронштейн
    find(r1=0, g1=130, b1=130, r2=0, g2=130, b2=130, bl=8, tr1=0, tr2=255, m=1) #Манипулятор
    find(r1=255, g1=0, b1=255, r2=255, g2=0, b2=255, bl=8, tr1=0, tr2=255, m=1) #Куб с чёрными наклейками
    find(r1=255, g1=130, b1=30, r2=255, g2=130, b2=30, bl=8, tr1=0, tr2=255, m=1) #Ваза
    find(r1=155, g1=25, b1=25, r2=155, g2=25, b2=25, bl=8, tr1=0, tr2=255, m=1) #Контейнер со стикером
    find(r1=155, g1=155, b1=255, r2=155, g2=155, b2=255, bl=8, tr1=0, tr2=255, m=1) #Ферма
    find(r1=255, g1=240, b1=135, r2=255, g2=240, b2=135, bl=8, tr1=0, tr2=255, m=1) #Палет с микросхемой
    find(r1=255, g1=75, b1=0, r2=255, g2=75, b2=0, bl=8, tr1=0, tr2=255, m=1) #Бабина
    find(r1=0, g1=145, b1=15, r2=0, g2=145, b2=15, bl=8, tr1=0, tr2=255, m=1) #Гусь
    find(r1=90, g1=0, b1=90, r2=90, g2=0, b2=90, bl=8, tr1=0, tr2=255, m=1) #Трактор

    output = cv2.resize(copy_out, dsize)
    cv2.imwrite('good.jpg', copy_out)
    cv2.imshow("ww", output)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyWindow("ww")
        break

cv2.waitKey(0)
