from __future__ import print_function

import os, cv2, random, psutil
import numpy as np
import time

os.environ["THEANO_FLAGS"] = "mode=FAST_RUN, device=gpu1, floatX=float32, lib.cnmem=0, optimizer=fast_run"
os.environ["KERAS_BACKEND"] = "theano" #"tensorflow"

from keras.optimizers import SGD, RMSprop, Adadelta, Adam
from keras.models import Sequential, model_from_json

from keras.layers import Dense, Activation, Dropout, BatchNormalization
from keras.layers import Convolution2D, MaxPooling2D, Flatten
from keras.layers.advanced_activations import PReLU, LeakyReLU

startTime = time.time()
def print_Time(message):
    print(message,": {:0.2f}s".format(time.time()-startTime))

#---------------- Parameters ----------------
Train_path = 'train/'
Test_path = 'test/'
IMG_ROW = 224
IMG_COL = 224
RGB = 3         # 1:gray, 3:rgb
Epsilon = 1e-15
Margin = 0.01

Train_img_list = [Train_path+i for i in os.listdir(Train_path)]
Train_img_dog = [Train_path+i for i in os.listdir(Train_path) if 'dog' in i]
Train_img_cat = [Train_path+i for i in os.listdir(Train_path) if 'cat' in i]

mode = 'run'
if mode=='test':
    Train_img = Train_img_dog[:11500] + Train_img_cat[:11500]
    Test_img = Train_img_dog[11500:12500] + Train_img_cat[11500:12500]
elif mode=='run':
    Train_img = Train_img_dog + Train_img_cat
    Test_img = [Test_path+str(i)+'.jpg' for i in xrange(1,12501)]
random.shuffle(Train_img)

print(Test_img[:10])
print(len(Train_img_dog), len(Train_img_cat), len(Test_img))
print_Time('Get Image files')


def Read_Image(file_name):
    if RGB==1: img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
    elif RGB==3: img = cv2.imread(file_name, cv2.IMREAD_COLOR)
    return cv2.resize(img, (IMG_ROW, IMG_COL), interpolation=cv2.INTER_CUBIC)

def Get_data(Img_list):
    size = len(Img_list)
    data = np.ndarray((size, RGB, IMG_ROW, IMG_COL), dtype=np.float32)

    for i, img_name in enumerate(Img_list):
        image = Read_Image(img_name)
        data[i] = image.T / 255.
        if i % 2500==0:
            print('Reading Image {} of {} is {}'.format(i, size, img_name))
    print('Read {} images'.format(size))
    return data

def Get_label(Img_list):
    labels = np.zeros((len(Img_list), 1), dtype=np.float32)
    for i, img in enumerate(Img_list):
        if 'dog' in img:
            labels[i][0] = 1.0
        elif 'cat' in img:
            labels[i][0] = 0.0
    return labels

train = Get_data(Train_img)
train_labels = Get_label(Train_img)
test = Get_data(Test_img)
test_labels = Get_label(Test_img)

print('Training data shape:',train.shape)
print_Time('Preparing data')
#print(train[0])

def Classfier():
    model = Sequential()
    model.add(Convolution2D(128, 3 , 3, border_mode='same',
                            input_shape=(RGB, IMG_ROW, IMG_COL)))
    #model.add(Activation('relu'))
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(128, 3, 3, border_mode='same'))
    #model.add(Activation('relu'))
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(128, 3, 3, border_mode='same'))
    #model.add(Activation('relu'))
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(128, 3, 3, border_mode='same'))
    #model.add(Activation('relu'))
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(128, 2, 2, border_mode='same'))
    #model.add(Activation('relu'))
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(128, 2, 2, border_mode='same'))
    #model.add(Activation('relu'))
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(256))
    #model.add(Activation('relu'))
    model.add(LeakyReLU())
    model.add(Dropout(0.25))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    return model

def Save_model(Model, file_name):
    json_file = Model.to_json()
    json_name = 'models/'+file_name+'.json'
    open(json_name, 'w').write(json_file)
    Model.save_weights('models/'+file_name+'.h5', overwrite=True)
    print(' Saved model:',file_name)

def Load_model(file_name):
    json_file =  'models/'+file_name+'.json'
    model=model_from_json(open(json_file).read())
    model.load_weights('models/'+file_name+'.h5')
    print(' Loaded model:', file_name)
    return model

def Check_test(Model):
    predict = Model.predict(test)
    right = 0
    print(predict)
    for i in xrange(len(Test_img)):
        if predict[i][0] > 0.5 and test_labels[i] > 0.5:
            right += 1
        if predict[i][0] <= 0.5 and test_labels[i] < 0.5:
            right += 1
    print(' Check: {} / {}'.format(right, len(Test_img)))

def print_CSV(Model, Model_name, epoch):
    dir_name = 'excel/'+Model_name+str(epoch)+'.csv'
    csv = open(dir_name, 'w')
    csv.write('id,label\n')
    predict = Model.predict(test)
    predict = np.clip(predict, Margin, 1-Margin)
    for i, tst in enumerate(Test_img):
        id = int(''.join(x for x in tst if x.isdigit()))
        prob = predict[i][0]
        line = str(id) + ',' + str(prob) + '\n'
        csv.write(line)
    csv.close()

def Evaluate(y_true, y_pred):
    y_pred = np.clip(y_pred, Epsilon, 1 - Epsilon)
    out = -(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred))
    print('Log loss:',np.mean(out))


def main():
    process = psutil.Process(os.getpid())
    Model_name = 'keras-cnn-v2(adad)'
    #Model = Load_model(Model_name)
    Model = Classfier()
    Model.summary()

    Optimizer = Adadelta()
    Model.compile(optimizer=Optimizer, loss='MSE', metrics=['LL'])

    HI
    Epochs = 120
    BATCH_SIZE = 100
    print('Memory:',process.memory_info().rss,'bytes')
    print('Mode:',mode)
    print('Margin:',Margin)
    print('RGB:',RGB)
    print('Epochs:',Epochs)
    print('Batch size:', BATCH_SIZE)

    for i in xrange(Epochs):
        print('Epoch',i+1)
        Model.fit(train, train_labels, batch_size=BATCH_SIZE, verbose=1, nb_epoch=1, shuffle=True,
                  validation_split=0.05) #validation_data=(test, test_labels))
        Save_model(Model, Model_name)

        print_CSV(Model, Model_name, i)
        if mode=='test':
            predict = Model.predict(test)
            predict = np.clip(predict, Margin, 1-Margin)
            Evaluate(test_labels, predict)
            Check_test(Model)

    print_Time('Finish Training')
    print (time.strftime("%H:%M:%S"))
    print('****** END ******')

main()
