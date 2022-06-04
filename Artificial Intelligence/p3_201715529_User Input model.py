import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import KFold

# CIFAR-10 데이터셋을 읽고 신경망에 입력할 형태로 변환
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
x_train=x_train.astype(np.float32)/255.0
x_test=x_test.astype(np.float32)/255.0
y_train=tf.keras.utils.to_categorical(y_train,100)
y_test=tf.keras.utils.to_categorical(y_test,100)

class CNN():
  def __init__(self):
    self.cnn = Sequential()
    self.kernels = []
    self.Dropout = float(input('Dropout: '))
    self.num_block = int(input('Number of Block: '))
    self.kernel_count = 0
    for i in range(self.num_block):
      print(i+1,'th', end = ' ')
      self.num_conv = int(input('Number of Conv:'))

      for j in range(self.num_conv):
        self.num_kernel = int(input('Each Number of Kernel:'))
        self.kernels.append(self.num_kernel)

        # print(self.kernels[self.kernel_count])
        if self.kernel_count == 0:
          self.cnn.add(Conv2D(self.kernels[self.kernel_count],(3,3),activation='relu',input_shape=(32,32,3),padding ='same'))
        else:
          self.cnn.add(Conv2D(self.kernels[self.kernel_count],(3,3),activation='relu',padding ='same'))
        self.kernel_count += 1

      self.cnn.add(MaxPooling2D(pool_size=(2,2)))
      self.cnn.add(Dropout(self.Dropout))

    # 분류
    self.cnn.add(Flatten())
    self.cnn.add(Dense(512,activation='relu'))
    self.cnn.add(Dropout(self.Dropout))
    self.cnn.add(Dense(100,activation='softmax'))


def cross_validation():
    accuracy = []
    n_epoch = 50
    batch_size = 128

    # 신경망 모델 설계
    model = CNN()

    for train_index, val_index in KFold(5).split(x_train):
        xtrain, xval = x_train[train_index], x_train[val_index]
        ytrain, yval = y_train[train_index], y_train[val_index]

        # 신경망을 학습하고 정확률 평가
        model.cnn.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
        # hist = model.cnn.fit(xtrain,ytrain,batch_size=batch_size,epochs=n_epoch,verbose=0)
        generator = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
        hist = model.cnn.fit(generator.flow(x_train, y_train, batch_size=batch_size), epochs=n_epoch,
                             validation_data=(x_test, y_test), verbose=2)
        accuracy.append(model.cnn.evaluate(xval, yval, verbose=0)[1])
        print(model.cnn.evaluate(xval, yval, verbose=0)[1])
    return accuracy

acc=cross_validation()