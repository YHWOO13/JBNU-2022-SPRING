import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import KFold

# CIFAR-100 데이터셋을 읽고 신경망에 입력할 형태로 변환
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
x_train=x_train.astype(np.float32)/255.0
x_test=x_test.astype(np.float32)/255.0
y_train=tf.keras.utils.to_categorical(y_train,100)
y_test=tf.keras.utils.to_categorical(y_test,100)

# 하이퍼 매개변수 설정
batch_size = 128
n_epoch = 50
k = 5  # k-겹 교차 검증
dropout_rate = [0.25, 0.25, 0.5]


# 드롭아웃 비율에 따라 교차 검증을 수행하고 정확률을 반환하는 함수
def cross_validation(dropout_rate):
    accuracy = []
    for train_index, val_index in KFold(k).split(x_train):
        # 훈련 집합과 검증 집합으로 분할
        xtrain, xval = x_train[train_index], x_train[val_index]
        ytrain, yval = y_train[train_index], y_train[val_index]

        # 신경망 모델 설계
        cnn = Sequential()
        cnn.add(Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3), padding='same'))
        cnn.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        cnn.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        cnn.add(MaxPooling2D(pool_size=(2, 2)))
        cnn.add(Dropout(dropout_rate[0]))
        cnn.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        cnn.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        cnn.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        cnn.add(MaxPooling2D(pool_size=(2, 2)))
        cnn.add(Dropout(dropout_rate[1]))
        
        # 분류
        cnn.add(Flatten())
        cnn.add(Dense(512, activation='relu'))
        cnn.add(Dropout(dropout_rate[2]))
        cnn.add(Dense(100, activation='softmax'))

        # 신경망 모델을 학습하고 평가하기
        cnn.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
        generator = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
        hist = cnn.fit(generator.flow(x_train, y_train, batch_size=batch_size), epochs=n_epoch,
                       validation_data=(x_test, y_test), verbose=2)
        accuracy.append(cnn.evaluate(xval, yval, verbose=0)[1])
        print(cnn.evaluate(xval, yval, verbose=0)[1])
    
    # 모델 구조 확인
    cnn.summary()
    # 모델 저장
    cnn.save('p3_201715529.h5')
    return accuracy


cross_validation(dropout_rate)

