import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import torch.optim as optim
import tensorflow as tf
from tensorflow.keras.applications import vgg16
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img, ImageDataGenerator
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras import optimizers
from tensorflow.keras.utils import *

from PIL import Image
import requests
from io import BytesIO
import os
import random
import pickle
import tqdm
import itertools

import torch
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import torch.nn as nn


from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

dir = 'C:/Users/15/Desktop/DataSet/'
train_dir = 'C:/Users/15/Desktop/DataSet/[Dataset]Module11TrainReducedQualityAssuranceSystem/'

#  [Dataset]Module11TrainReducedQualityAssuranceSystem을 사용하여 훈련 시간을 줄일 수도 있습니다.
test_dir = 'C:/Users/15/Desktop/DataSet/[Dataset]Module11TestQualityAssuranceSystem/'

# 클래스는 이러한 이름을 가진 각 디렉토리의 폴더입니다.
classes = ['cracked','uncracked']

# 우리의 인공 신경망은 특정 크기 224X224 이미지를 처리합니다.
# 따라서 우리는 먼저 이미지 크기를 변환합니다.
data_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                      transforms.ToTensor()])

train_data = datasets.ImageFolder(train_dir, transform=data_transform)
print("Num training images:",len(train_data))

batch_size = 32
num_workers=0

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                           num_workers=num_workers, shuffle=True)
#test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                        #  num_workers=num_workers, shuffle=True)

# 일부 이미지를 표시해 보겠습니다.
dataiter = iter(train_loader)
images, labels = dataiter.next()
images = images.numpy() # 표시를 위해 이미지를 numpy로 변환

# 해당 라벨과 함께 배치 이미지를 보여줍니다.
fig = plt.figure(figsize=(25, 4))
for idx in np.arange(20):
    ax = fig.add_subplot(2, 20//2, idx+1, xticks=[], yticks=[])
    plt.imshow(np.transpose(images[idx], (1, 2, 0)))
    ax.set_title(classes[labels[idx]])

plt.show()
# 모델을 초기화 합니다.
vgg16 = models.vgg16(pretrained=True)

# 모델 구조 출력
print(vgg16)
# 입력 및 출력 feature 수를 출력해 보겠습니다.
print(vgg16.classifier[6].in_features)
print(vgg16.classifier[6].out_features)

# VGG 모델에는 컨볼루션 레이어, 최대 풀링 레이어 및 고밀도 레이어와 같은 다양한 레이어가 있습니다.
# 우리는 계산을 위해 모든 레이어를 사용할 필요는 없습니다.

# 모든 "특징" 레이어에 대한 훈련 동결
for param in vgg16.features.parameters():
    param.requires_grad = False

n_inputs = vgg16.classifier[6].in_features
last_layer = nn.Linear(n_inputs, len(classes))
vgg16.classifier[6] = last_layer



# 손실 함수 지정(범주형 교차 엔트로피)
criterion = nn.CrossEntropyLoss()

# optimizer는 stochastic gradient descent로 지정
# 학습률(learning rate) = 0.001
optimizer = optim.SGD(vgg16.classifier.parameters(), lr=0.001)

n_epochs = 30
larr = []

for epoch in range(1, n_epochs + 1):

    # 훈련 및 검증 손실 추적
    train_loss = 0.0

    ###################
    # train the model #
    ###################
    # 기본적으로 모델은 훈련으로 설정되어 있습니다.
    for batch_i, (data, target) in enumerate(train_loader):
        # CUDA를 사용할 수 있는 경우 텐서를 GPU로 이동
        optimizer.zero_grad()
        # 순방향 전달: 입력을 모델에 전달하여 예측된 출력을 계산합니다.
        output = vgg16(data)
        # 배치 손실을 계산
        loss = criterion(output, target)
        # 역방향 패스: 모델 매개변수에 대한 손실 기울기 계산
        loss.backward()
        # 단일 최적화 단계 수행(매개변수 업데이트)
        optimizer.step()
        # 훈련 손실 업데이트
        train_loss += loss.item()

        if epoch % 5 == 0:  # 지정된 수의 미니 배치마다 훈련 손실 출력
            print('Epoch %d, Batch %d loss: %.16f' %
                  (epoch, batch_i + 1, train_loss / 32))
            larr.append(train_loss / 32)
            train_loss = 0.0

plt.plot(larr)
plt.show()