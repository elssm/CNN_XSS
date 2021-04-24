import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import cv2
from sklearn.model_selection import train_test_split

df = pd.read_csv("XSS_dataset.csv",encoding="utf-8-sig")
# print(df.head())
sentences = df['Sentence'].values
# print(sentences[1])
print(len(sentences))
batch_size = 50
epochs = 10

def convert_to_ascii(sentence):
    sentence_ascii = []
    # print(len(sentence))

    for i in sentence:
        # print(ord(i))

        """Some characters have values very big e.d 8221 adn some are chinese letters
        I am removing letters having values greater than 8222 and for rest greater 
        than 128 and smaller than 8222 assigning them values so they can easily be normalized"""

        if (ord(i) < 8222):  # ” has ASCII of 8221

            if (ord(i) == 8217):  # ’  :  8217
                sentence_ascii.append(134)

            if (ord(i) == 8221):  # ”  :  8221
                sentence_ascii.append(129)

            if (ord(i) == 8220):  # “  :  8220
                sentence_ascii.append(130)

            if (ord(i) == 8216):  # ‘  :  8216
                sentence_ascii.append(131)

            if (ord(i) == 8217):  # ’  :  8217
                sentence_ascii.append(132)

            if (ord(i) == 8211):  # –  :  8211
                sentence_ascii.append(133)

            """
            If values less than 128 store them else discard them
            """
            if (ord(i) <= 128):
                sentence_ascii.append(ord(i))

            else:
                pass

    zer = np.zeros((10000)) #初始化一个长度为10000的向量

    for i in range(len(sentence_ascii)):
        zer[i] = sentence_ascii[i]
    # print(zer.shape)
    zer.shape = (100, 100) #将一维转为二维
    # print(zer.shape)

    #     plt.plot(image)
    #     plt.show()
    return zer


# send each sentence to be converted to ASCII


arr = np.zeros((len(sentences), 100, 100))

for i in range(len(sentences)):
    image = convert_to_ascii(sentences[i])

    x = np.asarray(image, dtype='float') #将二维里的数据类型转为float型
    image = cv2.resize(x, dsize=(100, 100), interpolation=cv2.INTER_CUBIC)
    image /= 128

    #     if i==1:
    #         plt.plot(image)
    #         plt.show()
    arr[i] = image

# print("Input data shape : ", arr.shape)

# Reshape data for input to CNN
data = arr.reshape(arr.shape[0],1,100, 100)
# print(data.shape)
y=df['Label'].values
#划分数据集
trainX, testX, trainY, testY = train_test_split(data,y, test_size=0.2, random_state=42)
trainX = torch.from_numpy(trainX)
trainX = DataLoader(trainX,batch_size=batch_size,shuffle=False)
testX = torch.from_numpy(testX)
testX = DataLoader(testX,batch_size=batch_size,shuffle=False)
trainY = torch.from_numpy(trainY)
trainY = DataLoader(trainY,batch_size=batch_size,shuffle=False)
testY = torch.from_numpy(testY)
testY = DataLoader(testY,batch_size=batch_size,shuffle=False)

class CNN_XSS_Net(nn.Module):
    def __init__(self):
        super(CNN_XSS_Net, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1,64,3),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Conv2d(64,128,3),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Conv2d(128,256,3),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),
            nn.ReLU(),
            # nn.Sigmoid(),

        )
        self.fc1 = nn.Linear(123904,256)
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 2)


    def forward(self,x):
        x = torch.as_tensor(x, dtype=torch.float32)
        cnn_res = self.cnn(x)
        # print(cnn_res.shape) #128*256*22*22
        cnn_res = cnn_res.view(cnn_res.size(0), -1)
        # print(cnn_res.shape) #128*123904
        f1 = self.fc1(cnn_res)
        f2 = self.fc2(f1)
        f3 = self.fc3(f2)
        f4 = self.fc4(f3)

        return f4


model = CNN_XSS_Net()
optimizer = optim.Adam(model.parameters(),1e-4)
criterion = nn.CrossEntropyLoss()


def train(model,trainX,trainY,optimizer,epochs):
    model.train()
    # l = int(len(trainX.dataset)/batch_size)
    # print(l)
    i = 0
    for data, target in zip(trainX,trainY):
        i+=1
        # print(type(data))
        # print(target)
        # 部署到device上
        # data,target = data.to(device),target.to(device)
        # 梯度初始化为0
        # print(data.shape)
        # print(target.shape)
        optimizer.zero_grad()
        # 预测
        output = model(data)
        # 计算损失
        loss = criterion(output, target)
        # 找到概率值最大的下标
        # pred = output.max(1,keepdim=True)
        # 反向传播
        loss.backward()
        optimizer.step()
        if i % 50 == 0:
            print("Train Epoch : {} \t Loss : {:.6f}".format(epochs, loss.item()))

def test_model(model,testX,testY):
    #模型验证
    model.eval()
    #正确率
    correct = 0.0
    #测试损失
    test_loss = 0.0
    with torch.no_grad():#不会计算梯度也不会进行反向传播
        for data,target in zip(testX,testY):
            # data,target = data.to(device),target.to(device)
            #测试数据
            output = model(data)
            #计算测试损失
            test_loss+=criterion(output,target).item()
            #找到概率值最大的下标
            pred = output.max(1,keepdim=True)[1] #值 索引
            #pred = torch.max(output,dim=1)
            #pred = output.argmax(dim=1)
            #累计正确率
            correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(testX.dataset)
        print("Test ---- Average loss : {:.4f},Accuracy : {:.3f}\n".format(test_loss,100.0*correct/len(testX.dataset)))

for epoch in range(epochs):
    train(model,trainX,trainY,optimizer,epoch)
    test_model(model,testX,testY)