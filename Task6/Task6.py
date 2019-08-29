import torch
import torch.nn as nn
import torch.nn.init as init
import pandas as pd 
import numpy as np
import random

# 参数
ratio = 0.8
epochs = 500
learning_rate = 0.01

# 加载数据，划分数据集
datas = torch.from_numpy(pd.read_csv('diabetes.csv').values)
length = len(datas)
indices = list(range(length))
random.shuffle(indices)
train_idx, test_idx = indices[:int(ratio*length)], indices[int(ratio*length):]
train, test = datas[train_idx], datas[test_idx]
x_train, y_train = train[:,:-1], train[:,-1]
x_test, y_test = test[:,:-1], test[:,-1]

# 搭建网络
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.bn = nn.BatchNorm1d(8)
        self.linear1 = torch.nn.Linear(8, 10)
        self.linear2 = torch.nn.Linear(10, 20)
        self.linear3 = torch.nn.Linear(20, 10)
        self.linear4 = torch.nn.Linear(10, 1)
    def forward(self, x):
        x = self.bn(x.type(torch.FloatTensor))
        x = torch.tanh(self.linear1(x))
        x = torch.tanh(self.linear2(x))
        x = torch.tanh(self.linear3(x))
        return self.linear4(x)

m_SGD = Model()
m_Momentum = Model()
m_Adagrad = Model()
m_RMSProp = Model()
m_Adam = Model()
models = [m_SGD, m_Momentum, m_Adagrad, m_RMSProp, m_Adam]

opt_SGD = torch.optim.SGD(m_SGD.parameters(), lr=learning_rate)
opt_Momentum = torch.optim.SGD(m_Momentum.parameters(), lr=learning_rate, momentum=0.9)
opt_Adagrad = torch.optim.Adagrad(m_Adagrad.parameters(), lr=learning_rate)
opt_RMSProp = torch.optim.RMSProp(m_RMSProp.parameters(), lr=learning_rate, alpha=0.9)
opt_Adam = torch.optim.Adam(m_Adam.parameters(), lr=learning_rate, betas=(0.9,0.99))
opts = [opt_SGD, opt_Momentum, opt_Adagrad, opt_RMSProp, opt_Adam]

criterion = torch.nn.BCEWithLogitsLoss()
loss_list = [0,0,0,0]

# 训练
for epoch in range(epochs):
    for model,opt,loss in zip(models,opts,loss_list):
        y_pred = model(x_train)
        loss = criterion(y_pred, y_train.float().view(-1, 1))
        opt.zero_grad()
        loss.backward()
        opt.step()

# 测试
for model in models:
    y_pred = model(x_test)
    preds = (y_pred >= 0).clone().detach()
    corrects = torch.sum(preds.byte() == y_test.view(-1,1).byte())
    acc = corrects.item()/len(x_test)
    print(acc)