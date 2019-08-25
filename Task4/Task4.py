import torch
import torch.nn as nn
import torch.nn.init as init
import pandas as pd 
import numpy as np
import random
'''
https://blog.csdn.net/weixin_41710583/article/details/90379826
https://www.jianshu.com/p/342ff49a3656
'''
# 参数
ratio = 0.8
epochs = 500
learning_rate = 0.1

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
        self.linear1 = nn.Linear(8, 10)
        self.linear2 = nn.Linear(10, 20)
        self.linear3 = nn.Linear(20, 10)
        self.linear4 = nn.Linear(10, 1)
    def forward(self, x):
        x = self.bn(x.type(torch.FloatTensor))
        x = torch.tanh(self.linear1(x))
        x = torch.tanh(self.linear2(x))
        x = torch.tanh(self.linear3(x))
        return self.linear4(x)

model = Model()
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

# 训练
for epoch in range(epochs):
    y_pred = model(x_train)
    loss = criterion(y_pred, y_train.float().view(-1, 1))
    preds = (y_pred >= 0).clone().detach()
    corrects = torch.sum(preds.byte() == y_train.view(-1,1).byte())
    acc = corrects.item()/len(x_train)
    print(acc)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 测试
y_pred = model(x_test)
preds = (y_pred >= 0).clone().detach()
corrects = torch.sum(preds.byte() == y_test.view(-1,1).byte())
acc = corrects.item()/len(x_test)
print(acc)