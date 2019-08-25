@[TOC](（四）用PyTorch实现多层网络 )
# 1.引入模块，读取数据 

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.nn.init as init
    import pandas as pd 
    import numpy as np
    import random
    from sklearn.preprocessing import StandardScaler
    
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
    # 数据归一化
    # m = nn.BatchNorm1d(8, affine=False)
    # x_train, x_test = m(x_train), m(x_test)
    ss = StandardScaler()
    x_train = torch.tensor(ss.fit_transform(x_train))
    x_test = torch.tensor(ss.transform(x_test))

# 2.构建计算图（构建网络模型）
 # 搭建网络
    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.linear1 = torch.nn.Linear(8, 10)
            self.linear2 = torch.nn.Linear(10, 20)
            self.linear3 = torch.nn.Linear(20, 10)
            self.linear4 = torch.nn.Linear(10, 1)
        def forward(self, x):
            x = F.tanh(self.linear1(x))
            x = F.tanh(self.linear2(x))
            x = F.tanh(self.linear3(x))
            return self.linear4(x)
# 3.损失函数与优化器

    model = Model()
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

# 4.开始训练模型
    # 训练
    for epoch in range(epochs):
        y_pred = model(x_train)
        loss = criterion(y_pred, y_train)
        preds = torch.tensor(y_pred>=0)
        corrects = torch.sum(preds.byte() == y_train.view(-1,1).byte())
        acc = corrects.item()/len(x_train)
        print(acc)
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
# 5.对训练的模型预测结果进行评估
    # 测试
    y_pred = model(x_test)
    preds = torch.tensor(y_pred >= 0)
    corrects = torch.sum(preds.byte() == y_test.view(-1,1).byte())
    acc = corrects.item()/len(x_test)
    print(acc)
