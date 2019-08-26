@[TOC](（五）Pytorch实现L1，L2正则化以及Dropout)
# 1.了解知道Dropout原理
dropout，随机失活，是一种对深度神经网络的优化方法，可以防止过拟合、提升效率和在测试集上的效果。

 * 动机：引用自[百度百科](https://baike.baidu.com/item/%E9%9A%8F%E6%9C%BA%E5%A4%B1%E6%B4%BB/23293814?fromtitle=dropout&fromid=23294126&fr=aladdin)
> 随机失活是为解决深度神经网络的过拟合（overfitting）和梯度消失（gradient vanishing）问题而被提出的优化方法，其一般设想是在神经网络的学习过程中，随机将部分隐含层节点的权重归零，由于每次迭代受归零影响的节点不同，因此各节点的“重要性”会被平衡 [3-4]  。引入随机失活后，神经网络的每个节点都会贡献内容，不会出现少数高权重节点完全控制输出结果的情况，因此降低了网络的结构风险 [1]  。

 * 做法：参考[博客](https://blog.csdn.net/qq_40314507/article/details/89310870)
> Dropout的具体实现中，要求某个神经元节点激活值以一定的概率p被“丢弃”，即该神经元暂时停止工作。对于任意神经元，每次训练中都与一组随机挑选的不同的神经元集合共同进行优化，这个过程会减弱全体神经元之间的联合适应性，减少过拟合的风险，增加泛化能力。

# 2.用代码实现正则化(L1、L2、Dropout）
 * L2正则化是在原来的损失函数上加上模型权重参数的平方和。
 * L1正则化则是在原来的损失函数上加上权重参数的绝对值之和。
#

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
        
    model = Model()
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), weight_decay=1e-2, lr=learning_rate, momentum=0.9) # 默认为L2正则化
    
    # 训练
    for epoch in range(epochs):
        y_pred = model(x_train)
        loss = criterion(y_pred, y_train.float().view(-1, 1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # 测试
    y_pred = model(x_test)
    preds = (y_pred >= 0).clone().detach()
    corrects = torch.sum(preds.byte() == y_test.view(-1,1).byte())
    acc = corrects.item()/len(x_test)
    print(acc)
# 3.Dropout的numpy实现
    # dropout类
    class Dropout:
        def __init__(self, dropout_ratio=0.5):
            self.dropout_ratio=dropout_ratio
            self.mask=None
        def forward(self, x, train_flg=True):
            if train_flg:
                self.mask=np.random.rand(*x.shape) > self.dropout_ratio
                return x * self.mask
            else:
                return x * (1 - self.dropout_ratio)
        def backward(self, dout):
            return dout * self.mask

# 4.Pytorch中实现dropout
将网络换成

    # 搭建网络
    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.bn = nn.BatchNorm1d(8)
            self.linear1 = torch.nn.Linear(8, 10)
            self.dropout1 = torch.nn.Dropout(p=0.5)
            self.linear2 = torch.nn.Linear(10, 20)
            self.dropout2 = torch.nn.Dropout(p=0.5)
            self.linear3 = torch.nn.Linear(20, 10)
            self.dropout3 = torch.nn.Dropout(p=0.5)
            self.linear4 = torch.nn.Linear(10, 1)
        def forward(self, x):
            x = self.bn(x.type(torch.FloatTensor))
            x = torch.tanh(self.linear1(x))
            x = self.dropout1(x)
            x = torch.tanh(self.linear2(x))
            x = self.dropout2(x)
            x = torch.tanh(self.linear3(x))
            x = self.dropout3(x)
            return self.linear4(x)

# 5.参考资料
[PyTorch 中文文档](https://pytorch.apachecn.org/docs/1.0/)

