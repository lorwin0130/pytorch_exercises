@[TOC](（一）Pytorch的基本概念)
# Pytorch的基本概念
## 1.什么是Pytorch，为什么选择Pytorch？
 1. **Q:** 什么是Pytorch？
 **A:** Pytorch是一款深度学习框架。还有其他的深度学习框架：深度学习框架有PaddlePaddle、Tensorflow、Caffe、Theano、MXNet、Torch、Keras。[1]

2. **Q:** 为什么选择Pytorch？
**A:** Pytorch是基于Python，C ++和CUDA后端开发的，可用于Linux，macOS和Windows。
Pytorch的优点是在简洁、高效、易用。
更多有关不同深度学习框架的介绍：[如此多的深度学习框架，为什么我选择Pytorch](https://blog.csdn.net/broadview2006/article/details/79147351)

## 2.Pytorch的安装

环境：win10 + Python 3.6.2 + conda 4.5.11 + Pytorch


### 3.配置Python环境

[win10下Python3.6安装教程](https://blog.csdn.net/benben513624/article/details/80066136)

### 4.准备Python管理器

[安装anaconda教程](https://blog.csdn.net/lili_wuwu/article/details/82632162)

    #创建名为pytorch的环境
    conda create --name pytorch python=3 
    activate pytorch

### 5.通过命令行安装Pytorch
到pytorch官网选择合适的版本（本文选择的是cpu版本），https://pytorch.org/get-started/locally/
![111](https://img-blog.csdnimg.cn/20190807153813475.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MjI1NzgwMg==,size_16,color_FFFFFF,t_70)

    #命令行输入
    conda install pytorch-cpu torchvision-cpu -c pytorch

如果下载慢可以尝试[conda换国内源](https://blog.csdn.net/observador/article/details/83618540)
T_T刚听同学说conda国内源已经没有用了，直接去[清华镜像](https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/win-64/)直接下载就行。

测试一下：

    $ python
    Python 3.6.2 |Continuum Analytics, Inc.| (default, Jul 20 2017, 12:30:02) [MSC v.1900 64 bit (AMD64)] on win32
    Type "help", "copyright", "credits" or "license" for more information.
    >>> import torch
    >>> import pytorch
    >>> torch.__version__
    '1.1.0'
    >>>

## 6.Pytorch基础概念
* 张量：Tensorflow中数据的核心单元就是Tensor。张量包含了一个数据集合，这个数据集合就是原始值变形而来的，它可以是一个任何维度的数据。tensor的rank就是其维度。
* Pytorch对张量的操作：https://pytorch-cn.readthedocs.io/zh/latest/package_references/torch/
* Pytorch中的数学运算与Python中的numpy库类似
* Autograd模块：Pytorch中的自动微分模块
* Optim模块：Pytorch中的优化算法模块
* 神经网络模块：torch.nn


## 7.通用代码实现流程(实现一个深度学习的代码流程)
通用流程：
1. 设置训练参数
2. 加载训练集和测试集
3. 搭建网络
4. 选择优化器
5. 制定训练过程和测试过程
6. 主函数执行

根据教程[PyTorch: CNN实战MNIST手写数字识别](https://blog.csdn.net/m0_37306360/article/details/79311501)，进行的深度学习代码实现，代码如下：

    import torch
    import torch.nn as nn
    import torch.nn.functional as func
    import torch.optim as optim
    from torchvision import datasets, transforms
    from torch.autograd import Variable
    
    # 训练参数设置
    kernel_size = 5
    batch_size = 64
    epoch_num = 10
    
    # 下载MNIST数据集，并加载
    train_dataset = datasets.MNIST(root='./data/', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST(root='./data/', train=False, transform=transforms.ToTensor())
    
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    # 网络搭建
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 10, kernel_size=kernel_size)
            self.conv2 = nn.Conv2d(10, 20, kernel_size=kernel_size)
            self.mp = nn.MaxPool2d(2)
            self.fc = nn.Linear(320, 10)
    
        def forward(self, x):
            in_size = x.size(0)
            x = func.relu(self.mp(self.conv1(x)))
            x = func.relu(self.mp(self.conv2(x)))
            x = x.view(in_size, -1)
            x = self.fc(x)
            return func.log_softmax(x)
    
    # 生成实例，选择优化器
    model = Net()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    
    # 训练过程
    def train(epoch):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = model(data)
            loss = func.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 200 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
    
    # 测试过程
    def test():
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            data, target = Variable(data, volatile=True), Variable(target)
            output = model(data)
            test_loss += func.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
    
    # 主函数
    if __name__=="__main__":
        for epoch in range(1, epoch_num):
            train(epoch)
            test()
