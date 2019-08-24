import torch
import torch.nn as nn

# 生成实验数据
data = torch.ones(100,2)
xa = torch.normal(2*data, 1)
xb = torch.normal(-2*data, 1)
ya = torch.zeros(100)
yb = torch.ones(100)
x = torch.cat((xa,xb),0)
y = torch.cat((ya,yb),0)

# 模型参数
epochs = 500
learning_rate = 0.01

# pytorch基础实现
# 初始化权重
w = torch.randn(2,1,requires_grad=True)
b = torch.randn(1,1,requires_grad=True)

# 模型搭建，训练
for epoch in range(epochs):
    a = 1/(1+torch.exp(-(x.mm(w)+b)))
    j = -torch.mean(y*torch.log(a)+(1-y)*torch.log(1-a))
    j.backward()
    print(epoch, j)
    # 更新权重
    w -= learning_rate * w.grad
    b -= learning_rate * b.grad
    # 梯度置为0
    w.grad.zero_()
    b.grad.zero_()

# nn.module写网络结构
class LR(nn.Module):
    def __init__(self):
        super(LR, self).__init__()
        self.linear = nn.Linear(2,1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        return self.sigmoid(self.linear(x))

model = LR()

criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    optimizer.zero_grad()
    loss = criterion(model(x), y)
    print(epoch, loss)
    loss.backward()
    optimizer.step()
