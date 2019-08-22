import numpy as np
import torch

device = torch.device('cpu')
# 参数
batch_size, in_dim, hidden_dim, out_dim = 64, 1000, 100, 10

# 生成输入输出数据
## numpy
x_np = np.random.randn(batch_size, in_dim)
y_np = np.random.randn(batch_size, out_dim)
## pytorch
x_tc = torch.randn(batch_size, in_dim, device=device)
y_tc = torch.randn(batch_size, out_dim, device=device)

# 初始化权重
## numpy
w1_np = np.random.randn(in_dim, hidden_dim)
w2_np = np.random.randn(hidden_dim, out_dim)
## pytorch
w1_tc = torch.randn(in_dim, hidden_dim, device=device, requires_grad=True)
w2_tc = torch.randn(hidden_dim, out_dim, device=device, requires_grad=True)

# 学习率
learning_rate = 1e-6

# 更新轮数
for i in range(500):
    # forward
    ## numpy
    h = x_np.dot(w1_np)
    h_relu = np.maximum(h, 0)
    y_np_pred = h_relu.dot(w2_np)
    ## pytorch
    y_tc_pred = x_tc.mm(w1_tc).clamp(min=0).mm(w2_tc)

    # loss，print
    ## numpy
    loss_np = np.square(y_np_pred - y_np).sum()
    print(i, loss_np)
    ## pytorch
    loss_tc = (y_tc_pred - y_tc).pow(2).sum()
    print(i, loss_tc.item())

    # backward
    ## numpy
    d_y_np_pred = 2.0 * (y_np_pred - y_np)
    d_w2_np = h_relu.T.dot(d_y_np_pred)
    d_h_relu = d_y_np_pred.dot(w2_np.T)
    d_h = d_h_relu.copy()
    d_h[h < 0] = 0
    d_w1_np = x_np.T.dot(d_h)
    ## pytorch
    loss_tc.backward()

    # 更新权重
    w1_np -= learning_rate * d_w1_np
    w2_np -= learning_rate * d_w2_np
    ## pytorch
    with torch.no_grad():
        w1_tc -= learning_rate * w1_tc.grad
        w2_tc -= learning_rate * w2_tc.grad
        # 需要手动梯度置为0，否则梯度会进行累加
        w1_tc.grad.zero_()
        w2_tc.grad.zero_()


# # numpy实现线性回归


# # pytorch实现线性回归
# # 生成数据
# x = torch.rand(100) * 10
# y = 2 * x + torch.rand(100)
# w = torch.randn(1, requires_grad=True)
# b = torch.randn(1, requires_grad=True)
# y_pred = x.mm(w) + b
# learning_rate = 1e-6
# for i in range(100):
#     loss = (y_pred - y).pow(2).sum()
#     loss.backward()
#     with torch.no_grad():
#         w -= learning_rate * w.grad
#         b -= learning_rate * b.grad
#         w.grad.zero_()
#         b.grad.zero_()


# # pytorch简易神经网络
# import torch.nn.functional as F
# class Net():
#     def __init__(self):
#         self.w1 = torch.randn(1000, 100)
#         self.w2 = torch.randn(100, 10)

#     def forward(self, x):
#         x = F.relu(x.mm(self.w1)).mm(self.w2)
#         return F.log_softmax(x)
