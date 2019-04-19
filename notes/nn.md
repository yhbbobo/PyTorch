# 神经网络  
上一节学习了`autograd`,`nn`依赖`autograd`来定义model。一个`nn.Module`包含层、`forward(input)`方法，然后返回output。  
### 神经网络的典型训练程序如下：  
* 定义具有一些可学习参数（或权重）的神经网络
* 迭代输入数据集
* 通过网络处理输入
* 计算损失（输出距离正确多远）
* 将梯度反向传播，更新网络参数
* 通常使用简单的更新规则更新网络权重：`weight = weight - learning_rate * gradient`   
   
### TREE
* 一、定义网络
* 二、损失函数
* 三、Backprop  
* 四、更新权重

## 一、定义网络
```
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)
```
只需定义`forward`函数，and `backward` 自动定义`autograd`函数（计算gradient）。您可以在forward函数中使用任何Tensor操作。  
模型的可学习参数由`net.parameters()`返回 .   
```
params = list(net.parameters())
print(len(params))
print(params[0].size())  # conv1's .weight
```
让我们尝试一个随机的32x32输入。注意：此网络（LeNet）的预期输入大小为32x32。要在MNIST数据集上使用此网络，请将数据集中的图像调整为32x32。  
```
input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)
```
Zero the gradient buffers of all parameters and backprops with random gradients  
```
net.zero_grad()
out.backward(torch.randn(1, 10))
```
## 二、Loss Function
损失函数采用（输出，目标）输入对，并计算估计输出距目标的距离的值。

nn包下有几种不同的损失函数。一个简单的损失是：`nn.MSELoss`,它计算输入和目标之间的均方误差。

例如：  
```
output = net(input)
target = torch.randn(10)  # a dummy target, for example
target = target.view(1, -1)  # make it the same shape as output
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)
```
如果按照loss向后方向，使用其` .grad_fn`属性，将看到如下所示的计算图.  
```
input -> conv2d -> relu -> maxpool2d -> conv2d -> relu -> maxpool2d
      -> view -> linear -> relu -> linear -> relu -> linear
      -> MSELoss
      -> loss
```
因此，当我们调用时loss.backward()，整个图形会随着损失而区分，并且图形中的所有张量都requires_grad=True 将.grad使用渐变累积其Tensor。
  
为了说明，让我们向后退几步： 
``` 
print(loss.grad_fn)  # MSELoss
print(loss.grad_fn.next_functions[0][0])  # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU
```
## 三、Backprop  
要反向传播误差，我们所要做的就是`loss.backward()`。您需要清除现有梯度，否则梯度将累积到现有梯度。

现在我们call `loss.backward()`，看一下conv1在向后之前和之后的偏差梯度。  
```
net.zero_grad()     # zeroes the gradient buffers of all parameters

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)
```
## 四、更新权重
实践中使用的最简单的更新规则是随机梯度下降（SGD）：   
`weight = weight - learning_rate * gradient`  
我们可以使用简单的python代码实现它：   
```
learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)
```
但是，当您使用神经网络时，您希望使用各种不同的更新规则，例如SGD，Nesterov-SGD，Adam，RMSProp等。为了实现这一点，我们构建了一个小包：torch.optim它实现了所有这些方法。使用它非常简单：  
```
import torch.optim as optim

# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)

# in your training loop:
optimizer.zero_grad()   # zero the gradient buffers
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()    # Does the update
```
