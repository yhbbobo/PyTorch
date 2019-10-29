# PyTorch 中，nn 与 nn.functional 有什么区别？
其实这两个是差不多的，不过一个包装好的类，一个是可以直接调用的函数。我们可以去翻这两个模块的具体实现代码，我下面以卷积Conv1d为例。 
     
首先是torch.nn下的Conv1d:     
```python
class Conv1d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _single(kernel_size)
        stride = _single(stride)
        padding = _single(padding)
        dilation = _single(dilation)
        super(Conv1d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _single(0), groups, bias)

    def forward(self, input):
        return F.conv1d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
```
这是torch.nn.functional下的conv1d:   
```python
def conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1,
           groups=1):
    if input is not None and input.dim() != 3:
        raise ValueError("Expected 3D tensor as input, got {}D tensor instead.".format(input.dim()))

    f = ConvNd(_single(stride), _single(padding), _single(dilation), False,
               _single(0), groups, torch.backends.cudnn.benchmark,
               torch.backends.cudnn.deterministic, torch.backends.cudnn.enabled)
    return f(input, weight, bias)
```
可以看到torch.nn下的Conv1d类在forward时调用了nn.functional下的conv1d，当然最终的计算是通过C++编写的THNN库中的ConvNd进行计算的，因此这两个其实是互相调用的关系。你可能会疑惑为什么需要这两个功能如此相近的模块，其实这么设计是有其原因的。如果我们只保留nn.functional下的函数的话，在训练或者使用时，我们就要手动去维护weight, bias, stride这些中间量的值，这显然是给用户带来了不便。而如果我们只保留nn下的类的话，其实就牺牲了一部分灵活性，因为做一些简单的计算都需要创造一个类，这也与PyTorch的风格不符。下面我举几个例子方便各位理解：    
```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(F.dropout(self.fc2(x), 0.5))
        x = F.dropout(self.fc3(x), 0.5)
        return x
```
以一个最简单的三层网络为例。需要维持状态的，主要是三个线性变换，所以在构造Module是，定义了三个nn.Linear对象，而在计算时，relu,dropout之类不需要保存状态的可以直接使用。   

注：dropout的话有个坑，需要设置自行设定training的state。
### 两者的相同之处
#### 1  
nn.Xxx和nn.functional.xxx的实际功能是相同的，即nn.Conv2d和nn.functional.conv2d 都是进行卷积，nn.Dropout 和nn.functional.dropout都是进行dropout，。。。。。； 
#### 2   
运行效率也是近乎相同。nn.functional.xxx是函数接口，而nn.Xxx是nn.functional.xxx的类封装，并且nn.Xxx都继承于一个共同祖先nn.Module。这一点导致nn.Xxx除了具有nn.functional.xxx功能之外，内部附带了nn.Module相关的属性和方法，例如train(), eval(),load_state_dict, state_dict 等。
### 两者的差别之处
#### 1、两者的调用方式不同    
nn.Xxx 需要先实例化并传入参数，然后以函数调用的方式调用实例化的对象并传入输入数据。
```python
inputs = torch.rand(64, 3, 244, 244)
conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
out = conv(inputs)
```
nn.functional.xxx同时传入输入数据和weight, bias等其他参数 。
```python
weight = torch.rand(64,3,3,3)
bias = torch.rand(64) 
out = nn.functional.conv2d(inputs, weight, bias, padding=1)
```
#### 2 
nn.Xxx继承于nn.Module， 能够很好的与nn.Sequential结合使用， 而nn.functional.xxx无法与nn.Sequential结合使用。 
```python
fm_layer = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.2)
  )
```
#### 3、
nn.Xxx不需要你自己定义和管理weight；而nn.functional.xxx需要你自己定义weight，每次调用的时候都需要手动传入weight, 不利于代码复用。使用nn.Xxx定义一个CNN 。
```python
class CNN(nn.Module):  
    def __init__(self):
        super(CNN, self).__init__()
        
        self.cnn1 = nn.Conv2d(in_channels=1,  out_channels=16, kernel_size=5,padding=0)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5,  padding=0)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        
        self.linear1 = nn.Linear(4 * 4 * 32, 10)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.maxpool1(self.relu1(self.cnn1(x)))
        out = self.maxpool2(self.relu2(self.cnn2(out)))
        out = self.linear1(out.view(x.size(0), -1))
        return out
```
使用nn.function.xxx定义一个与上面相同的CNN。
```python
class CNN(nn.Module): 
    def __init__(self):
        super(CNN, self).__init__()
        
        self.cnn1_weight = nn.Parameter(torch.rand(16, 1, 5, 5))
        self.bias1_weight = nn.Parameter(torch.rand(16))
        
        self.cnn2_weight = nn.Parameter(torch.rand(32, 16, 5, 5))
        self.bias2_weight = nn.Parameter(torch.rand(32))
        
        self.linear1_weight = nn.Parameter(torch.rand(4 * 4 * 32, 10))
        self.bias3_weight = nn.Parameter(torch.rand(10))
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = F.conv2d(x, self.cnn1_weight, self.bias1_weight)
        out = F.relu(out)
        out = F.max_pool2d(out)
        
        out = F.conv2d(x, self.cnn2_weight, self.bias2_weight)
        out = F.relu(out)
        out = F.max_pool2d(out)
        
        out = F.linear(x, self.linear1_weight, self.bias3_weight)
        return out
```
上面两种定义方式得到CNN功能都是相同的，至于喜欢哪一种方式，是个人口味问题，但PyTorch官方推荐：具有学习参数的（例如，conv2d, linear, batch_norm)采用nn.Xxx方式，没有学习参数的（例如，maxpool, loss func, activation func）等根据个人选择使用nn.functional.xxx或者nn.Xxx方式。但关于dropout，个人强烈推荐使用nn.Xxx方式，因为一般情况下只有训练阶段才进行dropout，在eval阶段都不会进行dropout。使用nn.Xxx方式定义dropout，在调用model.eval()之后，model中所有的dropout layer都关闭，但以nn.function.dropout方式定义dropout，在调用model.eval()之后并不能关闭dropout。
```python
class Model1(nn.Module):    
    def __init__(self):
        super(Model1, self).__init__()
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        return self.dropout(x)
    
    
class Model2(nn.Module):

    def __init__(self):
        super(Model2, self).__init__()

    def forward(self, x):
        return F.dropout(x)
    

m1 = Model1()
m2 = Model2()
inputs = torch.rand(10)
print(m1(inputs))
print(m2(inputs))
print(20 * '-' + "eval model:" + 20 * '-' + '\r\n')
m1.eval()
m2.eval()
print(m1(inputs))
print(m2(inputs))
```
输出：<img src="https://pic3.zhimg.com/50/v2-b2b202a21c13bdacba02a9d69d602b77_hd.jpg" data-rawwidth="693" data-rawheight="264" data-size="normal" data-caption="" data-default-watermark-src="https://pic2.zhimg.com/50/v2-33b882501c7228888e0e4e6c44f984f2_hd.jpg" class="origin_image zh-lightbox-thumb" width="693" data-original="https://pic3.zhimg.com/v2-b2b202a21c13bdacba02a9d69d602b77_r.jpg"/>

从上面输出可以看出m2调用了eval之后，dropout照样还在正常工作。当然如果你有强烈愿望坚持使用nn.functional.dropout，也可以采用下面方式来补救。 class Model3(nn.Module):
```python
def __init__(self):
        super(Model3, self).__init__()

def forward(self, x):
        return F.dropout(x, training=self.training)
```


## Reference
[1] https://www.zhihu.com/question/66782101   
