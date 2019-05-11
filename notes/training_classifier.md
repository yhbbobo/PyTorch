# 训练分类器
### TREE
* 一、数据处理

## 一、数据处理
通常，当您必须处理图像，文本，音频或视频数据时，您可以使用标准的python包将数据加载到**numpy数组**中。然后你可以将这个数组转换成一个**torch.Tensor**。  
* 对于图像，Pillow，OpenCV等软件包很有用
* 对于音频，包括scipy和librosa
* 对于文本，无论是原始Python还是基于Cython的加载，还是NLTK和SpaCy都很有用  
  

特别是对于视觉，我们创建了一个名为的包`torchvision`，它包含用于常见数据集的数据加载器，如Imagenet，CIFAR10，MNIST等，以及用于图像的数据转换器，即 `torchvision.datasets`和`torch.utils.data.DataLoader`。  
## 二、训练图像分类器
* 1.使用加载和标准化CIFAR10训练和测试数据集 torchvision
* 2.定义卷积神经网络
* 3.定义损失函数
* 4.在训练数据上训练网络
* 5.在测试数据上测试网络
   
## 三、完整例子
jupyter例子   
```
%matplotlib inline
import torch
import torchvision
import torchvision.transforms as transforms

# 1. 准备数据
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='../data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='../data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 2.显示数据集中的一些图片
import matplotlib.pyplot as plt
import numpy as np

# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

# 3.定义网络
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
```
## [详细内容](https://github.com/fusimeng/pytorchexamples/blob/master/classifier.ipynb)   
