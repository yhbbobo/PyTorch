# nn(神经网络工具箱)  
autograd实现了自动微分系统，然而对千深度学习来说过于底层，nn模块是构建于autograd之上的神经网络模块。
除了nn之外，我们还会介绍神经网络中第用的工具，比如优化器optim 、初始化 init等。   
## 一、nn.Model  
使用autograd可实现深度学习橾型，但其抽象程度较低， 如果用其来实现深度学习模型， 则需要编写的代码量极大。 在这种情况下， torch.nn应运而生，其是专门为深度学习设计的模块。   
  
torch.nn的核心数据结构是Module, 它是一个抽象的概念 ，既可以表示神经网络中的某个层 {layer) , 也可以表示一个包含很多层的神经网络。在实际使用中 ，最常见的做法是**继承nn.Module**, 撰写自己的网络/层。  
## 二、常用的神经网络层 
### 1 图像相关层
图像相关层主要包括**卷积层** (Conv) 、池化层 (Pool) 等，这些层在实际使用中可分为—维 (1D) 、二维 (2D) 和三维 (3D) , **池化方式**又分为平均池化 (AvgPool) 、最大值池化 (MaxPool) 、自适应池化 (AdaptiveAvgPool) 等。卷积层除了常用的前向卷积外， 还 有逆卷积 (TransposeConv) 。  
### 2.激活函数  
### 3.损失函数  
### 4.优化器  
### 5.nn.functional
nn中还有一个很常用的模块 :nn.functional.nn中的大多数layer在functional中都有一个与之对应的函数。nn.functional中的函数和 nn.Module的主要区别在于：用 nn.Module实现的 layers是—个特殊的类，都是由classLayer (nn.Module) 定义，会自动提取可学习的参数;而 nn.functional中的函数更像是纯函数，由 def function（input）定义的。   
  
此时可能会问，应该什么时候使用 nn.Module, 什么时候 使用nn.functional呢?答案很简单，如果模型有可学习的参数，最好用nn.Module, 否则既可以使用nn.functional也可以使用nn.Module, 二者在性能上没有太大差异，具体的使用方式取决于个人喜好。由于激活函数 (ReLU、 sigmoid、 tanh) 、池化 (MaxPool) 等层没有可学习参数，可以使用对应的functional函数代替，而卷积、全连接等具有 可学习参数的网络建议使用 nn.Module。  
  
### 6.初始化策略  
PyTorch中的 nn.init橾块专门为 初始化设计，实现了常用的初始化策略。  

