# 训练分类器

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
   
## 三、[完整例子](../code/classifier.ipynb) 
