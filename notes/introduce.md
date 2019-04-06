
# 框架对比  
## 1.几个框架对比  
 1、Theano  
 由千Theano已经停止开发，不建议作为研究工具继续学习。  
 2、Tensorflow  
 不完美但最流行的深度学习框架，社区强大，适合生产 环境。  
 3、Keras  
 入门最简单，但是不够灵活，使用受限。  
 4、caffe/caffe2  
 文档不够完善，但性能优异，几乎全平台支持 (Caffe2) , 适合生产环境。  
 5、MxNet  
 文档略混乱，但分布式性能强大，语言支持最多，适合 AWS云平台使用。  
 6、CNTK   
 社区不够活跃，但是性能突出，擅长语音方面的相关研 究。  
 7、Pytorch  
 与Tensorflow相比，Pytorch是动态计算图设计思想。  
 动态图的思想直观明了，更符合人的思考过程。动态图的方式 使得我们可以任意修改前向传播，还可以随时查春变量的值。如果说静 态图框架好比C+ +'每次运行者限鸥箭译才行 (session.run) , 那么动 态图框架就是 Python, 动态执行，可以交互式查吾修改。动态图的这个 特性使律我们可以在IPython和儿 pyter Notebook上随时查香和修改变 最，十分灵活。  
 动态图带来的另外一个优势是调试更容易，在PyTorch中，代 码报错的地方，往往就是你写错代码的地方，而静态图需要先根据你的 代码生成Graph对象，然后在session.run ()时报错，这种报错几乎很
难找到对应的代码中莫正错误的地方。  
## 2.为什么选择pytorch  
PyTorch是当前难得的简洁优雅且离效快速的框架。在笔 者眼里， PyTorch达到目前深度学习框架的最高水平。当前开源的框架中，没有哪一个框架能够在**灵活性、易用性和速度**这三个方面有两个能 同时超过 PyTorch。   
* 简洁   
PyTorch的设计追求最少的封装，尽量避免重复造轮子。不像TensorFlow中充斥着session, graph、 operation、name_scope、 variable、 tensor、 layer等全新的概念， PyTorch的设 计遵循tensor->variable (autograd)->nn.Module三个由低到高的 抽象层次，分别代表高维数组(张量)、自动求导(变最)和神经网络
(层/模块)，而且这三个抽象之间联系紧密，可以同时进行修改和操
作。    
简洁的设计带来的另外一个好处就是代码易千理解。 PyTorch 的源码只有Tensor-Flow的十分之—左右，更少的抽象、更直观的设计 使倡 PyTorch的源码十分易千阅读。在笔者眼里， PyTorch的源码甚至 比许多框架的文档更容易理解。  

* 速度   
PyTorch的灵活性不以速度为代价，在许多评测中， PyTorch的速度表现胜过TensorFlo对打Keras等框架  
* 易用  
PyTorch是所有的框架中面向对象设计的最优雅的—
个。 PyTorch的面向对象的接口设计来源于Torch, 而 Torch的接口设计 以灵活易用而著称， Keras作者最初就是受Torch的启发才开发了 Keras。 PyTorch继承了 Torch的衣钵，尤其是API的设计和橾块的接口都与Torch高度—致。 PyTorch的设计最符合人们的思维，它让用户尽可 能地专注千实现自己的想法，即所思即所得，不需要考虑太多关于框架本身的束缚。  
* 活跃的社区   
PyTorch提供了完整的文档，循序渐进的指
南，作者亲自维护的论坛。  
### 3.ONNX  
Facebook,微软宣布，推出Open Neural Network
Exchange (ONNX, 开放神经网络交换)格式，这是一个用千表示深
度学习模型的标准，可使模型在不同框架之间进行转移。 ONNX是迈向开放生态系统的第—步， ONNX目 前支持 PyTorch、 Caffe2和 CNTK,未来会支持更多的框架。除了 Facebook和微软，ARM、 IBM、华为和英特尔、高通也宣布支持ONNX。  


