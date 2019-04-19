# 自动微分
autograd包中是PyTorch中**所有神经网络的核心**。我们先了解下，然后去训练我们的第一个神经网络。

该autograd包为Tensors上的**所有操作**提供自动微分。它是一个逐个运行的框架，这意味着您的backprop由您的代码运行方式定义，并且每个迭代都可以不同。  
## TREE
* 1.Tensor的梯度相关知识
* 2.Gradients
  
## 一.Tensor
`torch.Tensor`是包的核心类。如果将其属性`.requires_grad`设置为`True`，则会开始跟踪其上的所有操作。完成计算后，您可以调用`.backward()`并自动计算所有梯度。该张量的梯度将累积到`.grad`属性中。  

要阻止张量跟踪历史记录，您可以调用`.detach()`它将其从计算历史记录中分离出来，并防止将来的计算被跟踪。  
   
要防止跟踪历史记录（和使用Memory），您还可以将代码块放入` with torch.no_grad():`。这在评估模型时尤其有用，因为模型可能具有可训练的参数`requires_grad=True` ，但我们不需要梯度。  

还有一个类,对于autograd非常重要的实现——`Function`。  

`Tensor`和`Function`互相联系，并构建一个非循环图，它编码完整的计算图。每个张量都有一个`.grad_fn`属性，该属性引用`Function`已创建的属性`Tensor`（除了用户创建的张量 ——grad_fn is None）。  

如果你想计算导数，你可以调用`.backward()` on a Tensor。如果Tensor是标量（即它包含一个元素数据），则不需要指定任何参数`backward()`，但是如果它有更多元素，则需要指定一个`gradient `匹配形状的张量的参数。   
  
```
import torch


#创建一个张量并设置requires_grad=True,为了跟踪计算  
x = torch.ones(2, 2, requires_grad=True)
print(x)

#做一个张量操作
y = x + 2
print(y)


#y是作为一个操作的结果创建的，所以它有一个grad_fn
print(y.grad_fn)

z = y * y * 3
out = z.mean()

print(z, out)

#.requires_grad_( ... )如果没有给出，输入标志默认为False。
a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)
```

## 二、Gradients
Let’s backprop now. Because out contains a single scalar, `out.backward()` is equivalent to ` out.backward(torch.tensor(1.))`.  
```
out.backward()

#Print gradients d(out)/dx
print(x.grad)

```