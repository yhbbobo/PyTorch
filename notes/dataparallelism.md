# 数据并行  
我们将学习如何使用多个GPU DataParallel。

使用PyTorch非常容易使用GPU。您可以**将模型放在GPU上**：   
```
device = torch.device("cuda:0")
model.to(device)
```
然后，您可以将所有张量复制到GPU：  
```
mytensor = my_tensor.to(device)
```  
请注意，只是调用my_tensor.to(device)返回my_tensorGPU上的新副本 而不是重写my_tensor。您需要将其分配给新的张量并在GPU上使用该张量。  
  
在多个GPU上执行前向，后向传播是很自然的。但是，Pytorch默认只使用一个GPU。通过使用`DataParallel`以下方式并行运行模型，您可以轻松地在多个GPU上运行操作 ：  
```
model = nn.DataParallel(model)
```   

### [参考](../code/single_multigpus.ipynb)