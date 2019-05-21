# ImageFolder使用
ImageFolder假设所有的文件按文件夹保存好，每个文件夹下面存贮同一类别的图片，文件夹的名字为分类的名字。
```
ImageFolder(root,transform=None,target_transform=None,loader=default_loader)
```
* root : 在指定的root路径下面寻找图片 
* transform: 对PIL Image进行转换操作,transform 输入是loader读取图片返回的对象 
* target_transform :对label进行变换 
* loader: 指定加载图片的函数，默认操作是读取PIL image对象

## 例子
```
from torchvision.datasets import ImageFolder

dataset=ImageFolder('../data/dogcat/')

#对应文件夹的label
print(dataset.class_to_idx)
```
输出：
```
{'cat': 0, 'dog': 1}
```
   
```
#所有图片的路径和对应的label
print(dataset.imgs)
```
输出：
``` 
[(‘data/dogcat/cat/cat.12484.jpg’, 0), 
(‘data/dogcat/cat/cat.12485.jpg’, 0), 
(‘data/dogcat/cat/cat.12486.jpg’, 0), 
(‘data/dogcat/cat/cat.12487.jpg’, 0), 
(‘data/dogcat/dog/dog.12496.jpg’, 1), 
(‘data/dogcat/dog/dog.12497.jpg’, 1), 
(‘data/dogcat/dog/dog.12498.jpg’, 1), 
(‘data/dogcat/dog/dog.12499.jpg’, 1)]
```
   
```
#没有任何转变，所有返回的还是PIL Image对象
print(dataset[0][1]) #第二维度为1 ，表示label
print(dataset[0][0]) #第二维度为0，表示图片数据
```
输出:
``` 
0 
< PIL.Image.Image image mode=RGB size=497x500 at 0x7F25F3D31E10>
```

完整代码：
