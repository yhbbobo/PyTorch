# !/usr/bin/python
# -*- coding:utf-8 -*-
# author: Felix Fu

from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader

# 加上transforms
normalize = transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),  # 将图片转换为Tensor,归一化至[0,1]
    normalize
])

dataset = ImageFolder('../data/dogcat/', transform=transform)


# 对应文件夹的label
print(dataset.class_to_idx)

# 所有图片的路径和对应的label
# print(dataset.imgs)

# 没有任何转变，所有返回的还是PIL Image对象
print(dataset[0][1])  # 第二维度为1 ，表示label
print(dataset[0][0])  # 第二维度为0，表示图片数据

# dataloader是一个可迭代的对象，意味着我们可以像使用迭代器一样使用它 或者 or batch_datas, batch_labels in dataloader:
dataloader = DataLoader(dataset, batch_size=3, shuffle=True, num_workers=0, drop_last=False)

dataiter = iter(dataloader)
imgs, labels = next(dataiter)
print(imgs.size()) # batch_size, channel, height, weight
# 输出 torch.Size([3, 3, 224, 224])
