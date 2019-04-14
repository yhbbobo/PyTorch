import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import  datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.utils.data.distributed

import math
import time

import os

# os.environ["CUDA_VISIBLE_DEVICES"]="0, 1, 2, 3, 4, 5 ,6,7"
# -------------------- Parse CLI arguments  --------------------
parser = argparse.ArgumentParser(description='Pytorch MNIST Example')
parser.add_argument('--batch_size', type=int, default=200,
                    help='batch size for training and testing (default: 100)')
parser.add_argument('--test_batch_size', type=int, default=200,
                    help='batch size for training and testing (default: 100)')
parser.add_argument('--epochs', type=int, default=25,
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.1,
                    help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.8,
                    help='SGD momentum (default: 0.9)')
# parser.add_argument('--cuda', action='store_true', default=False,
#                     help='Train on GPU with CUDA')
parser.add_argument('--gpunum', type=int, default=1,
                    help='number of epochs to train (default: 10)')
parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
opt = parser.parse_args()
lr = opt.lr
batch_size = opt.batch_size
epochs = opt.epochs
test_batch_size= opt.test_batch_size
momentum= opt.momentum
log_interval=opt.log_interval
gpulist = [range(opt.gpunum)]

# lr = 0.001
# batch_size = 100
# epochs = 10
# test_batch_size=100
# momentum=0.5
# log_interval=100
#输入数据变换操作
transform_list = [
                transforms.Resize(40),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32),
                transforms.ToTensor()
                ]
transform = transforms.Compose(transform_list)


torch.manual_seed(2018)

# Horovod: pin GPU to local rank.
# torch.cuda.set_device(hvd.local_rank())
torch.cuda.manual_seed(2018)

kwargs = {'num_workers': 4, 'pin_memory': True}

train_dataset = \
    datasets.MNIST('data-0', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))


train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, **kwargs)


test_dataset = \
    datasets.MNIST('data-0', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]))

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size,
                                        **kwargs)

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


# ResNet Module
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(1, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[1], 2)
        self.layer3 = self.make_layer(block, 64, layers[2], 2)
        self.layer4 = self.make_layer(block,128,layers[3])
        self.avg_pool = nn.AvgPool2d(7)
        self.fc = nn.Linear(128, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        global  COUNTER
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        #print("###############",out.size())
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


# model = ResNet(ResidualBlock, [2, 2, 2, 2])#.cuda()
model = torch.nn.DataParallel(ResNet(ResidualBlock, [2, 2, 2, 2]),device_ids=gpulist).cuda()

# Horovod: scale learning rate by the number of GPUs.
optimizer = optim.SGD(model.parameters(), lr=lr,
                      momentum=momentum)


criterion = nn.CrossEntropyLoss()

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        #loss = F.nll_loss(output, target)
        loss=criterion(output,target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), 0,
                100. * batch_idx / len(train_loader), loss.item()))

def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
for epoch in range(1, epochs + 1):
    start = time.time()
    train(epoch)
    train_time = time.time() - start
    print('epoch %d, training time: %.1f sec' % (epoch + 1, train_time))
    #测试过程
    # model.load_state_dict(torch.load('param_model.pkl'))
    test()



