import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
path = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))

model_path = os.path.join(path,'model/ciafar10_checkpoints/cifar_model_weights_30_epochs.pth')


def load(device):
    model = Fast_AT(device)
    model.load()
    model.name = 'fast_at'
    return model

class Fast_AT(torch.nn.Module):
    def __init__(self,device,model_path):
        torch.nn.Module.__init__(self)
        self.device = device
        self.model_path = model_path
        self.model = PreActResNet18().to(device)
        self._mean_torch = torch.tensor((0.4914, 0.4822, 0.4465)).view(3,1,1).to(device)
        self._std_torch = torch.tensor((0.2471, 0.2435, 0.2616)).view(3,1,1).to(device)

    def forward(self, x):
        input_var = (x.to(self.device) - self._mean_torch) / self._std_torch
        #labels = self.model(x)
        labels = self.model(input_var)
        return labels

    def load(self):
        checkpoint = torch.load(self.model_path)
        self.model.load_state_dict(checkpoint)
        self.model.float()
        self.model.eval()


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        if type(x[0])!=list:
            all_out = [x[0]]
        else:
            all_out = x[0]
        x = x[1]
        out = F.relu(self.bn1(x))
        out1 = out
        shortcut = self.shortcut(x) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        #out1 = out
        out = F.relu(self.bn2(out))
        out2 = out
        out = self.conv2(out)
        #out2 = out
        out += shortcut
        return all_out + [out1, out2], out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        print("PreActBottleneck")
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(PreActResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.bn = nn.BatchNorm2d(512 * block.expansion)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        all_out, out = self.layer1((out, out))
        all_out, out = self.layer2((all_out, out))
        all_out, out = self.layer3((all_out, out))
        all_out, out = self.layer4((all_out, out))
        out = F.relu(self.bn(out))
        all_out = all_out + [out]
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return all_out, out


def PreActResNet18():
    return PreActResNet(PreActBlock, [2,2,2,2])


if __name__ == '__main__':
    if not os.path.exists(model_path):
        if not os.path.exists(os.path.dirname(model_path)):
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
        url = 'https://drive.google.com/file/d/1XM-v4hqi9u8EDrQ2xdCo37XXcM9q-R07/view'
        print('Please download "{}" to "{}".'.format(url, model_path))
