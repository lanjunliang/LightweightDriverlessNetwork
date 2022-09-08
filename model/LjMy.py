from distutils.command.bdist import show_formats
from turtle import forward
import torch.nn as nn
import torch
from torchvision import models
import torch.nn.functional as F
from collections import OrderedDict
# from misc.layer import Conv2d,MobileNet
# from misc.my_layer import ResidualBlock
# from misc.my_layer import PyDConv2d
import math
from thop import clever_format
from thop import profile
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from model.UNet import UNet

# 定义超参数
batch_size = 64
learning_rate = 1e-2
num_epoches = 20


class ghostnet_residualpyconv_cat(nn.Module):
    def __init__(self,inp, oup,new_channel, pyconv_dilation=[1,2,3,4],kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(ghostnet_residualpyconv_cat, self).__init__()

        self.shufflue=shuffle_chnls(new_channel)
        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, oup, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.pyconv_levels = [None] * len(pyconv_dilation)
        for i in range(len(pyconv_dilation)):
            self.pyconv_levels[i] = ghostnet_pyconv(oup=oup,new_channel=new_channel,dilation=pyconv_dilation[i])
        self.pyconv_levels = nn.ModuleList(self.pyconv_levels)
        # self.conv = nn.Sequential(
        #     nn.Conv2d(128,128, kernel_size, stride, kernel_size // 2, bias=False),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(inplace=True) if relu else nn.Sequential(),
        # )



    def forward(self,x):
        x1 = self.shufflue(x)
        x1=self.primary_conv(x)
        # print('primary:',x1.shape)
        out = []
        for level in self.pyconv_levels:
            # print('level:', level(x1).shape)
            out.append(level(x1))
            # print('each pyconv:',out)
        out = torch.cat(out, 1)
        # print(out.shape)
        # out=self.conv(out)
        # print('cheap:',out.shape)
        out = torch.cat([out, x1], dim=1)
        return out


class ghostnet_pyconv(nn.Module):
    def __init__(self,oup,new_channel , dilation,dw_size=3,relu=True):
        super(ghostnet_pyconv, self).__init__()
        self.cheap_operation = nn.Sequential(
                    # MobileNet(oup, [new_channel], dilation=dilation, kernel=3),
                    nn.Conv2d(oup,new_channel , dw_size, 1, padding=dilation,groups=new_channel,dilation=dilation,bias=False),
                    nn.BatchNorm2d(new_channel),
                    nn.ReLU(inplace=True) if relu else nn.Sequential(),
                    # nn.Conv2d(new_channel,new_channel,1,1,groups=1,bias=False),
                    # nn.BatchNorm2d(new_channel),
                    # nn.ReLU(inplace=True) if relu else nn.Sequential(),

                )
    def forward(self,x):
        x=self.cheap_operation(x)
        return x
class shuffle_chnls(nn.Module):
    def __init__(self,group) -> None:
        super().__init__()
        self.group=group//4
    def forward(self,x):
        bs,chnls,h,w=x.data.size()
        if chnls % self.group:
            return x
        chnls_per_group=chnls//self.group
        x=x.view(bs,self.group,chnls_per_group,h,w)
        x=torch.transpose(x,1,2).contiguous()
        x=x.view(bs,-1,h,w)
        return x
class ghostnet_residualpyconv(nn.Module):
    def __init__(self,inp, oup,new_channel, pyconv_dilation=[1,2,3,4],kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(ghostnet_residualpyconv, self).__init__()
        self.shufflule=shuffle_chnls(new_channel)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, oup, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.pyconv_levels = [None] * len(pyconv_dilation)
        for i in range(len(pyconv_dilation)):
            self.pyconv_levels[i] = ghostnet_pyconv(oup=oup,new_channel=new_channel,dilation=pyconv_dilation[i])
        self.pyconv_levels = nn.ModuleList(self.pyconv_levels)
        # self.conv=nn.Sequential(
        #     nn.Conv2d(oup, oup, kernel_size, stride, kernel_size // 2, bias=False),
        #     nn.BatchNorm2d(oup),
        #     nn.ReLU(inplace=True) if relu else nn.Sequential(),
        # )

    def forward(self,x):
        x1=self.shufflule(x)
        x1=self.primary_conv(x)
        # print('primary:',x1.shape)
        out = []
        for level in self.pyconv_levels:
            # print('level:', level(x1).shape)
            out.append(level(x1))
            
            # print('each pyconv:',out)
            # out+=level(x1)
        # out = torch.cat(out, 1)
        total=0
        for i in range(0,len(out)):
            total=total+out[i]
            # print('-------',total.shape)
            # print('total:',total.shape)
        # print('cheap:',out.shape)
        # total=self.conv(total)
        total= torch.cat([total,x1],dim=1)
        # print('residual-total:',total.shape)
        return total


class LjMy(nn.Module):
    def __init__(self,classes):
        super(LjMy, self).__init__()
        self.backbone=UNet(classes)
        # self.backbone=nn.Conv2d(3, 64, 3)
        self.residualpyconv1=nn.Sequential(
            ghostnet_residualpyconv(inp=64, oup=32,new_channel=32, pyconv_dilation=[1,2,3,6],kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True),
        )
        self.residualpyconv2=nn.Sequential(
            ghostnet_residualpyconv(inp=64,oup=64,new_channel=64,pyconv_dilation=[1,2,3,6],kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True))
        self.residualpyconv3=nn.Sequential(

            ghostnet_residualpyconv_cat(inp=128, oup=32, new_channel=32, pyconv_dilation=[1,2, 3,6], kernel_size=1,
                                    ratio=2, dw_size=3, stride=1, relu=True))
        self.out=nn.Conv2d(160, classes, 1)
    def forward(self,x):
        out1=self.backbone(x)
        # print(out1.shape)
        out2=self.residualpyconv1(out1)
        out3=self.residualpyconv2(out2)
        out4=self.residualpyconv3(out3)
        out5=self.out(out4)

        return out5



if __name__ == '__main__':
    model = LjMy(19)
    # print(model)
    dummy_input = torch.rand(1, 3, 1024, 1024)
    writer = SummaryWriter('log')
    with SummaryWriter(comment='LeNet') as w:
        w.add_graph(model, (dummy_input,))
    y=model(dummy_input)
    print(y.shape)
