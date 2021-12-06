import numpy as np
import torch
from torch import nn
import torch.utils.data as Data
import torchvision
from ops import *





class  train_model(nn.Module):
    def __init__(self):
        super(train_model, self).__init__()
        self.L1_1 = OctConv(2,64,3,1,1)
        self.L1_2 = deConv(64,64,3,1,1)
        self.L1_3 = deConv(64,64,3,1,1)
        self.L1_4 = OctConv(64,64,3,1,1)
        self.L1_5 = deConv(64,64,3,1,1)

        self.L2_5 = DilConv(64,64,3,1,2)
        self.L2_4 = DefConv(64,64,3,1,1)
        self.L2_3 = OctConv(64,64,3,1,1)
        self.L2_2 = deConv(64,64,3,1,1)
        self.L2_1 = DefConv(64,64,3,1,1)

        self.L3_5 = deConv(64,64,3,1,1)
        self.L3_4 = OctConv(64,64,3,1,1)
        self.L3_3 = Conv(64,64,3,1,1)
        self.L3_2 = DefConv(64,64,3,1,1)
        self.L3_1 = Zero(1)

        # self.relu = nn.ReLU()
        self.conv_last = nn.Conv2d(320, 128, 3, stride=1, padding=1, bias=False)
        self.conv_last_2 = nn.Conv2d(128, 1, 3, stride=1, padding=1, bias=False)


    def forward(self, x, l):
        inputs = torch.cat((x,l),1)
        l1_1 = self.L1_1(inputs)
        l1_2 = self.L1_2(l1_1)
        l1_3 = self.L1_3(l1_2)
        l1_4 = self.L1_4(l1_3)
        l1_5 = self.L1_5(l1_4)

        l2_5 = self.L2_5(l1_5)
        l3_5 = self.L3_5(l1_5)
        l5 = l2_5 + l3_5

        l2_4 = self.L2_4(l1_4)
        l3_4 = self.L3_4(l5)
        l4 = l2_4 + l3_4

        l2_3 = self.L2_3(l1_3)
        l3_3 = self.L3_3(l4)
        l3 = l2_3 + l3_3

        l2_2 = self.L2_2(l1_2)
        l3_2 = self.L3_2(l3)
        l2 = l2_2 + l3_2

        l2_1 = self.L2_1(l1_1)
        l3_1 = self.L3_1(l2)
        l1 = l2_1 + l3_1

        out = torch.cat((l1,l2,l3,l4,l5),1)
        out = self.conv_last(out)
        out = self.conv_last_2(out)
        out = x + out

        return out

