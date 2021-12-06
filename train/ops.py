""" Operations """
import torch
import torch.nn as nn
#import v2.genotypes as gt
from dcn_v2 import DCN


OPS = {
    'none': lambda C_p, C, stride, affine: Zero(stride),
    #'avg_pool_3x3': lambda C, stride, affine: PoolBN('avg', C, 3, stride, 1, affine=affine),
    #'max_pool_3x3': lambda C, stride, affine: PoolBN('max', C, 3, stride, 1, affine=affine),
    'skip_connect': lambda C_p, C, stride, affine: Identity(),
    'sep_conv_3x3': lambda C_p, C, stride, affine: SepConv(C_p, C, 3, stride, 1),
    'dil_conv_3x3': lambda C_p, C, stride, affine: DilConv(C_p, C, 3, stride, 2, 2),
    'def_conv_3x3': lambda C_p, C, stride, affine: DefConv(C_p, C, 3, stride, 1),
    'conv_3x3': lambda C_p, C, stride, affine: Conv(C_p,C,3,stride,padding=1),
    'deconv_3x3': lambda C_p, C, stride, affine: deConv(C_p,C,3,stride,padding=1),
}

class OctConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, alpha=0.8, groups=1):
        super(OctConv, self).__init__()
        self.relu = nn.ReLU()
        self._pool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self._2h = nn.Conv2d(C_in, int(alpha * C_out),
                                   kernel_size, 1, padding, groups, bias=False)
        self._2l = nn.Conv2d(C_in, C_out - int(alpha * C_out),
                                   kernel_size, 1, padding, groups, bias=False)
        self.h2h = nn.Conv2d(int(alpha * C_out), C_out,
                                   kernel_size, 1, padding, groups, bias=False)
        self.l2h = nn.Conv2d(C_out - int(alpha * C_out),
                                   C_out, kernel_size, 1, padding, groups=1, bias=False)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.BN = nn.BatchNorm2d(C_out)

    def forward(self, x):
        x = self.relu(x)
        X_2l = self._pool(x)
        X_l = self._2l(X_2l)
        X_h = self._2h(x)
        X_l2h = self.l2h(X_l)
        X_h2h = self.h2h(X_h)
        X_l2h = self.upsample(X_l2h)
        out = X_h2h + X_l2h
        out = self.BN(out)
        return out



class Conv(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, groups=1):
    super(Conv, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(),
      nn.Conv2d(C_in, C_out, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False),
      nn.BatchNorm2d(C_out)
    )

  def forward(self, x):
    return self.op(x)

class deConv(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding):
    super(deConv, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(),
      nn.ConvTranspose2d(C_in, C_out, kernel_size=kernel_size, stride=stride, padding=padding, groups=1, bias=False),
      nn.BatchNorm2d(C_out),
    )

  def forward(self, x):
    return self.op(x)


class DefConv(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding):
    super(DefConv, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(),
      DCN(C_in, C_out, (kernel_size,kernel_size), stride=1, padding=1, dilation=1, deformable_groups=1),
      nn.BatchNorm2d(C_out),
    )
    #self.relu=nn.ReLU()
    #self.bn=nn.BatchNorm2d(C_out)
    #self.dcnv2=DCNv2(C_in, C_out, (kernel_size,kernel_size), stride=1, padding=1, dilation=1, deformable_groups=1)
    #self.conv_offset = nn.Conv2d(C_in, 1 * 2 * 3 * 3,
    #                        kernel_size=(3, 3),
    #                        stride=(1, 1),
    #                        padding=(1, 1),
    #                        bias=True)

    #self.conv_mask = nn.Conv2d(C_in, 1 * 1 * 3 * 3,
    #                      kernel_size=(3, 3),
    #                      stride=(1, 1),
    #                      padding=(1, 1),
    #                      bias=True)


  def forward(self, x):
    #out = self.relu(x)
    #offset = self.conv_offset(x)
    #mask = self.conv_mask(x)
    #mask = torch.sigmoid(mask)
    #output = self.dcnv2(out, offset, mask)
    #output = self.bn(output)
    
    return self.op(x)


class DilConv(nn.Module):
    """ (Dilated) depthwise separable conv
    ReLU - (Dilated) depthwise separable - Pointwise - BN

    If dilation == 2, 3x3 conv => 5x5 receptive field
                      5x5 conv => 9x9 receptive field
    """
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(C_in, C_out, kernel_size, stride, padding, dilation=dilation, groups=1,
                      bias=False),
            nn.BatchNorm2d(C_out)
        )

    def forward(self, x):
        return self.net(x)


class SepConv(nn.Module):
    """ Depthwise separable conv
    """
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super().__init__()
        self.net = nn.Sequential(
            #nn.ReLU(),
            #nn.Conv2d(C_in, C_in, kernel_size, stride, padding, groups=C_in, bias=False),
            #nn.Conv2d(C_in, C_out, 1, stride=1, padding=0, bias=False),
            #nn.BatchNorm2d(C_out)
            Conv(C_in, C_in, kernel_size, stride, padding, groups=C_in),
            Conv(C_in, C_out,1,1,0)
        )

    def forward(self, x):
        return self.net(x)



class Identity(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.dropout(x)
        return x


class Zero(nn.Module):
    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x * 0.

        # re-sizing by stride
        return x[:, :, ::self.stride, ::self.stride] * 0.

class Zero1(nn.Module):
    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x * 0.

        # re-sizing by stride
        return x[:, 1, ::self.stride, ::self.stride] * 0.






