import torch
import torch.nn as nn
import torch.nn.functional as F

class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

class ShiftConv2d0(nn.Module):
    def __init__(self, inp_channels, out_channels):
        super(ShiftConv2d0, self).__init__()    
        self.n_div = 5
        self.inp_channels = inp_channels * self.n_div
        self.out_channels = out_channels
        g = inp_channels

        conv3x3 = nn.Conv2d(inp_channels, out_channels, 3, 1, 1)
        mask = nn.Parameter(torch.zeros((self.out_channels, self.inp_channels, 3, 3)), requires_grad=False)
        mask[:, 1*g:2*g, 1, 2] = 1.0
        mask[:, 2*g:3*g, 1, 0] = 1.0
        mask[:, 3*g:4*g, 2, 1] = 1.0
        mask[:, 4*g:5*g, 0, 1] = 1.0
        mask[:, 5*g:, 1, 1] = 1.0
        self.w = conv3x3.weight
        self.b = conv3x3.bias
        self.m = mask

    def forward(self, x):
        y = F.conv2d(input=x, weight=self.w * self.m, bias=self.b, stride=1, padding=1) 
        return y

class ShiftConv2d1(nn.Module):
    def __init__(self, inp_channels, out_channels):
        super(ShiftConv2d1, self).__init__()    
        self.inp_channels = inp_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.zeros(inp_channels, 1, 5, 5), requires_grad=False)
        self.n_div = 5
        g = inp_channels // self.n_div
        self.weight[0*g:1*g, 0, 2, 4] = 1.0 ## left
        self.weight[1*g:2*g, 0, 2, 0] = 1.0 ## right
        self.weight[2*g:3*g, 0, 4, 2] = 1.0 ## up
        self.weight[3*g:4*g, 0, 0, 2] = 1.0 ## down
        self.weight[4*g:, 0, 2, 2] = 1.0 ## identity

        self.conv1x1 = nn.Conv2d(inp_channels, out_channels, 1)

    def forward(self, x):
        y = F.conv2d(input=x, weight=self.weight, bias=None, stride=1, padding=2, groups=self.inp_channels)
        y = self.conv1x1(y) 
        return y


class ShiftConv2d(nn.Module):
    def __init__(self, inp_channels, out_channels, conv_type='fast-training-speed'):
        super(ShiftConv2d, self).__init__()    
        self.inp_channels = inp_channels
        self.out_channels = out_channels
        self.conv_type = conv_type
        if conv_type == 'low-training-memory': 
            self.shift_conv = ShiftConv2d0(inp_channels, out_channels)
        elif conv_type == 'fast-training-speed':
            self.shift_conv = ShiftConv2d1(inp_channels, out_channels)
        else:
            raise ValueError('invalid type of shift-conv2d')

    def forward(self, x):
        y = self.shift_conv(x)
        return y


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class SCM(nn.Module):
    def __init__(self, inp_channels, out_channels, exp_ratio=4, act_type='relu'):
        super(SCM, self).__init__()
        self.exp_ratio = exp_ratio
        self.act_type  = act_type

        self.conv0 = ShiftConv2d(inp_channels, out_channels*exp_ratio)
        self.conv1 = ShiftConv2d(out_channels*exp_ratio, out_channels)


        if self.act_type == 'linear':
            self.act = None
        elif self.act_type == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif self.act_type == 'gelu':
            self.act = nn.GELU()
        else:
            raise ValueError('unsupport type of activation')

    def forward(self, x):
        if len(x.shape)>3:
            B, n, N, C = x.shape
            x = x.reshape(B, n*C, N).permute(1, 0, 2)
            y = self.conv0(x)
            y = self.act(y)
            y = self.conv1(y)
            y = y.reshape(B, n, C, N).permute(0, 1, 3, 2)
            return y
        else:
            x = torch.permute(x, [1, 0, 2])
            y = self.conv0(x)
            y = self.act(y)
            y = self.conv1(y)
            y = torch.permute(y, [1, 0, 2])
            return y

class EAB(nn.Module):
    def __init__(self, inp_channels, out_channels, exp_ratio=2, shifts=0, window_sizes=[4, 8, 12], shared_depth=1):
        super(EAB, self).__init__()
        self.exp_ratio = exp_ratio
        self.shifts = shifts
        self.window_sizes = window_sizes
        self.inp_channels = inp_channels
        self.out_channels = out_channels
        self.shared_depth = shared_depth
        
        modules_scm = {}
        modules_scm['scm_0'] = SCM(inp_channels=inp_channels, out_channels=out_channels, exp_ratio=exp_ratio)
        for i in range(shared_depth):
            modules_scm['scm_{}'.format(i+1)] = SCM(inp_channels=inp_channels, out_channels=out_channels, exp_ratio=exp_ratio)
        self.modules_scm = nn.ModuleDict(modules_scm)

    def forward(self, x):
        for i in range(1 + self.shared_depth):
            x = self.modules_scm['scm_{}'.format(i)](x) + x
        return x
