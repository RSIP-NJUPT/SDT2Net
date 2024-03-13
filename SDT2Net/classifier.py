import torch
import torch.nn as nn
from PadeSqt import PadeSqt
import MPN
import svPN
from covariance import Covariance


class Classifier(nn.Module):
    def __init__(self,num_classes=1000, input_dim=384, representationConfig={}):
        super(Classifier, self).__init__()
        self.re_type = representationConfig['type']
        normConfig = representationConfig['normalization']
        if self.re_type == 'second-order':
            self.representation = Covariance(**representationConfig['args'])
            if normConfig['type'] == 'svPN':
                self.normalization = svPN(**normConfig['args'])
            elif normConfig['type'] == 'MPN':
                if representationConfig['args']['cov_type'] == 'cross':
                    raise TypeError('Cross-covraiance is not supported when using MPN')
                self.normalization = MPN(**normConfig['args'])
            elif normConfig['type'] == 'PadeSqt':
                self.normalization = PadeSqt(**normConfig['args'])
            else:
                raise TypeError('{:} is not implemented'.format(normConfig['type']))
            self.output_dim = self.normalization.output_dim
            self.visual_fc = nn.Linear(self.output_dim, num_classes) if num_classes > 0 else nn.Identity()
        elif self.re_type == 'first-order':
            self.representation = nn.AdaptiveAvgPool1d(1)
            self.normalization = nn.Identity()
            self.visual_fc = nn.Linear(input_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.cls_fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        if self.re_type is not None:
            x = x[:, 1:]
            if self.re_type == 'first-order':
                x = x.transpose(-1, -2)
            elif self.re_type == 'second-order':
                x = x.transpose(-1, -2).unsqueeze(-1)
            x = self.representation(x)  # covariance
            x = self.normalization(x)  # PadeSqt
            x = x.view(x.size(0), -1)
            x = self.visual_fc(x)
            return x
        else:
            return x
