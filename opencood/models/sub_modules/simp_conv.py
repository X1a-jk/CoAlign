import numpy as np
import torch
import torch.nn as nn


class SimpleConvolution(nn.Module):
    def __init__(self, model_cfg, input_channels, output_channels):
        super().__init__()
        self.model_cfg = model_cfg

        self.block = nn.Sequential(
                                    nn.ZeroPad2d(1), \
                                    nn.Conv2d(input_channels, output_channels, kernel_seize=3, padding=1, bias=False), \
                                    nn.BatchNorm2d(output_channels, eps=1e-3, momentum=0.01), \
                                    nn.ReLU()
                                  )
            


    def forward(self, data_dict):
        spatial_features = data_dict['spatial_features']

        ups = []
        ret_dict = {}
        x = spatial_features

        for i in range(len(self.blocks)):
            x = self.blocks[i](x)

            stride = int(spatial_features.shape[2] / x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = x

            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)

        data_dict['spatial_features_2d'] = x # [N,C,100,352]

        return data_dict


    def get_multiscale_feature(self, spatial_features):
        """
        before multiscale intermediate fusion
        """
        feature_list = []
        x = spatial_features
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            feature_list.append(x)

        return feature_list

    def decode_multiscale_feature(self, x):
        """
        after multiscale interemediate fusion
        """
        ups = []
        for i in range(self.num_levels):
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x[i]))
            else:
                ups.append(x[i])
        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > self.num_levels:
            x = self.deblocks[-1](x)
        return x
        
