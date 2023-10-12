# compress conv
import torch.nn as nn
import spconv
import torch

class CompressCore(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.out_channel= self.model_cfg['out_channel']
        self.top_k = self.model_cfg['top_k']
        self.uniform_r = self.model_cfg['uniform_r']

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:

        """
        voxel_features = batch_dict['spatial_features']
        info_features = torch.where(voxel_features > 1e-5, 1, 0)
        print("sparsity: "+str(torch.sum(info_features).item()/info_features.numel()))
       
        encoded_spconv_tensor = batch_dict['encoded_spconv_tensor']
        spatial_features = encoded_spconv_tensor.dense()
        N, C, D, H, W = spatial_features.shape
        spatial_features = spatial_features.view(N, C * D, H, W)
        batch_dict['spatial_features'] = spatial_features
        batch_dict['spatial_features_stride'] = \
            batch_dict['encoded_spconv_tensor_stride']
        return batch_dict
