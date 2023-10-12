# compress conv
import torch.nn as nn
import spconv

class CompressCore(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        print(self.model_cfg)
        self.out_channel= self.model_cfg['compress_core']['out_channel']
        self.top_k = self.model_cfg['compress_core']['top_k']
        self.uniform_r = self.model_cfg['compress_core']['uniform_r']

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:

        """
        print(batch_dict.keys())
        encoded_spconv_tensor = batch_dict['encoded_spconv_tensor']
        spatial_features = encoded_spconv_tensor.dense()
        N, C, D, H, W = spatial_features.shape
        spatial_features = spatial_features.view(N, C * D, H, W)
        batch_dict['spatial_features'] = spatial_features
        batch_dict['spatial_features_stride'] = \
            batch_dict['encoded_spconv_tensor_stride']
        return batch_dict
