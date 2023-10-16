# compress conv
import torch.nn as nn
import spconv
import torch
from opencood.utils import spconv_utils
from functools import partial


class CompressCore(nn.Module):
    def __init__(self, model_cfg, compressor, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        #self.in_channels = self.model_cfg['in_channels']
        #self.out_channels = self.model_cfg['out_channels']
        self.top_k = self.model_cfg['top_k']
        self.uniform_r = self.model_cfg['uniform_r']
        #self.sparse_shape = grid_size[::-1] + [1, 0, 0]
        norm_fn = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01)
        self.conv = compressor

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:

        """
        #print(batch_dict['record_len'])
        #voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        #batch_size = 1 # 应该是batch_dict['batch_size'] 之后有问题再看
        
        
        N, C, H, W = batch_dict['spatial_features'].shape
        features = batch_dict['spatial_features']
        
        '''
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features, 
            indices=voxel_coords.int(), 
            spatial_shape=self.sparse_shape, 
            batch_size=N
        )

        assert C == self.in_channels
        print(input_sp_tensor.dense())
        '''
        encoded = self.conv(features) # I_i^{down_arrow}
        compressed = torch.sum(encoded, dim=1)
        features_selected = int(H * W * self.top_k)
        compressed_flattened = compressed.flatten(1)
        values, indices = compressed_flattened.topk(features_selected, dim=1, largest=True)
        new_features_selected = int(features_selected * self.uniform_r)
        '''
        select = torch.Tensor(N, features_selected)
        select = select.uniform_(0, features_selected - 1).int()
        indices = indices[select]
        '''
        idx = torch.randperm(N)
        indices = indices[idx][:,:new_features_selected]
        
        sparse_features = torch.zeros((N,new_features_selected), device=features.device)
        sparse_indices = torch.zeros((N,new_features_selected,2), dtype=torch.int, device=features.device)
        for i in range(N):
            index_temp = indices[i]
            for j in range(len(index_temp)):
                coords = index_temp[j]
                h_temp = coords // W
                w_temp = coords % W
                
                sparse_indices[i,j,0]=h_temp
                sparse_indices[i,j,1]=w_temp

                sparse_features[i,j]=compressed[i,h_temp,w_temp]

        batch_dict['core_features'] = sparse_features
        batch_dict['core_indices'] = sparse_indices
        batch_dict['compressed_2d_feature'] = encoded
        '''
        encoded_spconv_tensor = batch_dict['encoded_spconv_tensor']
        spatial_features = encoded_spconv_tensor.dense()
        N, C, D, H, W = spatial_features.shape
        spatial_features = spatial_features.view(N, C * D, H, W)
        batch_dict['spatial_features'] = spatial_features
        batch_dict['spatial_features_stride'] = \
            batch_dict['encoded_spconv_tensor_stride']
        '''
        #print(sparse_features)
        #print(sparse_indices)
        return batch_dict
