# Copyright (c) THL A29 Limited, a Tencent company. All rights reserved.

import torch
import torch.nn as nn
from nncore.nn import (MODELS, build_linear_modules, build_model,
                       build_norm_layer,Parameter)


@MODELS.register()
class UniModalEncoder(nn.Module):

    def __init__(self,
                 dims=None,
                 p=0.5,
                 pos_cfg=None,
                 enc_cfg=None,
                 norm_cfg=None,
                 modal_type= None,
                 type_nums = 3,
                 **kwargs):
        super(UniModalEncoder, self).__init__()

        drop_cfg = dict(type='drop', p=p) if p > 0 else None
        # enc_dims = dims[0] if isinstance(dims, (list, tuple)) else dims
        enc_dims = dims[-1] if isinstance(dims, (list, tuple)) else dims
        

        # self.modal_type = torch.LongTensor([modal_type]).to("cuda")

        self.dropout = build_norm_layer(drop_cfg)
        # self.mapping = build_linear_modules(dims, **kwargs)
        # self.pos_enc = build_model(pos_cfg, enc_dims)
        # self.encoder = build_model(enc_cfg, enc_dims, bundler='sequential')
        # self.norm = build_norm_layer(norm_cfg, enc_dims)
        # self.type_emb = nn.Embedding(type_nums, enc_dims).to("cuda")

    def forward(self, x, **kwargs):
        if self.dropout is not None:
            x = self.dropout(x)
        # if self.mapping is not None:
        #     x = self.mapping(x)
        # if self.encoder is not None:
        #     pe = None if self.pos_enc is None else self.pos_enc(x)
        #     x = self.encoder(x, pe=pe, **kwargs)

        # if self.norm is not None:
        #     # pe = None if self.pos_enc is None else self.pos_enc(x)
        #     # emb_type = None if self.type_emb is None else self.type_emb(self.modal_type)
        #     x = self.norm(x)
        # if self.modal_type is not None:
        # pe = None if self.pos_enc is None else self.pos_enc(x)
        # emb_type = self.type_emb(self.modal_type)
        # x = x + pe + emb_type
        # x = x + emb_type 
        return x


@MODELS.register()
class CrossModalEncoder(nn.Module):

    def __init__(self,
                 dims=None,
                 fusion_type='sum',
                 pos_cfg=None,
                 enc_cfg=None,
                 norm_cfg=None,
                 **kwargs):
        super(CrossModalEncoder, self).__init__()
        assert fusion_type in ('sum', 'mean', 'concat')

        # map_dims = [2 * dims, dims] if fusion_type == 'concat' else [dims, dims // 2]
        # map_dims = [2 * dims , dims]
        map_dims = [2560 , 256]
        # map_dims = [8192, 256]

        # dims = 2816
        # self.fusion_type = fusion_type

        # self.pos_enc = build_model(pos_cfg, dims)
        # self.encoder = build_model(enc_cfg, dims)
        self.mapping = build_linear_modules(map_dims, **kwargs)
        # self.norm = build_norm_layer(norm_cfg, dims //2)
        self.norm = build_norm_layer(norm_cfg, 256)
        # self.w1 = Parameter(torch.tensor(0.5))

    def forward(self, a, b, c=None, **kwargs):

       
        d = torch.cat(( a,  b),dim=-1)
        x = self.mapping(d)
            
        # if self.encoder is not None:
        # #     d = torch.cat(( a,  b),dim=-1)
        #     pe = None if self.pos_enc is None else self.pos_enc(a)
        #     # pe = None
        #     # d = torch.cat((a, b, c),dim=-1)
        #     # d = self.encoder(d, pe=pe, **kwargs)

        #     # print(**kwargs)

        #     # x = self.mapping(d)
        #     a, b = self.encoder(a, b, c, pe=pe, **kwargs)
        # if self.fusion_type in ('sum', 'mean'):
        #     x = (a + b) / ((self.fusion_type == 'mean') + 1)
        #     # x = self.mapping(x)
        #     # x = (self.w1*a + (1 - self.w1)*b) / ((self.fusion_type == 'mean') + 1)
        # else:
        #     x = torch.cat((a, b), dim=-1)
        #     x = self.mapping(x)
        if self.norm is not None:
            x = self.norm(x)
        return x
