# Copyright (c) THL A29 Limited, a Tencent company. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
from nncore.nn import MODELS, build_linear_modules, build_loss


@MODELS.register()
class BoundaryHead_contrast(nn.Module):

    def __init__(self,
                 dims,
                 radius_factor=0.2,
                 sigma_factor=0.2,
                 kernel=1,
                 unit=2,
                 max_num_moments=100,
                 pred_indices=None,
                 loss_indices=None,
                 center_loss=dict(type='GaussianFocalLoss', loss_weight=1.0),
                 window_loss=dict(type='L1Loss', loss_weight=0.1),
                 offset_loss=dict(type='L1Loss', loss_weight=1.0),
                 contrast_loss=dict(type='TripletLoss'),
                 **kwargs):
        super(BoundaryHead_contrast, self).__init__()

        self.center_pred = build_linear_modules(dims, **kwargs)
        self.window_pred = build_linear_modules(dims, **kwargs)
        self.offset_pred = build_linear_modules(dims, **kwargs)

        self.center_loss = build_loss(center_loss)
        self.window_loss = build_loss(window_loss)
        self.offset_loss = build_loss(offset_loss)
        self.contrast_loss = build_loss(contrast_loss)

        self.radius_factor = radius_factor
        self.sigma_factor = sigma_factor
        self.kernel = kernel
        self.unit = unit
        self.max_num_moments = max_num_moments
        self.pred_indices = pred_indices or (-1, )
        self.loss_indices = loss_indices or self.pred_indices

    def get_targets(self, boundary, num_clips):
        batch_size = boundary.size(0)
        avg_factor = 0

        center_tgt = boundary.new_zeros(batch_size, num_clips)
        window_tgt = boundary.new_zeros(batch_size, num_clips)
        offset_tgt = boundary.new_zeros(batch_size, num_clips)
        weight = boundary.new_zeros(batch_size, num_clips)

        for batch_id in range(batch_size):
            batch_boundary = boundary[batch_id]
            batch_boundary[:, 1] -= self.unit

            keep = batch_boundary[:, 0] != -1
            batch_boundary = batch_boundary[keep] / self.unit

            num_centers = batch_boundary.size(0)
            avg_factor += num_centers                 # 正样本的数量

            centers = batch_boundary.mean(dim=-1).clamp(max=num_clips - 0.5)   #  -0.5:-1
            windows = batch_boundary[:, 1] - batch_boundary[:, 0]

            for i, center in enumerate(centers):
                radius = (windows[i] * self.radius_factor).int().item()      # 0.2
                sigma = (radius + 1) * self.sigma_factor                     # 0.2
                center_int = center.int().item()

                heatmap = batch_boundary.new_zeros(num_clips)
                start = max(0, center_int - radius)
                end = min(center_int + radius + 1, num_clips)

                kernel = torch.arange(start - center_int, end - center_int)
                kernel = (-kernel**2 / (2 * sigma**2)).exp()

                heatmap[start:end] = kernel
                center_tgt[batch_id] = torch.max(center_tgt[batch_id], heatmap)
                window_tgt[batch_id, center_int] = windows[i]
                offset_tgt[batch_id, center_int] = center - center_int
                weight[batch_id, center_int] = 1

        return center_tgt, window_tgt, offset_tgt, weight, avg_factor

    def get_boundary(self, center_pred, window_pred, offset_pred):
        pad = (self.kernel - 1) // 2
        hmax = F.max_pool1d(center_pred, self.kernel, stride=1, padding=pad)
        keep = (hmax == center_pred).float()
        center_pred = center_pred * keep

        topk = min(self.max_num_moments, center_pred.size(1))
        scores, inds = torch.topk(center_pred, topk)

        center = (inds + offset_pred.gather(1, inds)).clamp(min=0,max=center_pred.size(1)-1)
        # window = window_pred.gather(1, inds).clamp(min=0,max=center_pred.size(1))

        window = window_pred.gather(1, inds).clamp(min=0)

        boundry = center.unsqueeze(-1).repeat(1, 1, 2)
        boundry[:, :, 0] = center - window / 2
        boundry[:, :, 1] = center + window / 2
        boundry = boundry.clamp(min=0, max=center_pred.size(1) - 1) * self.unit
        boundry[:, :, 1] += self.unit

        boundary = torch.cat((boundry, scores.unsqueeze(-1)), dim=2)
        return boundary

    def get_contrast(self, data, video_feature, query_feature):

        pos_clips_list = data['pos_clip'].data.tolist()
        neg_clips_list = data['neg_clip'].data.tolist()
        batch_size, seq_len, feature_dim = video_feature.shape
        pos_sim_all = []
        neg_sim_all = []


        # 遍历每个样本
        for batch_idx, (pos_clip_indices,neg_clip_indices) in enumerate(zip(pos_clips_list,neg_clips_list)):
            # 提取当前样本的特征
            sample_features = video_feature[batch_idx]
            # 遍历当前样本的所有有效索引
            pos_sample_selected_features = []
            neg_sample_selected_features = []
            for idx in pos_clip_indices:
                if idx != -1 and idx < seq_len:  # 忽略无效的索引
                    pos_sample_selected_features.append(sample_features[idx])

            for idx in neg_clip_indices:
                if idx != -1 and idx < seq_len:  # 忽略无效的索引
                    neg_sample_selected_features.append(sample_features[idx])

            if(len(neg_sample_selected_features)==0 or len(pos_sample_selected_features)==0):
                continue

            # 如果有必要，您可以在这里将sample_selected_features转换为张量
            pos_sample_selected_features = torch.stack(pos_sample_selected_features)
            neg_sample_selected_features = torch.stack(neg_sample_selected_features)

            query = query_feature[batch_idx]

            # 方案一：取最后的token（clip）

            # 方案二：计算每个token,求平均
            pos_sim_idx = []
            neg_sim_idx = []
            for i in range(pos_sample_selected_features.size(0)):
                vector_a = query
                vector_b = pos_sample_selected_features[i]

                if vector_b.dim()==1:
                    vector_b = vector_b.unsqueeze(0)
                pos_sim = F.cosine_similarity(vector_a, vector_b)
                pos_sim_mean = torch.mean(pos_sim)
                pos_sim_idx.append(pos_sim_mean)


            for i in range(neg_sample_selected_features.size(0)):
                vector_a = query
                vector_b = neg_sample_selected_features[i]
                if vector_b.dim()==1:
                    vector_b = vector_b.unsqueeze(0)
                neg_sim = F.cosine_similarity(vector_a, vector_b)
                neg_sim_mean = torch.mean(neg_sim)
                neg_sim_idx.append(neg_sim_mean)
                # print("neg_sim",neg_sim)

            pos_sim_idx = torch.stack(pos_sim_idx)
            neg_sim_idx = torch.stack(neg_sim_idx)

            # print("pos_smi_idx", pos_sim_idx)
            # print("neg_smi_idx", neg_sim_idx)

            pos_sim_idx_mean = torch.mean(pos_sim_idx)
            # print("pos_smi_mean", pos_sim_idx_mean)
            neg_sim_idx_mean = torch.mean(neg_sim_idx)
            # print("neg_smi_mean", neg_sim_idx_mean)

            pos_sim_all.append(pos_sim_idx_mean)
            neg_sim_all.append(neg_sim_idx_mean)


        # 打印出新张量的形状以确认结果
        if(len(pos_sim_all) ==0 ):
            return None,None,0

        pos_sim_all = torch.stack(pos_sim_all)
        neg_sim_all = torch.stack(neg_sim_all)
        # print("pos_sim_all:", pos_sim_all)
        # print("neg_smi_all",neg_sim_all)

        avg = len(pos_sim_all)


        return pos_sim_all, neg_sim_all,avg

    def forward(self, inputs, data, output, mode,video_feature =None,query_feature=None):
        mask = torch.where(data['saliency'] >= 0, 1, 0)

        pred_indices = [idx % len(inputs) for idx in self.pred_indices]    # len(input)==3
        loss_indices = [idx % len(inputs) for idx in self.loss_indices]

        out = []
        for i, x in enumerate(inputs):
            center_pred = self.center_pred(x).squeeze(-1).sigmoid() * mask
            window_pred = self.window_pred(x).squeeze(-1)
            offset_pred = self.offset_pred(x).squeeze(-1)

            if i in pred_indices:
                boundary = self.get_boundary(center_pred, window_pred,
                                             offset_pred)
                out.append(boundary)

            if mode != 'test' and i in loss_indices:
                tgts = self.get_targets(data['boundary'], mask.size(1))
                center_tgt, window_tgt, offset_tgt, weight, avg_factor = tgts

                pos_sim, neg_sim ,avg = self.get_contrast(data,video_feature,query_feature)

                output[f'd{i}.center_loss'] = self.center_loss(
                    center_pred,
                    center_tgt,
                    weight=mask,
                    avg_factor=avg_factor)
                output[f'd{i}.window_loss'] = self.window_loss(
                    window_pred,
                    window_tgt,
                    weight=weight,
                    avg_factor=avg_factor)
                output[f'd{i}.offset_loss'] = self.offset_loss(
                    offset_pred,
                    offset_tgt,
                    weight=weight,
                    avg_factor=avg_factor)

                if avg > 0: 
                    output[f'd{i}.contrast_loss'] = self.contrast_loss(
                        pos_sim = pos_sim,
                        neg_sim = neg_sim,
                        avg_factor=avg)

        output['_out']['boundary'] = torch.cat(out, dim=1).detach().cpu()
        return output

@MODELS.register()
class BoundaryHead(nn.Module):

    def __init__(self,
                 dims,
                 radius_factor=0.2,
                 sigma_factor=0.2,
                 kernel=1,
                 unit=2,
                 max_num_moments=100,
                 pred_indices=None,
                 loss_indices=None,
                 center_loss=dict(type='GaussianFocalLoss', loss_weight=1.0),
                 window_loss=dict(type='L1Loss', loss_weight=0.1),
                 offset_loss=dict(type='L1Loss', loss_weight=1.0),
                 **kwargs):
        super(BoundaryHead, self).__init__()

        self.center_pred = build_linear_modules(dims, **kwargs)
        self.window_pred = build_linear_modules(dims, **kwargs)
        self.offset_pred = build_linear_modules(dims, **kwargs)

        self.center_loss = build_loss(center_loss)
        self.window_loss = build_loss(window_loss)
        self.offset_loss = build_loss(offset_loss)

        self.radius_factor = radius_factor
        self.sigma_factor = sigma_factor
        self.kernel = kernel
        self.unit = unit
        self.max_num_moments = max_num_moments
        self.pred_indices = pred_indices or (-1, )
        self.loss_indices = loss_indices or self.pred_indices

    def get_targets(self, boundary, num_clips):
        batch_size = boundary.size(0)
        avg_factor = 0

        center_tgt = boundary.new_zeros(batch_size, num_clips)
        window_tgt = boundary.new_zeros(batch_size, num_clips)
        offset_tgt = boundary.new_zeros(batch_size, num_clips)
        weight = boundary.new_zeros(batch_size, num_clips)


        for batch_id in range(batch_size):
            batch_boundary = boundary[batch_id]

            batch_boundary[:, 1] -= self.unit

            keep = batch_boundary[:, 0] != -1
            batch_boundary = batch_boundary[keep] / self.unit

            num_centers = batch_boundary.size(0)
            avg_factor += num_centers

            centers = batch_boundary.mean(dim=-1).clamp(max=num_clips - 0.5)   #  -0.5:-1
            windows = batch_boundary[:, 1] - batch_boundary[:, 0]


            for i, center in enumerate(centers):
                radius = (windows[i] * self.radius_factor).int().item()
                sigma = (radius + 1) * self.sigma_factor
                center_int = center.int().item()

                heatmap = batch_boundary.new_zeros(num_clips)
                start = max(0, center_int - radius)
                end = min(center_int + radius + 1, num_clips)

                kernel = torch.arange(start - center_int, end - center_int)
                kernel = (-kernel**2 / (2 * sigma**2)).exp()

                heatmap[start:end] = kernel

                center_tgt[batch_id] = torch.max(center_tgt[batch_id], heatmap)
                window_tgt[batch_id, center_int] = windows[i]
                offset_tgt[batch_id, center_int] = center - center_int
                weight[batch_id, center_int] = 1

        return center_tgt, window_tgt, offset_tgt, weight, avg_factor

    def get_boundary(self, center_pred, window_pred, offset_pred):
        pad = (self.kernel - 1) // 2
        hmax = F.max_pool1d(center_pred, self.kernel, stride=1, padding=pad)
        keep = (hmax == center_pred).float()
        center_pred = center_pred * keep

        # print("center_pre:",center_pred)

        topk = min(self.max_num_moments, center_pred.size(1))
        scores, inds = torch.topk(center_pred, topk)

        center = inds + offset_pred.gather(1, inds).clamp(min=0)
        window = window_pred.gather(1, inds).clamp(min=0)

        boundry = center.unsqueeze(-1).repeat(1, 1, 2)
        boundry[:, :, 0] = center - window / 2
        boundry[:, :, 1] = center + window / 2
        boundry = boundry.clamp(min=0, max=center_pred.size(1) - 1) * self.unit
        boundry[:, :, 1] += self.unit

        boundary = torch.cat((boundry, scores.unsqueeze(-1)), dim=2)
        return boundary

    def forward(self, inputs, data, output, mode):
        mask = torch.where(data['saliency'] >= 0, 1, 0)

        pred_indices = [idx % len(inputs) for idx in self.pred_indices]
        loss_indices = [idx % len(inputs) for idx in self.loss_indices]

        out = []
        for i, x in enumerate(inputs):
            center_pred = self.center_pred(x).squeeze(-1).sigmoid() * mask
            window_pred = self.window_pred(x).squeeze(-1)
            offset_pred = self.offset_pred(x).squeeze(-1)

            if i in pred_indices:
                boundary = self.get_boundary(center_pred, window_pred,
                                             offset_pred)
                out.append(boundary)

            if mode != 'test' and i in loss_indices:
                tgts = self.get_targets(data['boundary'], mask.size(1))
                center_tgt, window_tgt, offset_tgt, weight, avg_factor = tgts

                output[f'd{i}.center_loss'] = self.center_loss(
                    center_pred,
                    center_tgt,
                    weight=mask,
                    avg_factor=avg_factor)
                output[f'd{i}.window_loss'] = self.window_loss(
                    window_pred,
                    window_tgt,
                    weight=weight,
                    avg_factor=avg_factor)
                output[f'd{i}.offset_loss'] = self.offset_loss(
                    offset_pred,
                    offset_tgt,
                    weight=weight,
                    avg_factor=avg_factor)

        output['_out']['boundary'] = torch.cat(out, dim=1).detach().cpu()
        return output
