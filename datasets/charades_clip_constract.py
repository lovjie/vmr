# Copyright (c) THL A29 Limited, a Tencent company. All rights reserved.

import nncore
import torch
import torch.nn as nn
import torch.nn.functional as F
from nncore.dataset import DATASETS
from nncore.ops import temporal_iou
from nncore.parallel import DataContainer
from torch.utils.data import Dataset
from torchtext import vocab
import random
import numpy as np

import torch
import clip


@DATASETS.register()
class CharadesSTA_clip_contrast(Dataset):

    def __init__(self,
                 modality,
                 unit,
                 label_path,
                 video_path,
                 optic_path=None,
                 audio_path=None):
        assert modality in ('va', 'vo')
        self.label = nncore.load(label_path)

        self.modality = modality
        self.unit = unit
        self.label_path = label_path
        self.video_path = video_path
        self.optic_path = optic_path
        self.audio_path = audio_path

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, preprocess = clip.load("clip/ViT-B-32.pt", device=self.device)


        # cache_dir = 'Golve'
        # self.vocab = vocab.GloVe(name="6B", dim=300, cache=cache_dir)

        # self.vocab.itos.extend(['<unk>'])
        # self.vocab.stoi['<unk>'] = self.vocab.vectors.shape[0]
        # self.vocab.vectors = torch.cat(
        #     (self.vocab.vectors, torch.zeros(1, self.vocab.dim)), dim=0)
        # self.embedding = nn.Embedding.from_pretrained(self.vocab.vectors)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        video = self.get_video(idx)
        optic = self.get_optic(idx)
        # audio = self.get_audio(idx)
        query = self.get_query(idx)

        # num_clips = min(c.size(0) for c in (video, optic))
        num_clips = min(c.size(0) for c in (video, optic))

        boundary = self.get_boundary(idx)
        saliency = torch.ones(num_clips)

        pos_clip, neg_clip ,saliency_1 = self.get_saliency_labels_sub_as_query(boundary,num_clips,strategy=2)

        # gt_st = boundary[0][0].item()
        # gt_ed = boundary[0][1].item()
        # print(round(gt_st,2),round(gt_ed,2))

        data = dict(
            video=DataContainer(video[:num_clips]),
            optic=DataContainer(optic[:num_clips]),
            # audio=DataContainer(audio[:num_clips]),
            query=DataContainer(query, pad_value=float('inf')),
            saliency=DataContainer(saliency, pad_value=-1),
            pos_clip=DataContainer(pos_clip,pad_value=-1),
            neg_clip=DataContainer(neg_clip,pad_value=-1),
            meta=DataContainer(self.label[idx], cpu_only=True))

        if boundary is not None:
            data['boundary'] = DataContainer(boundary, pad_value=-1)

        return data

    def parse_boundary(self, label):
        boundary = label.split('##')[0].split()[1:]
        if float(boundary[1]) < float(boundary[0]):
            boundary = [boundary[1], boundary[0]]
        return torch.Tensor([[float(s) for s in boundary]])

    def get_video(self, idx):
        vid = self.label[idx].split()[0]
        video = nncore.load(nncore.join(self.video_path, f'{vid}.npz'))
        return F.normalize(torch.from_numpy(video['features']).float())
    
    def get_optic(self, idx):
        vid = self.label[idx].split()[0]
        optic = nncore.load(nncore.join(self.optic_path, f'{vid}_rgb_flow.npy'))
        return F.normalize(torch.from_numpy(optic).float())

    # def get_audio(self, idx):
    #     vid = self.label[idx].split()[0]
    #     path = self.audio_path if self.modality == 'va' else self.optic_path
    #     audio = nncore.load(nncore.join(path, f'{vid}.npy'))
    #     return F.normalize(torch.from_numpy(audio).float())

    # def get_query(self, idx):
    #     query = self.label[idx].split('##')[-1][:-1]
    #     word_inds = torch.LongTensor(
    #         [self.vocab.stoi.get(w.lower(), 400000) for w in query.split()])
    #     return self.embedding(word_inds)

    def get_query(self, idx):
        # clip 编码
        query = self.label[idx].split('##')[-1][:-1]
        text = clip.tokenize(query).to(self.device)
        text_features = torch.Tensor(self.model.encode_text(text).detach().cpu().numpy())

        return text_features

    def get_boundary(self, idx):
        return self.parse_boundary(self.label[idx])

    def get_saliency_labels_sub_as_query(self, boundary, num_clips ,max_n =2,strategy=2):
        '''
        :param boundary: [start_time，end_time]
        :param num_clips: video_fps: (batch of feature)
        :param max_n:
        :return: pos_clip , neg_clip , fps_score(0,1)


        >>> boundart = [20.8 , 30.0]  unit =1
        >>> st = 20
        >>> ed = 30
        >>> num_clips = 40

        方式1：
        max_n = 2
        pos_clip = 21,24
        neg_clip = 1,2

        方式2：
        max_n = (ed - st)*n
        pos_clip = [22,26]
        neg_clip = [5, 18]

        '''

        # print("num_clip",num_clips)
        # print("boundary:",boundary)

        gt_st = int(round(boundary[0][0].item(),2) / self.unit)
        gt_ed = max( 0, min( int(round(boundary[0][1].item(),2) / self.unit), num_clips - 1))

        # print("strat_end_time:",gt_st,gt_ed)

        if strategy == 1:
            if gt_st != gt_ed:
                pos_clip_indices = random.sample(range(gt_st, gt_ed + 1), k=max_n)
            else:
                pos_clip_indices = [gt_st, gt_st]

            neg_pool = list(range(0, gt_st)) + list(range(gt_ed + 1, num_clips))
            neg_clip_indices = random.sample(neg_pool, k=max_n)



        if strategy == 2:
            max_n = 2
            if (gt_ed - gt_st) <= max_n:
                pos_clip_indices = [gt_st, gt_ed]
            else:
                pos_clip_indices = random.sample(range(gt_st, gt_ed + 1), k=max_n)
            

            pos_clip_indices = sorted(pos_clip_indices)

            range1 = list(range(0, gt_st))
            range2 = list(range(gt_ed + 1, num_clips))

            # 检查采样数量是否超出范围内的元素数量
            if len(range1) < max_n or len(range2) < max_n:
                if len(range1) < max_n and len(range2) < max_n:
                    print("无法采样,负样本clip不足")
                elif len(range1) >= max_n:
                    neg_clip_indices = random.sample(range1, k=max_n)
                else:
                    neg_clip_indices = random.sample(range2, k=max_n)

            else:
                neg_pool = [random.sample(range1, k=max_n), random.sample(range2, k=max_n)]
                neg_clip_indices = random.choice(neg_pool)

            neg_clip_indices = sorted(neg_clip_indices)


        pos_clip_indices = list(range(pos_clip_indices[0],pos_clip_indices[1]+1))
        neg_clip_indices = list(range(neg_clip_indices[0],neg_clip_indices[1]+1))

        # print("pos:", pos_clip_indices)
        # print("neg", neg_clip_indices)

        pos_clip_indices = torch.tensor(pos_clip_indices, dtype=torch.int64)
        neg_clip_indices = torch.tensor(neg_clip_indices, dtype=torch.int64)


        score_array = np.zeros(num_clips)
        score_array[gt_st:gt_ed + 1] = 1

        return pos_clip_indices, neg_clip_indices, score_array

    def evaluate(self,
                 blob,
                 method='gaussian',
                 nms_thr=0.3,
                 sigma=0.5,
                 rank=[1, 5],
                 iou_thr=[0.5, 0.7],
                 **kwargs):
        assert method in ('fast', 'normal', 'linear', 'gaussian')

        blob = nncore.to_dict_of_list(blob)
        results = dict()

        print('Performing temporal NMS...')
        boundary = []


        for bnd in blob['boundary']:
            bnd = bnd[0]

            if method == 'fast':
                iou = temporal_iou(bnd[:, :-1], bnd[:, :-1]).triu(diagonal=1)
                keep = iou.amax(dim=0) <= nms_thr
                bnd = bnd[keep]
            else:
                for i in range(bnd.size(0)):
                    max_idx = bnd[i:, -1].argmax(dim=0)
                    bnd = nncore.swap_element(bnd, i, max_idx + i)
                    iou = temporal_iou(bnd[i, None, :-1], bnd[i + 1:, :-1])[0]

                    if method == 'normal':
                        bnd[i + 1:, -1][iou >= nms_thr] = 0
                    elif method == 'linear':
                        bnd[i + 1:, -1] *= 1 - iou
                    else:
                        bnd[i + 1:, -1] *= (-iou.pow(2) / sigma).exp()

            boundary.append(bnd)


        for k in rank:
            for thr in iou_thr:
                print(f'Evaluating Rank{k}@{thr}...')
                hits = 0

                for idx, bnd in enumerate(boundary):
                    inds = torch.argsort(bnd[:, -1], descending=True)
                    keep = inds[:k]
                    bnd = bnd[:, :-1][keep]

                    gt = self.parse_boundary(blob['meta'][idx][0])
                    iou = temporal_iou(gt, bnd)

                    if iou.max() >= thr:
                        hits += 1

                results[f'Rank{k}@{thr}'] = hits / len(self.label)

        return results
