_base_ = [
    '../_base_/models/umt_base_contrast.py', '../_base_/plugins/mr_0.py',
    '../_base_/datasets/charades_clip_contrast.py', '../_base_/schedules/100e.py',
    '../_base_/runtime.py','../_base_/seed.py'
]
# model settings
# model = dict(audio_enc=dict(dims=[4096, 256]))
# dataset settings

model = dict(video_enc=dict(dims=[4096, 256]))
model = dict(optic_enc=dict(dims=[4096, 256]))
model = dict(audio_enc=dict(dims=[2048, 256]))

data = dict(train=dict(modality='va'), val=dict(modality='va'))
