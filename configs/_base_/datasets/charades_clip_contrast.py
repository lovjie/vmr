_base_ = 'datasets'
# dataset settings
dataset_type = 'CharadesSTA_clip_contrast'
data_root = ' '   
data = dict(
    train=dict(
        type=dataset_type,
        modality=None,
        unit = 1/6 ,
        label_path=data_root + 'charades_sta_train.txt',
        # video_path=data_root + 'rgb_features',
        # optic_path=data_root + 'opt_features',
        video_path=data_root + 'charades_clip_feat/clip_feats_24fps_stride4',
        optic_path=data_root + 'charades_slowfast/rgb_flow',
        audio_path=data_root + 'audio_features',
        loader=dict(batch_size=8, num_workers=0, shuffle=True)),
    val=dict(
        type=dataset_type,
        modality=None,
        unit = 1/6 ,
        label_path=data_root + 'charades_sta_test.txt',
        # video_path=data_root + 'rgb_features',
        # optic_path=data_root + 'opt_features',
        video_path=data_root + 'charades_clip_feat/clip_feats_24fps_stride4',
        optic_path=data_root + 'charades_slowfast/rgb_flow',
        audio_path=data_root + 'audio_features',
        loader=dict(batch_size=1, num_workers=0, shuffle=False)))
