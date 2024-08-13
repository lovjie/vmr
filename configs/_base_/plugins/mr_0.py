# model settings
model = dict(
    video_enc=dict(dims=[512, 256]),
    optic_enc=dict(dims=[2048, 256]),
    # video_enc=dict(dims=[4096, 256]),
    # optic_enc=dict(dims=[4096, 256]),
    audio_enc=dict(dims=[2048, 256]),
    query_gen=dict(dims=[512, 256]),
    pred_head=dict(
        # type='BoundaryHead_contrast',
        type='BoundaryHead_contrast',
        dims=[256, 1],
        unit=1/6,
        window_loss=dict(type='L1Loss', loss_weight=0.05),
        offset_loss=dict(type='L1Loss', loss_weight=0.5)))
