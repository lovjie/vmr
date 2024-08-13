# runtime settings
stages = dict(
    epochs=10,
    optimizer=dict(type='Adam', lr=1e-3, weight_decay=1e-4),
    validation=dict(interval=1),
    checkpointing=dict(interval=5))