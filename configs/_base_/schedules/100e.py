# runtime settings
stages = dict(
    epochs=100,
    optimizer=dict(type='Adam', lr=0.001, weight_decay=1e-4),
    validation=dict(interval=1))

