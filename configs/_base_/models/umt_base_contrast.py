_base_ = 'umt_small_contrast.py'
# model settings
model = dict(query_dec=dict(dec_cfg=dict(_repeat_=3)))
