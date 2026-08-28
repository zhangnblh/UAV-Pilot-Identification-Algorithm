_base_ = 'mmdet::rtmdet/rtmdet_tiny_8xb32-300e_coco.py'

# Classes must match the category names in all COCO annotation files.
class_names = ('controller',)
metainfo = dict(
    classes=class_names,
    palette=[(255, 80, 80), (80, 255, 80), (80, 160, 255)])

# Resolve paths relative to this config so evaluation works from any cwd.
data_root = '{{fileDirname}}/../dataset/coco2.0/'
work_dir = '{{fileDirname}}/test_results'
backend_args = None
input_size = (960, 960)

# Disable the URL-based backbone initialization inherited from the official
# config. A complete local COCO-pretrained RTMDet-Tiny checkpoint is loaded
# through load_from below.
model = dict(
    backbone=dict(init_cfg=None, norm_cfg=dict(type='BN')),
    neck=dict(norm_cfg=dict(type='BN')),
    bbox_head=dict(num_classes=1, norm_cfg=dict(type='BN')))

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='CachedMosaic',
        img_scale=input_size,
        pad_val=114.0,
        max_cached_images=20,
        random_pop=False),
    dict(
        type='RandomResize',
        scale=(input_size[0] * 2, input_size[1] * 2),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    # Empty images are valid background samples, so negative crops are allowed.
    dict(
        type='RandomCrop',
        crop_size=input_size,
        allow_negative_crop=True),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='Pad',
        size=input_size,
        pad_val=dict(img=(114, 114, 114))),
    dict(
        type='CachedMixUp',
        img_scale=input_size,
        ratio_range=(1.0, 1.0),
        max_cached_images=10,
        random_pop=False,
        pad_val=(114, 114, 114),
        prob=0.3),
    dict(type='PackDetInputs')
]

train_pipeline_stage2 = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomResize',
        scale=input_size,
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_size=input_size,
        allow_negative_crop=True),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='Pad',
        size=input_size,
        pad_val=dict(img=(114, 114, 114))),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=input_size, keep_ratio=True),
    dict(
        type='Pad',
        size=input_size,
        pad_val=dict(img=(114, 114, 114))),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    _delete_=True,
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=None,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='annotations/instances_Train.json',
        data_prefix=dict(img='images/Train/'),
        metainfo=metainfo,
        # Keep genuine negative images that contain none of the three classes.
        filter_cfg=dict(filter_empty_gt=False, min_size=16),
        pipeline=train_pipeline,
        backend_args=backend_args))

val_dataloader = dict(
    _delete_=True,
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='annotations/instances_Validation.json',
        data_prefix=dict(img='images/Validation/'),
        metainfo=metainfo,
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))

test_dataloader = dict(
    _delete_=True,
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='annotations/instances_Test.json',
        data_prefix=dict(img='images/Test/'),
        metainfo=metainfo,
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))

val_evaluator = dict(
    _delete_=True,
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_Validation.json',
    metric='bbox',
    classwise=True,
    format_only=False,
    backend_args=backend_args)

test_evaluator = dict(
    _delete_=True,
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_Test.json',
    metric='bbox',
    classwise=True,
    format_only=False,
    backend_args=backend_args)

max_epochs = 150
stage2_num_epochs = 10
base_lr = 0.00025  # Scaled for one GPU and batch size 16.

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=5,
    dynamic_intervals=[(max_epochs - stage2_num_epochs, 1)])

optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0,
        bias_decay_mult=0,
        bypass_duplicate=True))

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=1000),
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True)
]

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=20),
    checkpoint=dict(
        type='CheckpointHook',
        interval=5,
        save_best='coco/bbox_mAP',
        rule='greater',
        max_keep_ckpts=3))

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49),
    dict(
        type='PipelineSwitchHook',
        switch_epoch=max_epochs - stage2_num_epochs,
        switch_pipeline=train_pipeline_stage2)
]

# Inference passes the repository checkpoint explicitly. Keep this unset so
# the standalone configuration does not depend on a training-machine path.
load_from = None
resume = False
randomness = dict(seed=42, deterministic=False)
auto_scale_lr = dict(enable=False, base_batch_size=256)
