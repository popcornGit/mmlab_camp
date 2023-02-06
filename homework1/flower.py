model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='EfficientNet',
        arch='b3',
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            'checkpoints/efficientnet/efficientnet-b3_3rdparty_8xb32_in1k_20220119-4b4d7487.pth',
            prefix='backbone')),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=5,
        in_channels=1536,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5)))
dataset_type = 'Flower'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        size=300,
        efficientnet_style=True,
        interpolation='bicubic'),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='CenterCrop',
        crop_size=300,
        efficientnet_style=True,
        interpolation='bicubic'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=2,
    train=dict(
        type='Flower',
        data_prefix='data/flower_dataset/train',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='RandomResizedCrop',
                size=300,
                efficientnet_style=True,
                interpolation='bicubic'),
            dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='ToTensor', keys=['gt_label']),
            dict(type='Collect', keys=['img', 'gt_label'])
        ],
        ann_file='data/flower_dataset/train.txt',
        classes=['daisy', 'rose', 'dandelion', 'sunflower', 'tulip']),
    val=dict(
        type='Flower',
        data_prefix='data/flower_dataset/val',
        ann_file='data/flower_dataset/val.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='CenterCrop',
                crop_size=300,
                efficientnet_style=True,
                interpolation='bicubic'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ],
        classes=['daisy', 'rose', 'dandelion', 'sunflower', 'tulip']),
    test=dict(
        type='Flower',
        data_prefix='data/flower_dataset/test',
        ann_file='data/flower_dataset/test.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='CenterCrop',
                crop_size=300,
                efficientnet_style=True,
                interpolation='bicubic'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ],
        classes=['daisy', 'rose', 'dandelion', 'sunflower', 'tulip']))
evaluation = dict(interval=1, metric='accuracy', metric_options=dict(topk=1))
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='step', step=[30, 60, 90])
runner = dict(type='EpochBasedRunner', max_epochs=100)
checkpoint_config = dict(interval=1)
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
classes = ['daisy', 'rose', 'dandelion', 'sunflower', 'tulip']
work_dir = './work_dirs/flower'
gpu_ids = [1]
