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
        num_classes=10,
        in_channels=1536,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5)))
dataset_type = 'Person'
img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[51.5865, 50.847, 51.255], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomCrop', size=32, padding=4),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Resize', size=32),
    dict(
        type='Normalize',
        mean=[127.5, 127.5, 127.5],
        std=[51.5865, 50.847, 51.255],
        to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=32),
    dict(
        type='Normalize',
        mean=[127.5, 127.5, 127.5],
        std=[51.5865, 50.847, 51.255],
        to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=64,
    workers_per_gpu=2,
    train=dict(
        type='Person',
        data_prefix='data/cifar10/train',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='RandomCrop', size=32, padding=4),
            dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
            dict(type='Resize', size=32),
            dict(
                type='Normalize',
                mean=[127.5, 127.5, 127.5],
                std=[51.5865, 50.847, 51.255],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='ToTensor', keys=['gt_label']),
            dict(type='Collect', keys=['img', 'gt_label'])
        ],
        ann_file='data/cifar10/meta/train.txt',
        classes=[
            'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog',
            'horse', 'ship', 'truck'
        ]),
    val=dict(
        type='Person',
        data_prefix='data/cifar10/val',
        ann_file='data/cifar10/meta/val.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', size=32),
            dict(
                type='Normalize',
                mean=[127.5, 127.5, 127.5],
                std=[51.5865, 50.847, 51.255],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ],
        classes=[
            'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog',
            'horse', 'ship', 'truck'
        ]),
    test=dict(
        type='Person',
        data_prefix='data/cifar10/test',
        ann_file='data/cifar10/meta/test.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', size=32),
            dict(
                type='Normalize',
                mean=[127.5, 127.5, 127.5],
                std=[51.5865, 50.847, 51.255],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ],
        classes=[
            'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog',
            'horse', 'ship', 'truck'
        ]))
evaluation = dict(interval=1, metric='accuracy', metric_options=dict(topk=1))
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='step', step=[15, 30, 45])
runner = dict(type='EpochBasedRunner', max_epochs=100)
checkpoint_config = dict(interval=1)
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
classes = [
    'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
    'ship', 'truck'
]
work_dir = './work_dirs/cifar10_raw'
gpu_ids = [1]
