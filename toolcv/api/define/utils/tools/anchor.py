from mmdet.core.anchor.anchor_generator import SSDAnchorGenerator, AnchorGenerator, YOLOAnchorGenerator, \
    LegacySSDAnchorGenerator, LegacyAnchorGenerator


def ssd300_anchors(anchor_generator={}, featmap_sizes=[(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)],
                   device='cpu', clip=False, normalization=False, clip_pixel=1):
    """
    clip = True 裁剪到图像范围内
    normalization=True 缩放到 0~1
    """
    deafual_anchor_generator = dict(
        # type='SSDAnchorGenerator',
        input_size=300,
        scale_major=False,
        basesize_ratio_range=(0.15, 0.9),
        strides=[8, 16, 32, 64, 100, 300],
        ratios=[[2], [2, 3], [2, 3], [2, 3], [2], [2]],
        # min_sizes=[21, 45, 99, 153, 207, 261],
        # max_sizes=[45, 99, 153, 207, 261, 315],
    )
    deafual_anchor_generator.update(anchor_generator)

    ssdanchors = SSDAnchorGenerator(**deafual_anchor_generator)
    anchors = ssdanchors.grid_priors(featmap_sizes, device)  # 未缩放
    if clip:  # 裁剪到图像范围内
        anchors = [anchor.clamp(clip_pixel, deafual_anchor_generator.get('input_size') - clip_pixel) for anchor in
                   anchors]
    if normalization:
        anchors = [anchor / deafual_anchor_generator.get('input_size') for anchor in anchors]

    return anchors


def ssd320_anchors(anchor_generator={}, featmap_sizes=[(20, 20), (10, 10), (5, 5), (3, 3), (2, 2), (1, 1)],
                   device='cpu', clip=False, normalization=False, clip_pixel=1):
    """
    clip = True 裁剪到图像范围内
    normalization=True 缩放到 0~1
    """
    deafual_anchor_generator = dict(
        # type='SSDAnchorGenerator',
        input_size=320,
        scale_major=False,
        strides=[16, 32, 64, 107, 160, 320],
        ratios=[[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]],
        min_sizes=[48, 100, 150, 202, 253, 304],
        max_sizes=[100, 150, 202, 253, 304, 320])

    deafual_anchor_generator.update(anchor_generator)

    ssdanchors = SSDAnchorGenerator(**deafual_anchor_generator)
    anchors = ssdanchors.grid_priors(featmap_sizes, device)  # 未缩放
    if clip:  # 裁剪到图像范围内
        anchors = [anchor.clamp(clip_pixel, deafual_anchor_generator.get('input_size') - clip_pixel) for anchor in
                   anchors]
    if normalization:
        anchors = [anchor / deafual_anchor_generator.get('input_size') for anchor in anchors]

    return anchors


def ssd512_anchors(anchor_generator={},
                   featmap_sizes=[(64, 64), (32, 32), (16, 16), (8, 8), (4, 4), (2, 2), (1, 1)],
                   device='cpu', clip=False, normalization=False, clip_pixel=1):
    """
    clip = True 裁剪到图像范围内
    normalization=True 缩放到 0~1
    """
    # in_channels = (512, 1024, 512, 256, 256, 256, 256)
    deafual_anchor_generator = dict(
        # type='SSDAnchorGenerator',
        scale_major=False,
        input_size=512,
        basesize_ratio_range=(0.1, 0.9),
        strides=[8, 16, 32, 64, 128, 256, 512],
        ratios=[[2], [2, 3], [2, 3], [2, 3], [2, 3], [2], [2]])

    deafual_anchor_generator.update(anchor_generator)

    ssdanchors = SSDAnchorGenerator(**deafual_anchor_generator)
    anchors = ssdanchors.grid_priors(featmap_sizes, device)  # 未缩放
    if clip:  # 裁剪到图像范围内
        anchors = [anchor.clamp(clip_pixel, deafual_anchor_generator.get('input_size') - clip_pixel) for anchor in
                   anchors]
    if normalization:
        anchors = [anchor / deafual_anchor_generator.get('input_size') for anchor in anchors]

    return anchors


def rcnn_anchors(anchor_generator, h=512, w=512,
                 featmap_sizes=[(64, 64), (32, 32), (16, 16), (8, 8), (4, 4)],
                 device='cpu', clip=False, normalization=False, clip_pixel=1):
    ag = AnchorGenerator(**anchor_generator)
    anchors = ag.grid_priors(featmap_sizes, device)

    if clip:  # 裁剪到图像范围内
        for i in range(len(anchors)):
            anchors[i][..., [0, 2]] = anchors[i][..., [0, 2]].clamp(clip_pixel, w - clip_pixel)
            anchors[i][..., [1, 3]] = anchors[i][..., [1, 3]].clamp(clip_pixel, h - clip_pixel)

    if normalization:
        for i in range(len(anchors)):
            anchors[i][..., [0, 2]] = anchors[i][..., [0, 2]] / (w - 1)
            anchors[i][..., [1, 3]] = anchors[i][..., [1, 3]] / (h - 1)

    return anchors


def get_anchor_cfg(mode="retinanet"):
    """
    resize 800,1333
    """
    if mode == "retinanet":
        anchor_generator = dict(
            # type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128])
    else:
        # mode == "rcnn":
        anchor_generator = dict(
            # type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64])

    return anchor_generator


def yolo_anchors(cfg={}, size=416, device='cpu', clip=False, normalization=False, clip_pixel=1):
    """size=416 or 608"""
    anchor_generator = dict(
        # type='YOLOAnchorGenerator',
        base_sizes=[[(116, 90), (156, 198), (373, 326)],
                    [(30, 61), (62, 45), (59, 119)],
                    [(10, 13), (16, 30), (33, 23)]],
        strides=[32, 16, 8])
    anchor_generator.update(cfg)
    featmap_sizes = [(size // stride, size // stride) for stride in anchor_generator.get('strides')]

    yg = YOLOAnchorGenerator(**anchor_generator)
    anchors = yg.grid_priors(featmap_sizes, device)

    if clip:  # 裁剪到图像范围内
        anchors = [anchor.clamp(clip_pixel, size - clip_pixel) for anchor in anchors]
    if normalization:
        anchors = [anchor / size for anchor in anchors]

    return anchors


if __name__ == "__main__":
    # target_stds = [0.1, 0.1, 0.2, 0.2]
    # anchors = ssd300_anchors(clip=True, normalization=True)
    # anchors = ssd512_anchors(clip=True, normalization=True)
    # anchors = rcnn_anchors(get_anchor_cfg('rcnn'))
    anchors = yolo_anchors()
    print()
