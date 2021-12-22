from .efficientnet_builder import *
from .feature_hooks import FeatureHooks
from .registry import register_model
from .helpers import load_pretrained
from .adaptive_avgmax_pool import SelectAdaptivePool2d
from .conv2d_layers import select_conv2d
from codebase.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD


__all__ = ['EfficientNet']


def _cfg(url='', **kwargs):
    return {
        'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'conv_stem', 'classifier': 'classifier',
        **kwargs
    }


default_cfgs = {
    'efficientmnext_s': _cfg(
        url='',),
    'efficientmnext_l': _cfg(
        url='',
        input_size=(3, 224, 224), pool_size=(8, 8)),
    'efficientnet_b0': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b0-d6904d92.pth'),
    'efficientnet_b1': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b1-533bc792.pth',
        input_size=(3, 240, 240), pool_size=(8, 8)),
    'efficientnet_b2': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b2_ra-bcdf34b7.pth',
        input_size=(3, 260, 260), pool_size=(9, 9)),
    'efficientnet_b2a': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b2_ra-bcdf34b7.pth',
        input_size=(3, 288, 288), pool_size=(9, 9), crop_pct=1.0),
    'efficientnet_b3': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b3_ra-a5e2fbc7.pth',
        input_size=(3, 300, 300), pool_size=(10, 10), crop_pct=0.904),
    'efficientnet_b3a': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b3_ra-a5e2fbc7.pth',
        input_size=(3, 320, 320), pool_size=(10, 10), crop_pct=1.0),
    'efficientnet_b4': _cfg(
        url='', input_size=(3, 380, 380), pool_size=(12, 12), crop_pct=0.922),
    'efficientnet_b5': _cfg(
        url='', input_size=(3, 456, 456), pool_size=(15, 15), crop_pct=0.934),
    'efficientnet_b6': _cfg(
        url='', input_size=(3, 528, 528), pool_size=(17, 17), crop_pct=0.942),
    'efficientnet_b7': _cfg(
        url='', input_size=(3, 600, 600), pool_size=(19, 19), crop_pct=0.949),
    'efficientnet_b8': _cfg(
        url='', input_size=(3, 672, 672), pool_size=(21, 21), crop_pct=0.954),
    'efficientnet_es': _cfg(
        url=''),
    'efficientnet_em': _cfg(
        url='', input_size=(3, 240, 240), pool_size=(8, 8), crop_pct=0.882),
    'efficientnet_el': _cfg(
        url='', input_size=(3, 300, 300), pool_size=(10, 10), crop_pct=0.904),
    'mixnet_s': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mixnet_s-a907afbc.pth'),
    'mixnet_m': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mixnet_m-4647fc68.pth'),
    'mixnet_l': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mixnet_l-5a9a2ed8.pth'),
    'mixnet_xl': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mixnet_xl_ra-aac3c00c.pth'),
    'mixnet_xxl': _cfg(),
    'tf_mixnet_s': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mixnet_s-89d3354b.pth'),
    'tf_mixnet_m': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mixnet_m-0f4d8805.pth'),
    'tf_mixnet_l': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mixnet_l-6c92e0c8.pth'),
}

_DEBUG = False


class Efficientmnext(nn.Module):
    """ (Generic) EfficientNet

    A flexible and performant PyTorch implementation of efficient network architectures, including:
      * EfficientNet B0-B8
      * EfficientNet-EdgeTPU
      * EfficientNet-CondConv
      * MixNet S, M, L, XL
      * MnasNet A1, B1, and small
      * FBNet C
      * Single-Path NAS Pixel1

    """

    def __init__(self, block_args, num_classes=1000, num_features=1280, in_chans=3, stem_size=32,
                 channel_multiplier=1.0, channel_divisor=8, channel_min=None,
                 pad_type='', act_layer=nn.ReLU, drop_rate=0., drop_connect_rate=0.,
                 se_kwargs=None, norm_layer=nn.BatchNorm2d, norm_kwargs=None, global_pool='avg'):
        super(Efficientmnext, self).__init__()
        norm_kwargs = norm_kwargs or {}

        self.num_classes = num_classes
        self.num_features = num_features
        self.drop_rate = drop_rate
        self._in_chs = in_chans

        # Stem
        stem_size = round_channels(stem_size, channel_multiplier, channel_divisor, channel_min)
        self.conv_stem = select_conv2d(self._in_chs, stem_size, 3, stride=2, padding=pad_type)
        self.bn1 = norm_layer(stem_size, **norm_kwargs)
        self.act1 = act_layer(inplace=True)
        self._in_chs = stem_size

        # Middle stages (IR/ER/DS Blocks)
        builder = EfficientNetBuilder(
            channel_multiplier, channel_divisor, channel_min, 32, pad_type, act_layer, se_kwargs,
            norm_layer, norm_kwargs, drop_connect_rate, verbose=_DEBUG)
        self.blocks = nn.Sequential(*builder(self._in_chs, block_args))
        self.feature_info = builder.features
        self._in_chs = builder.in_chs

        # Head + Pooling
        # self.conv_head = select_conv2d(self._in_chs, self.num_features, 1, padding=pad_type)
        # self.bn2 = norm_layer(self.num_features, **norm_kwargs)
        # self.act2 = act_layer(inplace=True)
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)

        # Classifier
        self.classifier = nn.Linear(self.num_features * self.global_pool.feat_mult(), self.num_classes)

        efficientnet_init_weights(self)

    def as_sequential(self):
        layers = [self.conv_stem, self.bn1, self.act1]
        layers.extend(self.blocks)
        layers.extend([self.conv_head, self.bn2, self.act2, self.global_pool])
        layers.extend([nn.Flatten(), nn.Dropout(self.drop_rate), self.classifier])
        return nn.Sequential(*layers)

    def get_classifier(self):
        return self.classifier

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.classifier = nn.Linear(
            self.num_features * self.global_pool.feat_mult(), num_classes) if num_classes else None

    def forward_features(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.blocks(x)
        # x = self.conv_head(x)
        # x = self.bn2(x)
        # x = self.act2(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.global_pool(x)
        x = x.flatten(1)
        if self.drop_rate > 0.:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        return self.classifier(x)
class EfficientNet(nn.Module):
    """ (Generic) EfficientNet

    A flexible and performant PyTorch implementation of efficient network architectures, including:
      * EfficientNet B0-B8
      * EfficientNet-EdgeTPU
      * EfficientNet-CondConv
      * MixNet S, M, L, XL
      * MnasNet A1, B1, and small
      * FBNet C
      * Single-Path NAS Pixel1

    """

    def __init__(self, block_args, num_classes=1000, num_features=1280, in_chans=3, stem_size=32,
                 channel_multiplier=1.0, channel_divisor=8, channel_min=None,
                 pad_type='', act_layer=nn.ReLU, drop_rate=0., drop_connect_rate=0.,
                 se_kwargs=None, norm_layer=nn.BatchNorm2d, norm_kwargs=None, global_pool='avg'):
        super(EfficientNet, self).__init__()
        norm_kwargs = norm_kwargs or {}

        self.num_classes = num_classes
        self.num_features = num_features
        self.drop_rate = drop_rate
        self._in_chs = in_chans

        # Stem
        stem_size = round_channels(stem_size, channel_multiplier, channel_divisor, channel_min)
        self.conv_stem = select_conv2d(self._in_chs, stem_size, 3, stride=2, padding=pad_type)
        self.bn1 = norm_layer(stem_size, **norm_kwargs)
        self.act1 = act_layer(inplace=True)
        self._in_chs = stem_size

        # Middle stages (IR/ER/DS Blocks)
        builder = EfficientNetBuilder(
            channel_multiplier, channel_divisor, channel_min, 32, pad_type, act_layer, se_kwargs,
            norm_layer, norm_kwargs, drop_connect_rate, verbose=_DEBUG)
        self.blocks = nn.Sequential(*builder(self._in_chs, block_args))
        self.feature_info = builder.features
        self._in_chs = builder.in_chs

        # Head + Pooling
        self.conv_head = select_conv2d(self._in_chs, self.num_features, 1, padding=pad_type)
        self.bn2 = norm_layer(self.num_features, **norm_kwargs)
        self.act2 = act_layer(inplace=True)
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)

        # Classifier
        self.classifier = nn.Linear(self.num_features * self.global_pool.feat_mult(), self.num_classes)

        efficientnet_init_weights(self)

    def as_sequential(self):
        layers = [self.conv_stem, self.bn1, self.act1]
        layers.extend(self.blocks)
        layers.extend([self.conv_head, self.bn2, self.act2, self.global_pool])
        layers.extend([nn.Flatten(), nn.Dropout(self.drop_rate), self.classifier])
        return nn.Sequential(*layers)

    def get_classifier(self):
        return self.classifier

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.classifier = nn.Linear(
            self.num_features * self.global_pool.feat_mult(), num_classes) if num_classes else None

    def forward_features(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.blocks(x)
        x = self.conv_head(x)
        x = self.bn2(x)
        x = self.act2(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.global_pool(x)
        x = x.flatten(1)
        if self.drop_rate > 0.:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        return self.classifier(x)


class EfficientNetFeatures(nn.Module):
    """ EfficientNet Feature Extractor

    A work-in-progress feature extraction module for EfficientNet, to use as a backbone for segmentation
    and object detection models.
    """

    def __init__(self, block_args, out_indices=(0, 1, 2, 3, 4), feature_location='pre_pwl',
                 in_chans=3, stem_size=32, channel_multiplier=1.0, channel_divisor=8, channel_min=None,
                 output_stride=32, pad_type='', act_layer=nn.ReLU, drop_rate=0., drop_connect_rate=0.,
                 se_kwargs=None, norm_layer=nn.BatchNorm2d, norm_kwargs=None):
        super(EfficientNetFeatures, self).__init__()
        norm_kwargs = norm_kwargs or {}

        # TODO only create stages needed, currently all stages are created regardless of out_indices
        num_stages = max(out_indices) + 1

        self.out_indices = out_indices
        self.drop_rate = drop_rate
        self._in_chs = in_chans

        # Stem
        stem_size = round_channels(stem_size, channel_multiplier, channel_divisor, channel_min)
        self.conv_stem = select_conv2d(self._in_chs, stem_size, 3, stride=2, padding=pad_type)
        self.bn1 = norm_layer(stem_size, **norm_kwargs)
        self.act1 = act_layer(inplace=True)
        self._in_chs = stem_size

        # Middle stages (IR/ER/DS Blocks)
        builder = EfficientNetBuilder(
            channel_multiplier, channel_divisor, channel_min, output_stride, pad_type, act_layer, se_kwargs,
            norm_layer, norm_kwargs, drop_connect_rate, feature_location=feature_location, verbose=_DEBUG)
        self.blocks = nn.Sequential(*builder(self._in_chs, block_args))
        self.feature_info = builder.features  # builder provides info about feature channels for each block
        self._in_chs = builder.in_chs

        efficientnet_init_weights(self)
        if _DEBUG:
            for k, v in self.feature_info.items():
                print('Feature idx: {}: Name: {}, Channels: {}'.format(k, v['name'], v['num_chs']))

        # Register feature extraction hooks with FeatureHooks helper
        hook_type = 'forward_pre' if feature_location == 'pre_pwl' else 'forward'
        hooks = [dict(name=self.feature_info[idx]['name'], type=hook_type) for idx in out_indices]
        self.feature_hooks = FeatureHooks(hooks, self.named_modules())

    def feature_channels(self, idx=None):
        """ Feature Channel Shortcut
        Returns feature channel count for each output index if idx == None. If idx is an integer, will
        return feature channel count for that feature block index (independent of out_indices setting).
        """
        if isinstance(idx, int):
            return self.feature_info[idx]['num_chs']
        return [self.feature_info[i]['num_chs'] for i in self.out_indices]

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        self.blocks(x)
        return self.feature_hooks.get_output(x.device)


def _create_model(model_kwargs, default_cfg, pretrained=False, efficientmnext=False):
    if model_kwargs.pop('features_only', False):
        load_strict = False
        model_kwargs.pop('num_classes', 0)
        model_kwargs.pop('num_features', 0)
        model_kwargs.pop('head_conv', None)
        model_class = EfficientNetFeatures
    else:
        load_strict = True
        if efficientmnext:
            model_class = Efficientmnext
        else:
            model_class = EfficientNet

    model = model_class(**model_kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(
            model,
            default_cfg,
            num_classes=model_kwargs.get('num_classes', 0),
            in_chans=model_kwargs.get('in_chans', 3),
            strict=load_strict)
    return model


def _gen_mnasnet_a1(variant, channel_multiplier=1.0, pretrained=False, **kwargs):
    """Creates a mnasnet-a1 model.

    Ref impl: https://github.com/tensorflow/tpu/tree/master/models/official/mnasnet
    Paper: https://arxiv.org/pdf/1807.11626.pdf.

    Args:
      channel_multiplier: multiplier to number of channels per layer.
    """
    arch_def = [
        # stage 0, 112x112 in
        ['ds_r1_k3_s1_e1_c16_noskip'],
        # stage 1, 112x112 in
        ['ir_r2_k3_s2_e6_c24'],
        # stage 2, 56x56 in
        ['ir_r3_k5_s2_e3_c40_se0.25'],
        # stage 3, 28x28 in
        ['ir_r4_k3_s2_e6_c80'],
        # stage 4, 14x14in
        ['ir_r2_k3_s1_e6_c112_se0.25'],
        # stage 5, 14x14in
        ['ir_r3_k5_s2_e6_c160_se0.25'],
        # stage 6, 7x7 in
        ['ir_r1_k3_s1_e6_c320'],
    ]
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def),
        stem_size=32,
        channel_multiplier=channel_multiplier,
        norm_kwargs=resolve_bn_args(kwargs),
        **kwargs
    )
    model = _create_model(model_kwargs, default_cfgs[variant], pretrained)
    return model


def _gen_mnasnet_b1(variant, channel_multiplier=1.0, pretrained=False, **kwargs):
    """Creates a mnasnet-b1 model.

    Ref impl: https://github.com/tensorflow/tpu/tree/master/models/official/mnasnet
    Paper: https://arxiv.org/pdf/1807.11626.pdf.

    Args:
      channel_multiplier: multiplier to number of channels per layer.
    """
    arch_def = [
        # stage 0, 112x112 in
        ['ds_r1_k3_s1_c16_noskip'],
        # stage 1, 112x112 in
        ['ir_r3_k3_s2_e3_c24'],
        # stage 2, 56x56 in
        ['ir_r3_k5_s2_e3_c40'],
        # stage 3, 28x28 in
        ['ir_r3_k5_s2_e6_c80'],
        # stage 4, 14x14in
        ['ir_r2_k3_s1_e6_c96'],
        # stage 5, 14x14in
        ['ir_r4_k5_s2_e6_c192'],
        # stage 6, 7x7 in
        ['ir_r1_k3_s1_e6_c320_noskip']
    ]
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def),
        stem_size=32,
        channel_multiplier=channel_multiplier,
        norm_kwargs=resolve_bn_args(kwargs),
        **kwargs
    )
    model = _create_model(model_kwargs, default_cfgs[variant], pretrained)
    return model


def _gen_mnasnet_small(variant, channel_multiplier=1.0, pretrained=False, **kwargs):
    """Creates a mnasnet-b1 model.

    Ref impl: https://github.com/tensorflow/tpu/tree/master/models/official/mnasnet
    Paper: https://arxiv.org/pdf/1807.11626.pdf.

    Args:
      channel_multiplier: multiplier to number of channels per layer.
    """
    arch_def = [
        ['ds_r1_k3_s1_c8'],
        ['ir_r1_k3_s2_e3_c16'],
        ['ir_r2_k3_s2_e6_c16'],
        ['ir_r4_k5_s2_e6_c32_se0.25'],
        ['ir_r3_k3_s1_e6_c32_se0.25'],
        ['ir_r3_k5_s2_e6_c88_se0.25'],
        ['ir_r1_k3_s1_e6_c144']
    ]
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def),
        stem_size=8,
        channel_multiplier=channel_multiplier,
        norm_kwargs=resolve_bn_args(kwargs),
        **kwargs
    )
    model = _create_model(model_kwargs, default_cfgs[variant], pretrained)
    return model


def _gen_mobilenet_v2(variant, channel_multiplier=1.0, pretrained=False, **kwargs):
    """ Generate MobileNet-V2 network
    Ref impl: https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet_v2.py
    Paper: https://arxiv.org/abs/1801.04381
    """
    arch_def = [
        ['ds_r1_k3_s1_c16'],
        ['ir_r2_k3_s2_e6_c24'],
        ['ir_r3_k3_s2_e6_c32'],
        ['ir_r4_k3_s2_e6_c64'],
        ['ir_r3_k3_s1_e6_c96'],
        ['ir_r3_k3_s2_e6_c160'],
        ['ir_r1_k3_s1_e6_c320'],
    ]
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def),
        stem_size=32,
        channel_multiplier=channel_multiplier,
        norm_kwargs=resolve_bn_args(kwargs),
        act_layer=nn.ReLU6,
        **kwargs
    )
    model = _create_model(model_kwargs, default_cfgs[variant], pretrained)
    return model


def _gen_fbnetc(variant, channel_multiplier=1.0, pretrained=False, **kwargs):
    """ FBNet-C

        Paper: https://arxiv.org/abs/1812.03443
        Ref Impl: https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/modeling/backbone/fbnet_modeldef.py

        NOTE: the impl above does not relate to the 'C' variant here, that was derived from paper,
        it was used to confirm some building block details
    """
    arch_def = [
        ['ir_r1_k3_s1_e1_c16'],
        ['ir_r1_k3_s2_e6_c24', 'ir_r2_k3_s1_e1_c24'],
        ['ir_r1_k5_s2_e6_c32', 'ir_r1_k5_s1_e3_c32', 'ir_r1_k5_s1_e6_c32', 'ir_r1_k3_s1_e6_c32'],
        ['ir_r1_k5_s2_e6_c64', 'ir_r1_k5_s1_e3_c64', 'ir_r2_k5_s1_e6_c64'],
        ['ir_r3_k5_s1_e6_c112', 'ir_r1_k5_s1_e3_c112'],
        ['ir_r4_k5_s2_e6_c184'],
        ['ir_r1_k3_s1_e6_c352'],
    ]
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def),
        stem_size=16,
        num_features=1984,  # paper suggests this, but is not 100% clear
        channel_multiplier=channel_multiplier,
        norm_kwargs=resolve_bn_args(kwargs),
        **kwargs
    )
    model = _create_model(model_kwargs, default_cfgs[variant], pretrained)
    return model


def _gen_spnasnet(variant, channel_multiplier=1.0, pretrained=False, **kwargs):
    """Creates the Single-Path NAS model from search targeted for Pixel1 phone.

    Paper: https://arxiv.org/abs/1904.02877

    Args:
      channel_multiplier: multiplier to number of channels per layer.
    """
    arch_def = [
        # stage 0, 112x112 in
        ['ds_r1_k3_s1_c16_noskip'],
        # stage 1, 112x112 in
        ['ir_r3_k3_s2_e3_c24'],
        # stage 2, 56x56 in
        ['ir_r1_k5_s2_e6_c40', 'ir_r3_k3_s1_e3_c40'],
        # stage 3, 28x28 in
        ['ir_r1_k5_s2_e6_c80', 'ir_r3_k3_s1_e3_c80'],
        # stage 4, 14x14in
        ['ir_r1_k5_s1_e6_c96', 'ir_r3_k5_s1_e3_c96'],
        # stage 5, 14x14in
        ['ir_r4_k5_s2_e6_c192'],
        # stage 6, 7x7 in
        ['ir_r1_k3_s1_e6_c320_noskip']
    ]
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def),
        stem_size=32,
        channel_multiplier=channel_multiplier,
        norm_kwargs=resolve_bn_args(kwargs),
        **kwargs
    )
    model = _create_model(model_kwargs, default_cfgs[variant], pretrained)
    return model


def _gen_efficientnet(variant, channel_multiplier=1.0, depth_multiplier=1.0, pretrained=False, **kwargs):
    """Creates an EfficientNet model.

    Ref impl: https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py
    Paper: https://arxiv.org/abs/1905.11946

    EfficientNet params
    name: (channel_multiplier, depth_multiplier, resolution, dropout_rate)
    'efficientnet-b0': (1.0, 1.0, 224, 0.2),
    'efficientnet-b1': (1.0, 1.1, 240, 0.2),
    'efficientnet-b2': (1.1, 1.2, 260, 0.3),
    'efficientnet-b3': (1.2, 1.4, 300, 0.3),
    'efficientnet-b4': (1.4, 1.8, 380, 0.4),
    'efficientnet-b5': (1.6, 2.2, 456, 0.4),
    'efficientnet-b6': (1.8, 2.6, 528, 0.5),
    'efficientnet-b7': (2.0, 3.1, 600, 0.5),
    'efficientnet-b8': (2.2, 3.6, 672, 0.5),

    Args:
      channel_multiplier: multiplier to number of channels per layer
      depth_multiplier: multiplier to number of repeats per stage

    """
    arch_def = [
        ['ds_r1_k3_s1_e1_c16_se0.25'],
        ['ir_r2_k3_s2_e6_c24_se0.25'],
        ['ir_r2_k5_s2_e6_c40_se0.25'],
        ['ir_r3_k3_s2_e6_c80_se0.25'],
        ['ir_r3_k5_s1_e6_c112_se0.25'],
        ['ir_r4_k5_s2_e6_c192_se0.25'],
        ['ir_r1_k3_s1_e6_c320_se0.25'],
    ]
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def, depth_multiplier),
        num_features=round_channels(1280, channel_multiplier, 8, None),
        stem_size=32,
        channel_multiplier=channel_multiplier,
        act_layer=Swish,
        norm_kwargs=resolve_bn_args(kwargs),
        **kwargs,
    )
    model = _create_model(model_kwargs, default_cfgs[variant], pretrained)
    return model

def _gen_efficientmxnet(variant, channel_multiplier=1.0, depth_multiplier=1.0, pretrained=False, arch_mode = 'large', **kwargs):
    """Creates an EfficientNet model.

    Ref impl: https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py
    Paper: https://arxiv.org/abs/1905.11946

    EfficientNet params
    name: (channel_multiplier, depth_multiplier, resolution, dropout_rate)
    'efficientnet-b0': (1.0, 1.0, 224, 0.2),
    'efficientnet-b1': (1.0, 1.1, 240, 0.2),
    'efficientnet-b2': (1.1, 1.2, 260, 0.3),
    'efficientnet-b3': (1.2, 1.4, 300, 0.3),
    'efficientnet-b4': (1.4, 1.8, 380, 0.4),
    'efficientnet-b5': (1.6, 2.2, 456, 0.4),
    'efficientnet-b6': (1.8, 2.6, 528, 0.5),
    'efficientnet-b7': (2.0, 3.1, 600, 0.5),
    'efficientnet-b8': (2.2, 3.6, 672, 0.5),

    Args:
      channel_multiplier: multiplier to number of channels per layer
      depth_multiplier: multiplier to number of repeats per stage

    """
    arch_mode = arch_mode
    if arch_mode == 'large':
        arch_def = [
            # ['ds_r1_k3_s1_e1_c32_se0.25'],
            ['i2r_r1_k3_s1_e2_c96_se0.25'],
            ['i2r_r1_k3_s2_e6_c144_se0.25'],
            ['i2r_r3_k5_s2_e6_c192_se0.25'],
            ['i2r_r3_k3_s2_e6_c288_se0.25'],
            ['i2r_r4_k5_s1_e6_c384_se0.25'],
            ['i2r_r4_k5_s2_e6_c576_se0.25'],
            ['i2r_r3_k3_s1_e6_c960_se0.25'],
            ['i2r_r1_k3_s1_e6_c1280_se0.25'],
        ]
    elif arch_mode == 'large-ghost':
        arch_def = [
            # ['ds_r1_k3_s1_e1_c32_se0.25'],
            ['i2rghost_r1_k3_s1_e2_c96_se0.25'],
            ['i2rghost_r1_k3_s2_e6_c144_se0.25'],
            ['i2rghost_r3_k5_s2_e6_c192_se0.25'],
            ['i2rghost_r3_k3_s2_e6_c288_se0.25'],
            ['i2rghost_r4_k5_s1_e6_c384_se0.25'],
            ['i2rghost_r4_k5_s2_e6_c576_se0.25'],
            ['i2rghost_r3_k3_s1_e6_c960_se0.25'],
            ['i2rghost_r1_k3_s1_e6_c1280_se0.25'],
        ]
    elif arch_mode == 'large_v1':
        arch_def = [
            ['ds_r1_k3_s1_e1_c32_se0.25'],
            ['i2r_r1_k3_s1_e2_c96_se0.25'],
            ['i2r_r1_k3_s2_e6_c144_se0.25'],
            ['i2r_r3_k5_s2_e6_c192_se0.25'],
            ['i2r_r3_k5_s2_e6_c288_se0.25'],
            ['i2r_r4_k3_s1_e6_c384_se0.25'],
            ['i2r_r4_k5_s2_e6_c576_se0.25'],
            ['i2r_r3_k3_s1_e6_c960_se0.25'],
        ]
    elif arch_mode == 'efficientnet':
        arch_def = [
            ['ds_r1_k3_s1_e1_c96_se0.25'],
            ['i2r_r2_k3_s2_e6_c144_se0.25'],
            ['i2r_r2_k5_s2_e6_c192_se0.25'],
            ['i2r_r3_k3_s2_e6_c288_se0.25'],
            ['i2r_r3_k5_s1_e6_c384_se0.25'],
            ['i2r_r4_k5_s2_e6_c576_se0.25'],
            ['i2r_r1_k3_s1_e6_c960_se0.25'],
        ]
    elif arch_mode == 'mbv3':
        arch_def = [
              'i2r_r1_k3_s11_e1_c16_se0.25',
              'i2r_r1_k3_s11_e2_c64_se0.25', 'i2r_r1_k3_s11_e3_c72_se0.25',
              'i2r_r1_k5_s11_e3_c72_se0.25', 'i2r_r1_k5_s22_e4_c120_se0.25',
              'i2r_r2_k5_s11_e4_c120_se0.25', 'i2r_r4_k5_s11_e3_c240_se0.25',
              'i2r_r2_k5_s22_e3_c200_se0.25', 'i2r_r4_k5_s11_e3_c184_se0.25',
              'i2r_r2_k5_s11_e2_c184_se0.25', 'i2r_r4_k5_s11_e2_c480_se0.25',
              'i2r_r2_k5_s11_e6_c672_se0.25', 'i2r_r4_k5_s22_e6_c960_se0.25',
              'i2r_r2_k3_s11_e6_c960_se0.25', 'i2r_r1_k3_s11_e6_c1280_se0.25',
        ]
    elif arch_mode == 'small':
        arch_def = [
            ['i2r_r1_k3_s1_e2_c96_se0.25'],
            ['i2r_r1_k3_s2_e6_c144_se0.25'],
            ['i2r_r3_k5_s2_e6_c192_se0.25'],
            ['i2r_r3_k3_s2_e6_c288_se0.25'],
            ['i2r_r4_k5_s1_e6_c384_se0.25'],
            ['i2r_r4_k5_s2_e6_c576_se0.25'],
            ['i2r_r3_k3_s1_e6_c960_se0.25'],
            ['i2r_r1_k3_s1_e6_c1280_se0.25'],
        ]
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def, depth_multiplier),
        num_features=round_channels(1280, channel_multiplier, 8, None),
        stem_size=32,
        channel_multiplier=channel_multiplier,
        act_layer=Swish,
        norm_kwargs=resolve_bn_args(kwargs),
        **kwargs,
    )
    model = _create_model(model_kwargs, default_cfgs[variant], pretrained, efficientmnext=True)
    # import pdb;pdb.set_trace()
    return model

def _gen_efficientnet_edge(variant, channel_multiplier=1.0, depth_multiplier=1.0, pretrained=False, **kwargs):
    """ Creates an EfficientNet-EdgeTPU model

    Ref impl: https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet/edgetpu
    """

    arch_def = [
        # NOTE `fc` is present to override a mismatch between stem channels and in chs not
        # present in other models
        ['er_r1_k3_s1_e4_c24_fc24_noskip'],
        ['er_r2_k3_s2_e8_c32'],
        ['er_r4_k3_s2_e8_c48'],
        ['ir_r5_k5_s2_e8_c96'],
        ['ir_r4_k5_s1_e8_c144'],
        ['ir_r2_k5_s2_e8_c192'],
    ]
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def, depth_multiplier),
        num_features=round_channels(1280, channel_multiplier, 8, None),
        stem_size=32,
        channel_multiplier=channel_multiplier,
        norm_kwargs=resolve_bn_args(kwargs),
        act_layer=nn.ReLU,
        **kwargs,
    )
    model = _create_model(model_kwargs, default_cfgs[variant], pretrained)
    return model


def _gen_efficientnet_condconv(
        variant, channel_multiplier=1.0, depth_multiplier=1.0, experts_multiplier=1, pretrained=False, **kwargs):
    """Creates an EfficientNet-CondConv model.

    Ref impl: https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet/condconv
    """
    arch_def = [
      ['ds_r1_k3_s1_e1_c16_se0.25'],
      ['ir_r2_k3_s2_e6_c24_se0.25'],
      ['ir_r2_k5_s2_e6_c40_se0.25'],
      ['ir_r3_k3_s2_e6_c80_se0.25'],
      ['ir_r3_k5_s1_e6_c112_se0.25_cc4'],
      ['ir_r4_k5_s2_e6_c192_se0.25_cc4'],
      ['ir_r1_k3_s1_e6_c320_se0.25_cc4'],
    ]
    # NOTE unlike official impl, this one uses `cc<x>` option where x is the base number of experts for each stage and
    # the expert_multiplier increases that on a per-model basis as with depth/channel multipliers
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def, depth_multiplier, experts_multiplier=experts_multiplier),
        num_features=round_channels(1280, channel_multiplier, 8, None),
        stem_size=32,
        channel_multiplier=channel_multiplier,
        norm_kwargs=resolve_bn_args(kwargs),
        act_layer=Swish,
        **kwargs,
    )
    model = _create_model(model_kwargs, default_cfgs[variant], pretrained)
    return model


def _gen_mixnet_s(variant, channel_multiplier=1.0, pretrained=False, **kwargs):
    """Creates a MixNet Small model.

    Ref impl: https://github.com/tensorflow/tpu/tree/master/models/official/mnasnet/mixnet
    Paper: https://arxiv.org/abs/1907.09595
    """
    arch_def = [
        # stage 0, 112x112 in
        ['ds_r1_k3_s1_e1_c16'],  # relu
        # stage 1, 112x112 in
        ['ir_r1_k3_a1.1_p1.1_s2_e6_c24', 'ir_r1_k3_a1.1_p1.1_s1_e3_c24'],  # relu
        # stage 2, 56x56 in
        ['ir_r1_k3.5.7_s2_e6_c40_se0.5_nsw', 'ir_r3_k3.5_a1.1_p1.1_s1_e6_c40_se0.5_nsw'],  # swish
        # stage 3, 28x28 in
        ['ir_r1_k3.5.7_p1.1_s2_e6_c80_se0.25_nsw', 'ir_r2_k3.5_p1.1_s1_e6_c80_se0.25_nsw'],  # swish
        # stage 4, 14x14in
        ['ir_r1_k3.5.7_a1.1_p1.1_s1_e6_c120_se0.5_nsw', 'ir_r2_k3.5.7.9_a1.1_p1.1_s1_e3_c120_se0.5_nsw'],  # swish
        # stage 5, 14x14in
        ['ir_r1_k3.5.7.9.11_s2_e6_c200_se0.5_nsw', 'ir_r2_k3.5.7.9_p1.1_s1_e6_c200_se0.5_nsw'],  # swish
        # 7x7
    ]
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def),
        num_features=1536,
        stem_size=16,
        channel_multiplier=channel_multiplier,
        norm_kwargs=resolve_bn_args(kwargs),
        **kwargs
    )
    model = _create_model(model_kwargs, default_cfgs[variant], pretrained)
    return model


def _gen_mixnet_m(variant, channel_multiplier=1.0, depth_multiplier=1.0, pretrained=False, **kwargs):
    """Creates a MixNet Medium-Large model.

    Ref impl: https://github.com/tensorflow/tpu/tree/master/models/official/mnasnet/mixnet
    Paper: https://arxiv.org/abs/1907.09595
    """
    arch_def = [
        # stage 0, 112x112 in
        ['ds_r1_k3_s1_e1_c24'],  # relu
        # stage 1, 112x112 in
        ['ir_r1_k3.5.7_a1.1_p1.1_s2_e6_c32', 'ir_r1_k3_a1.1_p1.1_s1_e3_c32'],  # relu
        # stage 2, 56x56 in
        ['ir_r1_k3.5.7.9_s2_e6_c40_se0.5_nsw', 'ir_r3_k3.5_a1.1_p1.1_s1_e6_c40_se0.5_nsw'],  # swish
        # stage 3, 28x28 in
        ['ir_r1_k3.5.7_s2_e6_c80_se0.25_nsw', 'ir_r3_k3.5.7.9_a1.1_p1.1_s1_e6_c80_se0.25_nsw'],  # swish
        # stage 4, 14x14in
        ['ir_r1_k3_s1_e6_c120_se0.5_nsw', 'ir_r3_k3.5.7.9_a1.1_p1.1_s1_e3_c120_se0.5_nsw'],  # swish
        # stage 5, 14x14in
        ['ir_r1_k3.5.7.9_s2_e6_c200_se0.5_nsw', 'ir_r3_k3.5.7.9_p1.1_s1_e6_c200_se0.5_nsw'],  # swish
        # 7x7
    ]
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def, depth_multiplier, depth_trunc='round'),
        num_features=1536,
        stem_size=24,
        channel_multiplier=channel_multiplier,
        norm_kwargs=resolve_bn_args(kwargs),
        **kwargs
    )
    model = _create_model(model_kwargs, default_cfgs[variant], pretrained)
    return model
# models added by DQ
@register_model
def mbv2_100(pretrained=False,in_chans=3,drop_rate=0.2,drop_connect_rate=0.5,bn_tf=False,bn_momentum=0.9,bn_eps=0.001, global_pool=False, **kwargs):
    """Creates a MixNet Large model. Tensorflow compatible variant
    """
    from .mobilenetv2 import mobilenetv2
    model = mobilenetv2(**kwargs)
    return model
@register_model
def mbv2_140(pretrained=False,in_chans=3,drop_rate=0.2,drop_connect_rate=0.5,bn_tf=False,bn_momentum=0.9,bn_eps=0.001, global_pool=False, **kwargs):
    """Creates a MixNet Large model. Tensorflow compatible variant
    """
    from .mobilenetv2 import mobilenetv2
    kwargs['width_mult'] = 1.4
    model = mobilenetv2(**kwargs)
    return model
@register_model
def mnext_mbv2_cfg(pretrained=False,in_chans=3,drop_rate=0.2,drop_connect_rate=0.5,bn_tf=False,bn_momentum=0.9,bn_eps=0.001, global_pool=False, **kwargs):
    """Creates a MNeXt Large model. Tensorflow compatible variant
    """
    from .mnext import mnext
    model = mnext(**kwargs)
    return model
@register_model
def mnext_s(pretrained=False, margs=None, **kwargs):
    """ MobileNeXt small """
    from .efficient_mnext.model import EfficientNet
    model = EfficientNet.from_name('mnext-b0')
    return model
@register_model
def mnext_l(pretrained=False, margs=None, **kwargs):
    """ MobileNeXt large """
    from .efficient_mnext.model import EfficientNet
     # model = _create_model(model_kwargs, default_cfgs[variant], pretrained)
    model = EfficientNet.from_name('mnext-b1', mgroup=2)
    model.default_cfg = default_cfgs['efficientmnext_l']
    return model
