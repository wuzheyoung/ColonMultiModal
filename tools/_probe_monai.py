# -*- coding: utf-8 -*-
"""临时探针: 确认 MONAI ResNetFeatures 支持的 block_names (resnet10?) 与 resnet10 预训练权重。"""
import inspect
import re

import monai

print("monai", monai.__version__)
from monai.networks.nets import ResNetFeatures

print("signature:", inspect.signature(ResNetFeatures.__init__))
src = inspect.getsource(ResNetFeatures)
print("block names found:", sorted(set(re.findall(r"resnet\d+", src))))
print("pretrained_urls:", sorted(set(re.findall(r"https?://\S+resnet\d+\S+", src)))[:5])
