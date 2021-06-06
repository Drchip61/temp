# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 00:20:16 2021

@author: 严天宇
"""
import torch.nn as nn
from ConTNet import build_model

model = build_model(arch='ConT-M', use_avgdown=True, relative=True, qkv_bias=True, pre_norm=True)
net1 = nn.Conv2d(3,64,3,1,1)
model0 = model.layer0
model1 = model.layer1
model2 = model.layer2
model3 = model.layer3
model4 = model.layer4
