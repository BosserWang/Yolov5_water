# !/usr/bin/env python
# -*- encoding: utf-8 -*-

import os

img_path = 'VOCdevkit/VOC2007/Annotations'
img_list = os.listdir(img_path)
print('img_list: ', img_list)

with open('train.txt', 'w') as f:
    for img_name in img_list:
        f.write(img_name + '\n')