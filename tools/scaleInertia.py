#!/usr/bin/env python3
#-*- coding: utf-8 -*-

import numpy as np
import re

if __name__ == '__main__':
    line = '<inertia ixx="0.00232" ixy="0" ixz="0" iyy="0.00296" iyz="0" izz="0.00136"/>'

    scaling = 1.95833333333333
    nums = np.array(re.findall(r'[-+]?\d*\.\d+|\d+', line)).astype(float)
    nums *= scaling
    nums = nums.astype(str)

    pieces = []
    old_end = 0
    for m in re.finditer(r'[-+]?\d*\.\d+|\d+', line):
        pieces.append(line[old_end:m.start()])
        old_end = m.end()

    new_line = ""
    i = 0
    for p in pieces:
        new_line += p + nums[i]
        i += 1
    new_line += line[old_end:]
    print(new_line)
