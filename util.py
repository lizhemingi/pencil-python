#!/usr/bin/env python
# encoding: utf-8

import numpy as np


def im2double(I):
    Min = I.min()
    Max = I.max()
    dis = float(Max - Min)
    m, n = I.shape
    J = np.zeros((m, n), dtype="float")
    for x in range(m):
        for y in range(n):
            a = I[x, y]
            if a != 255 and a != 0:
                b = float((I[x, y] - Min) / dis)
                J[x, y] = b
            J[x, y] = float((I[x, y] - Min) / dis)
    return J


def rot90(I, n=1):
    '''
    逆时针旋转n次90度
    :param I:
    :param n:
    :return:
    '''
    rI = I
    for x in range(n):
        rI = zip(*rI[::-1])
    return rI


def rot90c(I):
    '''
    顺时针旋转90度
    :param I:
    :return:
    '''
    rI = I
    for x in range(3):
        rI = rot90(rI)
    return rI