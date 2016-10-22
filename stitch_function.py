#!/usr/bin/env python
# encoding: utf-8

import numpy as np


def horizontal_stitch(I, width):
    '''
    列扩展,直到宽为width
    use alpha blending to smooth the borders in the replication
    0 < I < 1
    :param I:
    :param width:
    :return:
    '''
    Istitched = I
    while Istitched.shape[1] < width:
        window_size = int(round(I.shape[1] / 4))
        left = I[:, (I.shape[1] - window_size) : I.shape[1]]
        right = I[:, 0:window_size]
        aleft = np.zeros((left.shape[0], window_size))
        aright = np.zeros((left.shape[0], window_size))
        for i in range(window_size):
            aleft[:, i] = left[:, i] * (1 - float(i+1)/window_size)
            aright[:, i] = right[:, i] * float(i+1)/window_size
        Istitched = np.column_stack(
            (Istitched[:, 0:(Istitched.shape[1] - window_size)],
             aleft + aright,
             Istitched[:, window_size: Istitched.shape[1]])
        )
    Istitched = Istitched[:, 0:width]
    return Istitched


def vertical_stitch(I, height):
    '''
    行扩展,直到长为height
    use alpha blending to smooth the borders in the replication
    0 < I < 1
    :param I:
    :param height:
    :return:
    '''
    Istitched = I
    while Istitched.shape[0] < height:
        window_size = int(round(I.shape[0] / float(4)))
        up = I[(I.shape[0] - window_size):I.shape[0], :]
        down = I[0:window_size, :]
        aup = np.zeros((window_size, up.shape[1]))
        adown = np.zeros((window_size, up.shape[1]))
        for i in range(window_size):
            aup[i, :] = up[i, :] * (1 - float(i+1)/window_size)
            adown[i, :] = down[i, :] * float(i+1)/window_size
        Istitched = np.row_stack(
            (Istitched[0: Istitched.shape[0] - window_size, :],
             aup + adown,
             Istitched[window_size: Istitched.shape[0], :])
        )
    Istitched = Istitched[0: height, :]
    return Istitched