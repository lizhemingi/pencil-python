#!/usr/bin/env python
# encoding: utf-8

"""
=================================================
The python version implementation
"Combining Sketch and Tone for Pencil Drawing Production"
Cewu Lu, Li Xu, Jiaya Jia

International Symposium on Non-Photorealistic Animation and Rendering
(NPAR 2012), June, 2012

=================================================
pencil drawing implementation
usage:
    cd {file directory}
    python pencil.py {path of img file you want to try}

"""

from stitch_function import horizontal_stitch as hstitch, vertical_stitch as vstitch
from util import im2double, rot90, rot90c
from natural_histogram_matching import natural_histogram_matching
from PIL import Image
import numpy as np
from scipy import signal
from scipy.ndimage import interpolation
from scipy.sparse import csr_matrix as csr_matrix, spdiags as spdiags
from scipy.sparse.linalg import spsolve as spsolve
import math
import os
import sys
reload(sys)
sys.setdefaultencoding("utf-8")


basedir = os.path.dirname(__file__)
output = os.path.join(basedir, 'output')

line_len_divisor = 40   # 卷积核大小与图片的倍数关系
# gammaS = 1      # 值越大, 轮廓的线条越粗
# gammaI = 1      # 值越大, 最后输出图片的颜色越深

Lambda = 0.2
texture_resize_ratio = 0.2
texture_file_name = 'texture.jpg'


def get_s(J, gammaS=1):
    '''
    产生笔画结构（Stroke Structure Generation ）
    stroke drawing aims at expressing general structures of the scene

    1. classification:
        首先计算图片的梯度, 分别在x和y两个方向上进行计算, 然后在相应位置求平方和 开根号
        因为存在噪声的干扰, 直接使用计算梯度生成的轮廓效果不好
        论文采用的方式是通过预测每个像素点的方向的方式生成线条
        一共分成8个方向, 每个45度, 计算得到8个方向向量, 同时作为卷积核
        每个像素点的方向是卷积过后8个值里面最大的那个表示的方向, 获得一个map set c, 用1表示方向, 其余为0
    2. line shaping:
        生成轮廓线条的过程
        通过map set c与方向向量做卷积, 将同一个方向上的像素点聚合起来
        同时可以将原本梯度图中的边缘像素点连接到线段中

    :param J:   图片转换成灰度后的矩阵
    :gammaS:    控制参数, 值越大线条越粗
    :return:    图片的笔画结构, 轮廓线S
    '''
    h, w = J.shape
    line_len_double = float(min(h, w)) / line_len_divisor

    line_len = int(line_len_double)
    line_len += line_len % 2

    half_line_len = line_len / 2

    # 计算梯度
    # compute the image gradient 'Imag'
    dJ = im2double(J)
    Ix = np.column_stack((abs(dJ[:, 0:-1] - dJ[:, 1:]), np.zeros((h, 1))))
    Iy = np.row_stack((abs(dJ[0:-1, :] - dJ[1:, :]), np.zeros((1, w))))
    # eq.1
    Imag = np.sqrt(Ix*Ix + Iy*Iy)

    # 注释下面一行代码可以看到通过简单求梯度的方式进行轮廓结构的产生方式, 容易被噪声影响
    # Image.fromarray((1 - Imag) * 255).show()

    # create the 8 directional line segments L
    # L[:, :, index]是一个用来表示第index+1个方向的线段
    # 是一个卷积核
    L = np.zeros((line_len, line_len, 8))
    for n in range(8):
        if n == 0 or n == 1 or n == 2 or n == 7:
            for x in range(0, line_len):
                y = round(((x+1) - half_line_len) * math.tan(math.pi/8*n))
                y = half_line_len - y
                if 0 < y <= line_len:
                    L[int(y-1), x, n] = 1
                if n < 7:
                    L[:, :, n+4] = rot90c(L[:, :, n])
    L[:, :, 3] = rot90(L[:, :, 7])

    G = np.zeros((J.shape[0], J.shape[1], 8))
    for n in range(8):
        G[:, :, n] = signal.convolve2d(Imag, L[:, :, n], "same")    # eq.2

    Gindex = G.argmax(axis=2)   # 获取最大值元素所在的下标 axis表示维度
    # C is map set
    C = np.zeros((J.shape[0], J.shape[1], 8))
    for n in range(8):
        # 八个方向  选取最大值所在的方向
        # eq.3  论文中公式与解释有出入 选取的应该是最大值方向
        C[:, :, n] = Imag * (1 * (Gindex == n))

    # line shaping
    # generate lines at each pixel
    Spn = np.zeros((J.shape[0], J.shape[1], 8))
    for n in range(8):
        Spn[:, :, n] = signal.convolve2d(C[:, :, n], L[:, :, n], "same")

    # 八个方向的求和, 并执行归一化操作
    Sp = Spn.sum(axis=2)
    Sp = (Sp - Sp[:].min()) / (Sp[:].max() - Sp[:].min())
    S = (1 - Sp) ** gammaS

    img = Image.fromarray(S * 255)
    # img.show()

    return S


def get_t(J, type, gammaI=1):
    '''
    色调渲染(tone rendering):
    Tone Rendering tone drawing focuses more on shapes, shadow, and shading than on the use of lines
    铅笔画的直方图有一定的pattern, 因为只是铅笔和白纸的结合
    可以分成三个区域: 1.亮 2.暗 3.居于中间的部分, 于是就有三个用来模拟的模型
    铅笔画的色调 颜色等通过用铅笔重复的涂画来体现

    1. 直方图匹配
        运用三种分布计算图片的直方图, 然后匹配一个正常图片的直方图
    2. 纹理渲染(texture rendering):
        计算模拟需要用铅笔重复涂画的次数beta

    :param J:       图片转换成灰度后的矩阵
    :param type:    图片类型
    :param gammaI:  控制参数, 值越大最后的结果颜色越深
    :return:        色调渲染后的图片矩阵T
    '''

    Jadjusted = natural_histogram_matching(J, type=type) ** gammaI
    # Jadjusted = natural_histogram_matching(J, type=type)

    texture = Image.open(texture_file_name)
    texture = np.array(texture.convert("L"))
    # texture = np.array(texture)
    texture = texture[99: texture.shape[0]-100, 99: texture.shape[1]-100]

    ratio = texture_resize_ratio * min(J.shape[0], J.shape[1]) / float(1024)
    texture_resize = interpolation.zoom(texture, (ratio, ratio))
    texture = im2double(texture_resize)
    htexture = hstitch(texture, J.shape[1])
    Jtexture = vstitch(htexture, J.shape[0])

    size = J.shape[0] * J.shape[1]

    nzmax = 2 * (size-1)
    i = np.zeros((nzmax, 1))
    j = np.zeros((nzmax, 1))
    s = np.zeros((nzmax, 1))
    for m in range(1, nzmax+1):
        i[m-1] = int(math.ceil((m+0.1) / 2)) - 1
        j[m-1] = int(math.ceil((m-0.1) / 2)) - 1
        s[m-1] = -2 * (m % 2) + 1
    dx = csr_matrix((s.T[0], (i.T[0], j.T[0])), shape=(size, size))

    nzmax = 2 * (size - J.shape[1])
    i = np.zeros((nzmax, 1))
    j = np.zeros((nzmax, 1))
    s = np.zeros((nzmax, 1))
    for m in range(1, nzmax+1):
        i[m-1, :] = int(math.ceil((m-1+0.1)/2) + J.shape[1] * (m % 2)) - 1
        j[m-1, :] = math.ceil((m-0.1)/2) - 1
        s[m-1, :] = -2 * (m % 2) + 1
    dy = csr_matrix((s.T[0], (i.T[0], j.T[0])), shape=(size, size))

    # +0.01是为了避免出现有0被进行log运算的情况, 但对正常值影响可以被忽略
    Jtexture1d = np.log(np.reshape(Jtexture.T, (1, Jtexture.size), order="f") + 0.01)
    Jtsparse = spdiags(Jtexture1d, 0, size, size)
    Jadjusted1d = np.log(np.reshape(Jadjusted.T, (1, Jadjusted.size), order="f").T + 0.01)

    nat = Jtsparse.T.dot(Jadjusted1d)   # lnJ(x)
    a = np.dot(Jtsparse.T, Jtsparse)
    b = dx.T.dot(dx)
    c = dy.T.dot(dy)
    mat = a + Lambda * (b + c)     # lnH(x)

    # x = spsolve(a,b) <--> a*x = b
    # lnH(x) * beta(x) = lnJ(x) --> beta(x) = spsolve(lnH(x), lnJ(x))
    # 使用sparse matrix的spsolve 而不是linalg.solve()
    beta1d = spsolve(mat, nat)  # eq.8
    beta = np.reshape(beta1d, (J.shape[0], J.shape[1]), order="c")

    # 模拟素描时通过重复画线来加深阴影, 用pattern Jtexture重复画beta次
    T = Jtexture ** beta    # eq.9
    T = (T - T.min()) / (T.max() - T.min())

    img = Image.fromarray(T * 255)
    # img.show()

    return T


def pencil_draw(path="img/sjtu.jpg", gammaS=1, gammaI=1):
    name = path.rsplit("/")[-1].split(".")[0]
    suffix = path.rsplit("/")[-1].split(".")[1]

    imr = Image.open(path)
    type = "colour" if imr.mode == "RGB" else "black"
    im = imr.convert("L")
    J = np.array(im)
    S = get_s(J, gammaS=gammaS)
    T = get_t(J, type, gammaI=gammaI)
    IPencil = S * T
    img = Image.fromarray(IPencil * 255)
    # img.show()

    save_output(Image.fromarray(S * 255), name + "_s", suffix)
    save_output(Image.fromarray(T * 255), name + "_t", suffix)
    save_output(img, name + "_pencil", suffix)

    return name + suffix


def make_output_dir():
    if not os.path.exists(output):
        os.mkdir(output)


def save_output(img, name, suffix):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    make_output_dir()
    name = os.path.join(output, name)
    filename = "{0}.{1}".format(name, suffix)
    img.save(filename)


if __name__ == "__main__":
    args = sys.argv
    length = len(args)
    if length > 1:
        path = args[1]
        pencil_draw(path=path)
    else:
        pencil_draw()
