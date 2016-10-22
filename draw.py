#!/usr/bin/env python
# encoding: utf-8

import time
from pencil import pencil_draw
from color_pencil import color_draw
import argparse

parser = argparse.ArgumentParser(description='Pencil Drawing Program. '
                                             'You will get the productions at the output folder.')

parser.add_argument('--p', action='store_true', default=False,
                    dest='p', help='Add this when you want to try pencil drawing.')

parser.add_argument('--c', action='store_true', default=False,
                    dest='c', help='Add this when you want to try color pencil drawing, '
                                   'please make sure you get opencv installed in your environment.')

parser.add_argument('-img', dest='image', type=str, default='input/sjtu.jpg',
                    help="The path of image you want to try, default is 'img/sjtu.jpg'.")

parser.add_argument('-s', dest="gammaS", type=float, default=1,
                    help='Larger when you want the line of strokes darker, default value is 1.')

parser.add_argument('-i', dest='gammaI', type=float, default=1,
                    help='Larger when you want the color of productions deeper, default value is 1.')

args = parser.parse_args()

if not args.p and not args.c:
    args.p = True

if args.p:
    start = time.time()
    print 'pencil draw begin'
    pencil_draw(path=args.image, gammaS=args.gammaS, gammaI=args.gammaI)
    print 'pencil drawing end'
    print 'time consumes: {0:.2f}s'.format(time.time() - start)

if args.c:
    start = time.time()
    print 'color pencil draw begin'
    color_draw(path=args.image, gammaS=args.gammaS, gammaI=args.gammaI)
    print 'time consumes: {0:.2f}s'.format(time.time() - start)
