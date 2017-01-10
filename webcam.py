import argparse
import tools.find_mxnet
import mxnet as mx
import os
import importlib
import sys
from detect.detector import Detector
import cv2
import time
import matplotlib.pyplot as plt

#plt.ion()

CLASSES = ('goose', 'person')
CLASSES = ('goose', 'person', 'golfcart', 'lawnmower', 'dog')
# --cpu --images <> --thresh 0.25 --epoch 200

def get_detector(net, prefix, epoch, data_shape, mean_pixels, ctx,
                 nms_thresh=0.5, force_nms=True):
    """
    wrapper for initialize a detector

    Parameters:
    ----------
    net : str
        test network name
    prefix : str
        load model prefix
    epoch : int
        load model epoch
    data_shape : int
        resize image shape
    mean_pixels : tuple (float, float, float)
        mean pixel values (R, G, B)
    ctx : mx.ctx
        running context, mx.cpu() or mx.gpu(?)
    force_nms : bool
        force suppress different categories
    """
    sys.path.append(os.path.join(os.getcwd(), 'symbol'))
    net = importlib.import_module("symbol_" + net) \
        .get_symbol(len(CLASSES), nms_thresh, force_nms)
    detector = Detector(net, prefix + "_" + str(data_shape), epoch, \
        data_shape, mean_pixels, ctx=ctx)
    return detector

def parse_args():
    parser = argparse.ArgumentParser(description='Single-shot detection network demo')
    parser.add_argument('--network', dest='network', type=str, default='vgg16_reduced',
                        choices=['vgg16_reduced', 'ssd_300'], help='which network to use')
    parser.add_argument('--dir', dest='dir', nargs='?',
                        help='demo image directory, optional', type=str)
    parser.add_argument('--ext', dest='extension', help='image extension, optional',
                        type=str, nargs='?')
    parser.add_argument('--epoch', dest='epoch', help='epoch of trained model',
                        default=0, type=int)
    parser.add_argument('--prefix', dest='prefix', help='trained model prefix',
                        default=os.path.join(os.getcwd(), 'model', 'ssd'), type=str)
    parser.add_argument('--cpu', dest='cpu', help='(override GPU) use CPU to detect',
                        action='store_true', default=False)
    parser.add_argument('--gpu', dest='gpu_id', type=int, default=0,
                        help='GPU device id to detect with')
    parser.add_argument('--data-shape', dest='data_shape', type=int, default=300,
                        help='set image shape')
    parser.add_argument('--mean-r', dest='mean_r', type=float, default=123,
                        help='red mean value')
    parser.add_argument('--mean-g', dest='mean_g', type=float, default=117,
                        help='green mean value')
    parser.add_argument('--mean-b', dest='mean_b', type=float, default=104,
                        help='blue mean value')
    parser.add_argument('--thresh', dest='thresh', type=float, default=0.5,
                        help='object visualize score threshold, default 0.6')
    parser.add_argument('--nms', dest='nms_thresh', type=float, default=0.5,
                        help='non-maximum suppression threshold, default 0.5')
    parser.add_argument('--force', dest='force_nms', type=bool, default=True,
                        help='force non-maximum suppression on different class')
    parser.add_argument('--timer', dest='show_timer', type=bool, default=True,
                        help='show detection time')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    args = parse_args()
    # get initial image and display
    ret, frame = cap.read()
    im = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('temp.jpg',im)
    result=cv2.imread('temp.jpg');
    result=cv2.imread('detection.jpg');
    cv2.imshow('frame',result)
    cv2.waitKey(500)
    if args.cpu:
        ctx = mx.cpu()
    else:
        ctx = mx.gpu(args.gpu_id)

    detector = get_detector(args.network, args.prefix, args.epoch,
                            args.data_shape,
                            (args.mean_r, args.mean_g, args.mean_b),
                            ctx, args.nms_thresh, args.force_nms)
    # run detection in loop
    while True:
        print('capturing frame')
        cap.release()
        cap.open(0)
        ret, frame = cap.read()
        #ret = cap.grab()
        #ret, frame = cap.retrieve()
        # Our operations on the frame come here
        im = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('temp.jpg',im)
        print('  running detector')
        detector.detect_and_visualize_inplace('temp.jpg', args.dir, args.extension,CLASSES, args.thresh, args.show_timer)
        print('  done with detection, displaying result');
        result=cv2.imread('temp_detection.jpg');
        # Display the resulting frame
        #cv2.namedWindow(winname)
        cv2.imshow('frame',result)
        time.sleep(0.1)
        #plt.imshow(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(0.5)

    cap.release()
    cv2.destroyAllWindows()

