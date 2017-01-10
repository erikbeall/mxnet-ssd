"""
given a SSD/YOLO imdb, compute mAP
"""

import numpy as np
import os
import cPickle

CLASSES = ['goose', 'person','golfcart','lawncare','dog']

def parse_ssd_rec(filename):
    """
    parse ssd record into a dictionary
    :param filename: txt file path
    :return: list of dict
    NOTE: original (yolo) label file is in xcenter, ycenter, width, height
          while bounding box detections are xmin, ymin, xmax, ymax
    """
    objects = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            obj_dict = dict()
            temp_label = line.strip().split()
            assert len(temp_label) == 5, "Invalid label file" + label_file
            cls_id = int(temp_label[0])
            x = float(temp_label[1])
            y = float(temp_label[2])
            half_width = float(temp_label[3]) / 2
            half_height = float(temp_label[4]) / 2
            xmin = x - half_width
            ymin = y - half_height
            xmax = x + half_width
            ymax = y + half_height
            obj_dict['bbox'] = [xmin,ymin,xmax,ymax]
            obj_dict['name'] = CLASSES[cls_id]
            objects.append(obj_dict)
    return objects


def ssd_ap(rec, prec):
    """
    average precision calculations
    [precision integrated to recall]
    :param rec: recall
    :param prec: precision
    :return: average precision
    """
    # append sentinel values at both ends
    #print('rec=%s, prec=%s'%(str(rec),str(prec)))
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute precision integration ladder
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # look for recall value changes
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # sum (\delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def ssd_eval(detpath, annopath, imageset_file, classname, cache_dir, ovthresh=0.5):
    """
    SSD evaluation
    :param detpath: detection results detpath.format(classname)
    :param annopath: annotations annopath.format(classname)
    :param imageset_file: text file containing list of images
    :param classname: category name
    :param cache_dir: caching annotations
    :param ovthresh: overlap threshold
    :return: rec, prec, ap
    """
    if not os.path.isdir(cache_dir):
        os.mkdir(cache_dir)
    cache_file = os.path.join(cache_dir, 'annotations.pkl')
    with open(imageset_file, 'r') as f:
        lines = f.readlines()
    image_filenames = [x.strip() for x in lines]
    recs = {}
    for ind, image_filename in enumerate(image_filenames):
        recs[image_filename] = parse_ssd_rec(annopath.format(image_filename))
        #if ind % 100 == 0:
        #    print 'reading annotations for {:d}/{:d}'.format(ind + 1, len(image_filenames))

    '''
    # load annotations from cache
    if not os.path.isfile(cache_file):
        recs = {}
        for ind, image_filename in enumerate(image_filenames):
            recs[image_filename] = parse_ssd_rec(annopath.format(image_filename))
            if ind % 100 == 0:
                print 'reading annotations for {:d}/{:d}'.format(ind + 1, len(image_filenames))
        print 'saving annotations cache to {:s}'.format(cache_file)
        with open(cache_file, 'w') as f:
            cPickle.dump(recs, f)
    else:
        with open(cache_file, 'r') as f:
            recs = cPickle.load(f)
    '''

    # extract objects in :param classname:
    class_recs = {}
    npos = 0
    for image_filename in image_filenames:
        objects = [obj for obj in recs[image_filename] if obj['name'] == classname]
        # bbox is from the label record - these are positives
        bbox = np.array([x['bbox'] for x in objects])
        det = [False] * len(objects)  # stand for detected
        npos = npos + len(bbox)
        #print('image filename= %s, len(bbox)=%d, len(det)=%d, bbox=%s'%(image_filename,len(bbox),len(det),str(bbox)))
        class_recs[image_filename] = {'bbox': bbox,
                                      'det': det}

    # read detections
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    bbox = np.array([[float(z) for z in x[2:]] for x in splitlines])
    #print('detected bbox = %s, %s, %s'%(str(image_ids),str(confidence),str(bbox)))
    if (len(bbox)==0):
        return (0.0, 0.0, 0.0)

    # sort by confidence
    sorted_inds = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    bbox = bbox[sorted_inds, :]
    image_ids = [image_ids[x] for x in sorted_inds]

    # go down detections and mark true positives and false positives
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        r = class_recs[image_ids[d]]
        bb = bbox[d, :].astype(float)
        ovmax = -np.inf
        bbgt = r['bbox'].astype(float)

        if bbgt.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(bbgt[:, 0], bb[0])
            iymin = np.maximum(bbgt[:, 1], bb[1])
            ixmax = np.minimum(bbgt[:, 2], bb[2])
            iymax = np.minimum(bbgt[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (bbgt[:, 2] - bbgt[:, 0] + 1.) *
                   (bbgt[:, 3] - bbgt[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not r['det'][jmax]:
                tp[d] = 1.
                r['det'][jmax] = 1
            else:
                fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    # precision is the fraction of detections that are true positives (intersection of truth and detected over detecteds)
    # recall is the fraction of true positives that are detected (intersection of truth and detected over truths)
    # combining classification with overlap, we need...
    rec = tp / float(npos)
    # avoid division by zero in case first detection matches a difficult ground ruth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    #print('npos=%s # rec=%s # prec=%s, tp=%s'%(str(npos),str(rec),str(prec),str(tp)))
    ap = ssd_ap(rec, prec)

    return rec, prec, ap
