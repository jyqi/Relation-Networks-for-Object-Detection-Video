# --------------------------------------------------------
# Relation Networks for Object Detection
# Copyright (c) 2017 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Dazhi Cheng, Jiayuan Gu
# Written by Yuwen Xiong
# --------------------------------------------------------


import numpy as np
from dataset import *


def load_gt_roidb(dataset_name, image_set_name, root_path, dataset_path, result_path=None,
                  flip=False, return_classmap=False):
    """ load ground truth roidb """
    imdb = eval(dataset_name)(image_set_name, root_path, dataset_path, result_path)
    roidb = imdb.gt_roidb()
    if flip:
        roidb = imdb.append_flipped_images(roidb)
    if return_classmap:
        return roidb, imdb.classes_map
    return roidb


def load_proposal_roidb(dataset_name, image_set_name, root_path, dataset_path, result_path=None, rpn_path=None,
                        proposal='rpn', append_gt=True, flip=False, top_roi=-1):
    """ load proposal roidb (append_gt when training) """
    imdb = eval(dataset_name)(image_set_name, root_path, dataset_path, result_path, rpn_path)

    gt_roidb = imdb.gt_roidb()
    roidb = eval('imdb.' + proposal + '_roidb')(gt_roidb, append_gt, top_roi)
    if flip:
        roidb = imdb.append_flipped_images(roidb)
    return roidb

def fix_classes(roibds, classmaps):
    classes = ['__background__']
    for i in range(len(roibds)):
        classmap = classmaps[i]
        for cls in classmap:
            if cls not in classes:
                classes.append(cls)

    for i in range(len(roibds)):
        classmap = classmaps[i]
        for j in range(len(roibds[i])):
            overlaps = np.zeros((len(roibds[i][j]['gt_classes']), len(classes)), dtype=np.float32)

            for k in range(len(roibds[i][j]['gt_classes'])):
                current_class_index = roibds[i][j]['gt_classes'][k]
                new_class_index = classes.index(classmap[current_class_index])
                roibds[i][j]['gt_classes'][k] = new_class_index
                overlaps[k, new_class_index] = 1.0

            roibds[i][j]['gt_overlaps'] = overlaps
            roibds[i][j]['max_classes'] = overlaps.argmax(axis=1)
            roibds[i][j]['max_overlaps'] = overlaps.max(axis=1)

    return roibds, classes

def merge_roidb(roidbs):
    """ roidb are list, concat them together """
    roidb = roidbs[0]
    for r in roidbs[1:]:
        roidb.extend(r)
    return roidb


def filter_roidb(roidb, config):
    """ remove roidb entries without usable rois """

    def is_valid(entry):
        """ valid images have at least 1 fg or bg roi """

        if all(entry['gt_classes'] == 0):
            valid = False
        else:
            overlaps = entry['max_overlaps']
            fg_inds = np.where(overlaps >= config.TRAIN.FG_THRESH)[0]
            bg_inds = np.where((overlaps < config.TRAIN.BG_THRESH_HI) & (overlaps >= config.TRAIN.BG_THRESH_LO))[0]
            valid = len(fg_inds) > 0 or len(bg_inds) > 0
        return valid

    num = len(roidb)
    filtered_roidb = [entry for entry in roidb if is_valid(entry)]
    num_after = len(filtered_roidb)
    print 'filtered %d roidb entries: %d -> %d' % (num - num_after, num, num_after)

    return filtered_roidb
