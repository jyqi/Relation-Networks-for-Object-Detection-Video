# --------------------------------------------------------
# Relation-Networks-for-Object-Detection-Video
# --------------------------------------------------------

"""
ImageNet LOC database
This class loads ground truth notations from standard ImageNet LOC VOC XML data formats
and transform them into IMDB format. Selective search is used for proposals, see roidb
function. Results are written as the Pascal VOC format. Evaluation is based on mAP
criterion.
"""

import cPickle
import cv2
import os
import numpy as np
import PIL

from imdb import IMDB
from imagenet_loc_eval import loc_eval
from ds_utils import unique_boxes, filter_small_boxes

class ImageNetLOC(IMDB):
    def __init__(self, image_set, root_path, dataset_path, result_path=None):
        """
        fill basic information to initialize imdb
        :param image_set: 2007_trainval, 2007_test, etc
        :param root_path: 'selective_search_data' and 'cache'
        :param dataset_path: data and results
        :return: imdb object
        """
        super(ImageNetLOC, self).__init__('ImageNetLOC', image_set, root_path, dataset_path, result_path)  # set self.name

        if image_set[0].isdigit:
            self.num_classes = int(image_set.split('_')[0])
            self.split = image_set.split('_')[1]
        else:
            self.num_classes = None
            self.split = image_set.split('_')[0]

        self.root_path = root_path
        self.data_path = dataset_path

        self.load_classes()
        self.num_classes = len(self.classes_map)
        self.load_image_set_index()
        self.num_images = len(self.image_set_index)
        print 'num_images', self.num_images

        self.config = {'use_diff': False,
                       'min_size': 2}

    def load_classes(self):
        if self.num_classes:
            synset_file = os.path.join(self.data_path, 'ImageSets', 'LOC_%d_synset_mapping.txt' % self.num_classes)
        else:
            synset_file = os.path.join(self.data_path, 'ImageSets', 'LOC_synset_mapping.txt')

        with open(synset_file, 'r') as f:
            lines = f.readlines()

        self.classes = ['__background__']
        self.classes_map = ['__background__']
        for line in lines:
            self.classes_map.append(line[:9])
            self.classes.append(line[10:])

    def load_image_set_index(self):
        """
        find out which indexes correspond to given image set (train or val)
        :return:
        """
        image_set_index_file = os.path.join(self.data_path, 'ImageSets', 'LOC', 'CLS-LOC', self.image_set + '.txt')
        assert os.path.exists(image_set_index_file), 'Path does not exist: {}'.format(image_set_index_file)
        with open(image_set_index_file) as f:
            self.image_set_index = [x.split()[0] for x in f.readlines()]

    def image_path_from_index(self, index):
        """
        given image index, find out full path
        :param index: index of a specific image
        :return: full path of this image
        """
        image_file = os.path.join(self.data_path, 'Data', 'LOC', 'CLS-LOC', self.split, index + '.JPEG')
        assert os.path.exists(image_file), 'Path does not exist: {}'.format(image_file)
        return image_file


    def gt_roidb(self):
        """
        return ground truth image regions database
        :return: imdb[image_index]['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = [self.load_loc_annotation(index) for index in self.image_set_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb


    def load_loc_annotation(self, index):
        """
        for a given index, load image and bounding boxes info from XML file
        :param index: index of a specific image
        :return: record['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
        """
        import xml.etree.ElementTree as ET
        roi_rec = dict()
        roi_rec['image'] = self.image_path_from_index(index)

        filename = os.path.join(self.data_path, 'Annotations', 'LOC', 'CLS-LOC', self.split, index + '.xml')
        tree = ET.parse(filename)
        size = tree.find('size')
        roi_rec['height'] = float(size.find('height').text)
        roi_rec['width'] = float(size.find('width').text)
        #im_size = cv2.imread(roi_rec['image'], cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION).shape
        #assert im_size[0] == roi_rec['height'] and im_size[1] == roi_rec['width']

        objs = tree.findall('object')
        if not self.config['use_diff']:
            non_diff_objs = [obj for obj in objs if int(obj.find('difficult').text) == 0]
            objs = non_diff_objs
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)

        class_to_index = dict(zip(self.classes_map, range(self.num_classes)))
        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            cls = class_to_index[obj.find('name').text.lower().strip()]
            boxes[ix, :] = [max(x1,0), max(y1,0), x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0

        roi_rec.update({'boxes': boxes,
                        'gt_classes': gt_classes,
                        'gt_overlaps': overlaps,
                        'max_classes': overlaps.argmax(axis=1),
                        'max_overlaps': overlaps.max(axis=1),
                        'flipped': False,
                        'is_gt': np.ones(boxes.shape[0])})
        return roi_rec

    def evaluate_detections(self, detections):
        """
        top level evaluations
        :param detections: result matrix, [bbox, confidence]
        :return: None
        """
        # make all these folders for results
        result_dir = os.path.join(self.result_path, 'results')
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)

        self.write_pascal_results(detections)
        info = self.do_python_eval()
        return info

    def get_confusion_matrix(self, gt_label, pred_label, class_num):
        """
        Calcute the confusion matrix by given label and pred
        :param gt_label: the ground truth label
        :param pred_label: the pred label
        :param class_num: the nunber of class
        :return: the confusion matrix
        """
        index = (gt_label * class_num + pred_label).astype('int32')
        label_count = np.bincount(index)
        confusion_matrix = np.zeros((class_num, class_num))

        for i_label in range(class_num):
            for i_pred_label in range(class_num):
                cur_index = i_label * class_num + i_pred_label
                if cur_index < len(label_count):
                    confusion_matrix[i_label, i_pred_label] = label_count[cur_index]

        return confusion_matrix


    # def get_result_file_template(self, gpu_id):
    #     """
    #     :return: a string template
    #     """
    #     res_file_folder = os.path.join(self.result_path, 'results')
    #     filename = 'det_' + self.image_set + str(gpu_id) + '_{:s}.txt'
    #     path = os.path.join(res_file_folder, filename)
    #     return path

    def get_result_file_template(self):
        """
        :return: a string template
        """
        res_file_folder = os.path.join(self.result_path, 'results')
        filename = 'det_' + self.image_set + '_{:s}.txt'
        path = os.path.join(res_file_folder, filename)
        return path

    def writeloc_results(self, all_boxes):
        """
        write results files in pascal devkit path
        :param all_boxes: boxes to be processed [bbox, confidence]
        :return: None
        """
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print 'Writing {} ImageNetLOC results file'.format(cls)
            filename = self.get_result_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_set_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if len(dets) == 0:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in range(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1, dets[k, 2] + 1, dets[k, 3] + 1))

    def do_python_eval(self):
        """
        python evaluation wrapper
        :return: info_str
        """
        info_str = ''
        annopath = os.path.join(self.data_path, 'Annotations', '{0!s}.xml')
        imageset_file = os.path.join(self.data_path, 'ImageSets', 'LOC', 'CLS-LOC', self.image_set + '.txt')
        annocache = os.path.join(self.cache_path, self.name + '_annotations.pkl')
        aps = []
        # The PASCAL VOC metric changed in 2010
        print 'VOC07 metric'
        info_str += 'VOC07 metric'
        info_str += '\n'
        for cls_ind, cls in enumerate(self.classes_map):
            if cls == '__background__':
                continue
            filename = self.get_result_file_template().format(cls)
            rec, prec, ap = loc_eval(filename, annopath, imageset_file, cls, annocache,
                                     ovthresh=0.5, use_07_metric=True)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            info_str += 'AP for {} = {:.4f}\n'.format(cls, ap)
        print('Mean AP@0.5 = {:.4f}'.format(np.mean(aps)))
        info_str += 'Mean AP@0.5 = {:.4f}\n\n'.format(np.mean(aps))

        info_str += '\n'
        print 'VOC10 metric'
        info_str += 'VOC10 metric'
        info_str += '\n'
        # @0.7
        aps = []
        for cls_ind, cls in enumerate(self.classes_map):
            if cls == '__background__':
                continue
            filename = self.get_result_file_template().format(cls)
            rec, prec, ap = loc_eval(filename, annopath, imageset_file, cls, annocache,
                                     ovthresh=0.7, use_07_metric=False)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            info_str += 'AP for {} = {:.4f}\n'.format(cls, ap)
        print('Mean AP@0.7 = {:.4f}'.format(np.mean(aps)))
        info_str += 'Mean AP@0.7 = {:.4f}'.format(np.mean(aps))
        return info_str
