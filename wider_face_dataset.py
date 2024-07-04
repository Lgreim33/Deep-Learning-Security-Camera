# coding: utf-8

import numpy as np
import os
import scipy.io

import chainer
import cv2
from tqdm import tqdm




class WIDERFACEDataset(chainer.dataset.DatasetMixin):

    def __init__(self, data_dir, label_mat_file,
                 use_difficult=False, return_difficult=False, 
                 exclude_file_list=None, logger=None):
        
        self.data_dir = data_dir
        self.label_mat_file = label_mat_file
        self.use_difficult = use_difficult
        self.return_difficult = return_difficult
        
        self.logger = logger #for 
        # list up files
        mat = scipy.io.loadmat(self.label_mat_file)
        self.ids = []
        self.bboxs = {}
        self.labels = {}
        self.difficult = {}

        for i in range(len(mat['event_list'])):
            event = mat['event_list'][i,0][0]
            for j in range(len(mat['file_list'][i,0])):
                file = mat['file_list'][i,0][j,0][0]
                filename = "{}.jpg".format(file)
                filepath = os.path.join(data_dir, 'images', event, filename)
                if exclude_file_list != None and filename in exclude_file_list:
                    continue
                # bounding boxes and labels of the picture file
                bboxs = mat['face_bbx_list'][i,0][j,0]
                # convert from (x, y, w, h) to (x1, y1, x2, y2)
                swapped_bbxs = bboxs[:, [0,1,2,3]] #  (y,x,h,w)
                swapped_bbxs[:,2:4] = swapped_bbxs[:,2:4] + swapped_bbxs[:,0:2]
               
                
                invalid_labels = mat['invalid_label_list'][i,0][j,0].ravel()
                pose_labels = mat['pose_label_list'][i,0][j,0].ravel()
                illum_labels = mat['illumination_label_list'][i,0][j,0].ravel()
                occlusion_labels = mat['occlusion_label_list'][i,0][j,0].ravel()
                blur_labels = mat['blur_label_list'][i,0][j,0].ravel()
                expression_labels = mat['expression_label_list'][i,0][j,0].ravel()
                
                self.ids.append(filepath)
                self.bboxs[filepath] = swapped_bbxs.astype(np.float32)
                self.labels[filepath] = np.ones(len(bboxs), dtype=np.int32) #dummy, always 1
                self.difficult[filepath] = invalid_labels

    def __len__(self):
        return len(self.ids)

    def get_example(self, i):
        """Returns the i-th example.
        Returns a color image and bounding boxes. The image is in CHW format.
        The returned image is RGB.
        Args:
            i (int): The index of the example.
        Returns:
            tuple of an image and bounding boxes
        """
        id_ = self.ids[i]
        bbox = self.bboxs[id_].astype(np.float32)
        label = self.labels[id_].astype(np.int32)
        difficult = self.difficult[id_].astype(np.bool_)
        if not self.use_difficult:
            bbox = bbox[np.where(difficult==False)]
            label = label[np.where(difficult==False)]
            difficult = difficult[np.where(difficult==False)]

        # Load a image
        img_file = id_
        img = cv2.imread(img_file)
        #print(img_file)
        if self.logger:
            self.logger.debug(img_file)
        if self.return_difficult:
            return img, bbox, label, difficult
        return img, bbox, label
        


