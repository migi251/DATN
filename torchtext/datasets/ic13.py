from .bases import BaseImageDataset
import os
import json
import numpy as np
from tqdm import tqdm


class IC13(BaseImageDataset):
    dataset_dir = 'IC13'

    def __init__(self, root='./data', ignore_list=None, is_training=True, verbose=True, **kwargs):
        super(IC13, self).__init__(root)
        self.dataset_dir = os.path.join(self.root, self.dataset_dir)
        self.is_training = is_training
        if ignore_list:
            with open(ignore_list) as f:
                ignore_list = f.readlines()
                ignore_list = [line.strip() for line in ignore_list]
        else:
            ignore_list = []
        self.image_dir = os.path.join(
            self.dataset_dir, 'Images', 'Train' if is_training else 'Test')
        self.annotation_dir = os.path.join(
            self.dataset_dir, 'gt', 'Train' if is_training else 'Test')

        self.check_before_run()

        self.image_list = os.listdir(self.image_dir)
        self.image_list = list(filter(lambda img: img.replace(
            '.jpg', '') not in ignore_list, self.image_list))
        self.annotation_list = []
        for i in range(len(self.image_list)):
            image_index = self.image_list[i].replace('.jpg', '')
            self.annotation_list.append(os.path.join(
                self.annotation_dir, 'gt_{}.txt'.format(image_index)))
            self.image_list[i] = os.path.join(
                self.image_dir, self.image_list[i])
        if verbose:
            print('=> IC13 loaded')
            self.print_dataset_statistics(
                self.image_list, self.annotation_list)
        self.train = self.process_dir(self.image_list, self.annotation_list)
        # image_path, annot = self.train[0]
        # print(annot[0][0],annot[0][1],annot[0][2])

    def check_before_run(self):
        """Check if all files are available before going deeper"""
        if not os.path.exists(self.dataset_dir):
            raise RuntimeError(
                '"{}" is not available'.format(self.dataset_dir))
        if not os.path.exists(self.image_dir):
            raise RuntimeError('"{}" is not available'.format(self.image_dir))
        if not os.path.exists(self.annotation_dir):
            raise RuntimeError(
                '"{}" is not available'.format(self.annotation_dir))

    def process_dir(self, image_list, annotation_list):
        dataset = []
        for i in tqdm(range(len(image_list))):
            dataset.append(
                [image_list[i], annotation_list[i], self.parse_txt])
        return dataset

    def parse_txt(self, txt_path):
        """
        .mat file parser
        :param txt_path: (str), txt file path
        :return: (list), TextInstance
        """
        with open(txt_path, 'r') as fp:
            gt = fp.readlines()
        polygons = []
        for i, gt_i in enumerate(gt):
            gt[i] = gt_i.strip().split(' ')
            # if len(gt[i]) != 2:
            #     print(gt[i],txt_path,gt_i)
            assert len(gt[i]) == 5
            text = gt[i][4]
            left = gt[i][0]
            top = gt[i][1]
            right = gt[i][2]
            bottom = gt[i][3]
            orient = 'h'
            x = [left, right, right, left]
            y = [top, top, bottom, bottom]
            pts = np.stack([x, y]).T.astype(np.int32)
            polygons.append([pts, orient, text])
        return polygons


if __name__ == "__main__":
    a = IC13()
