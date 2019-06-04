from .bases import BaseImageDataset
import os
from scipy import io
import numpy as np
from tqdm import tqdm


class CTW1500(BaseImageDataset):
    dataset_dir = 'ctw1500'

    def __init__(self, root='./data', is_training=True, verbose=True, **kwargs):
        super(CTW1500, self).__init__(root)
        self.dataset_dir = os.path.join(self.root, self.dataset_dir)
        self.is_training = is_training
        ignore_list = []
        self.image_dir = os.path.join(
            self.dataset_dir, 'train' if is_training else 'test', 'text_image')
        self.annotation_dir = os.path.join(
            self.dataset_dir, 'label')

        self.check_before_run()

        self.image_list = os.listdir(self.image_dir)
        self.image_list = list(filter(lambda img: img.replace(
            '.jpg', '') not in ignore_list, self.image_list))
        self.annotation_list = []
        for i in tqdm(range(len(self.image_list))):
            image_index = self.image_list[i].replace('.jpg', '')
            self.annotation_list.append(os.path.join(
                self.annotation_dir, '{}.txt'.format(image_index)))
            self.image_list[i] = os.path.join(
                self.image_dir, self.image_list[i])
        if verbose:
            print('=> ctw1500 loaded')
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
        for i in range(len(image_list)):
            dataset.append([image_list[i], annotation_list[i], self.parse_txt])
        return dataset

    def parse_txt(self, txt_path):
        """
        .mat file parser
        :param txt_path: (str), txt file path
        :return: (list), TextInstance
        """
        with open(txt_path, 'r') as fp:
            gt_txt = fp.readlines()
        len_gt = int(gt_txt[0])
        gt = gt_txt[1:]  # ignore line 0
        assert len_gt == len(gt)
        polygon = []
        for i, gt_i in enumerate(gt):
            gt[i] = gt_i.strip().split(',"')
            # if len(gt[i]) != 2:
            #     print(gt[i],txt_path,gt_i)
            # assert len(gt[i]) == 2
            text = ''.join(gt[i][1:]).replace('"', '')
            ori = 'c' if text != '###' else '#'
            gt[i] = gt[i][0].split(',')
            try:
                gt[i] = np.array(gt[i], np.int32).reshape(-1, 2)
            except:
                print(gt[i],txt_path,gt_i)
            polygon.append([gt[i], ori, text])
        return polygon


if __name__ == "__main__":
    a = CTW1500()
