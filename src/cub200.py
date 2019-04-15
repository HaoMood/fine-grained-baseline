# -*- coding: utf-8 -*
"""This module is served as torchvision.datasets to load CUB200-2011.

CUB200-2011 dataset has 11,788 images of 200 bird species. The project page
is as follows.
    http://www.vision.caltech.edu/visipedia/CUB-200-2011.html
- Images are contained in the directory data/cub200/raw/images/,
  with 200 subdirectories.
- Format of images.txt: <image_id> <image_name>
- Format of train_test_split.txt: <image_id> <is_training_image>
- Format of classes.txt: <class_id> <class_name>
- Format of iamge_class_labels.txt: <image_id> <class_id>

This file is modified from:
    https://github.com/vishwakftw/vision.
"""


import os
import pickle

import numpy as np
import PIL.Image
import torch


torch.set_default_dtype(torch.float32)
torch.set_default_tensor_type(torch.FloatTensor)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.benchmark = True


__all__ = ['CUB200']
__author__ = 'Hao Zhang'
__copyright__ = '2019 LAMDA'
__date__ = '2018-02-27'
__email__ = 'zhangh0214@gmail.com'
__license__ = 'CC BY-SA 4.0'
__status__ = 'Development'
__updated__ = '2019-04-08'
__version__ = '3.0'


class CUB200(torch.utils.data.Dataset):
    """CUB200 dataset.

    Args:
        _root, str: Root directory of the dataset.
        _train, bool: Load train/test data.
        _transform, callable: A function/transform that takes in a PIL.Image
            and transforms it.
        _target_transform, callable: A function/transform that takes in the
            target and transforms it.
        _train_instances, list<np.ndarray>.
        _train_labels, list<int>.
        _test_instances, list<np.ndarray>.
        _test_labels, list<int>.
    """
    def __init__(self, root: str, train: bool = True, transform=None,
                 target_transform=None, download: bool = False):
        """Load the dataset.

        Args
            root, str: Root directory of the dataset.
            train, bool [True]: Load train/test data.
            transform, callable [None]: A function/transform that takes in a
                PIL.Image and transforms it.
            target_transform, callable [None]: A function/transform that takes
                in the target and transforms it.
            download, bool [False]: If true, downloads the dataset from the
                internet and puts it in root directory. If dataset is already
                downloaded, it is not downloaded again.
        """
        self._root = os.path.expanduser(root)  # Replace ~ by the complete dir
        self._train = train
        self._transform = transform
        self._target_transform = target_transform
        self.name = 'cub200'

        if (os.path.isfile(os.path.join(self._root, 'train.pkl'))
            and os.path.isfile(os.path.join(self._root, 'test.pkl'))):
            print('Dataset already downloaded and verified.')
        elif download:
            url = ('http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/'
                   'CUB_200_2011.tgz')
            self._download(url)
            self._extract()
        else:
            self._extract()

        # Now load the picked data.
        if self._train:
            self._train_instances, self._train_labels = pickle.load(open(
                os.path.join(self._root, 'train.pkl'), 'rb'))
            assert (len(self._train_instances) == 5994
                    and len(self._train_labels) == 5994)
        else:
            self._test_instances, self._test_labels = pickle.load(open(
                os.path.join(self._root, 'test.pkl'), 'rb'))
            assert (len(self._test_instances) == 5794
                    and len(self._test_labels) == 5794)

    def _download(self, url):
        """Download and uncompress the tar.gz file from a given URL.

        Args:
            url, str: URL to be downloaded.
        """
        import six.moves
        import tarfile

        raw_path = os.path.join(self._root, 'raw')
        processed_path = os.path.join(self._root, 'processed')
        if not os.path.isdir(raw_path):
            os.mkdir(raw_path, mode=0o775)
        if not os.path.isdir(processed_path):
            os.mkdir(processed_path, mode=0o775)

        # Downloads file.
        fpath = os.path.join(self._root, 'raw', 'CUB_200_2011.tgz')
        try:
            print('Downloading ' + url + ' to ' + fpath)
            six.moves.urllib.request.urlretrieve(url, fpath)
        except six.moves.urllib.error.URLError:
            if url[:5] == 'https:':
                self._url = self._url.replace('https:', 'http:')
                print('Failed download. Trying https -> http instead.')
                print('Downloading ' + url + ' to ' + fpath)
                six.moves.urllib.request.urlretrieve(url, fpath)

        # Extract file.
        cwd = os.getcwd()
        tar = tarfile.open(fpath, 'r:gz')
        os.chdir(os.path.join(self._root, 'raw'))
        tar.extractall()
        tar.close()
        os.chdir(cwd)

    def _extract(self):
        """Prepare the data for train/test split and save onto disk."""
        image_path = os.path.join(self._root, 'raw', 'CUB_200_2011', 'images')
        # Format of images.txt: <image_id> <image_name>
        id2name = np.genfromtxt(os.path.join(
            self._root, 'raw', 'CUB_200_2011', 'images.txt'), dtype=str)
        # Format of train_test_split.txt: <image_id> <is_training_image>
        id2train = np.genfromtxt(os.path.join(
            self._root, 'raw', 'CUB_200_2011', 'train_test_split.txt'),
            dtype=int)

        train_instances = []
        train_labels = []
        test_instances = []
        test_labels = []
        for id_ in range(id2name.shape[0]):
            image = PIL.Image.open(os.path.join(image_path, id2name[id_, 1]))
            label = int(id2name[id_, 1][:3]) - 1  # Label starts with 0

            # Convert gray scale image to RGB image.
            if image.getbands()[0] == 'L':
                image = image.convert('RGB')
            image_np = np.array(image)
            image.close()

            if id2train[id_, 1] == 1:
                train_instances.append(image_np)
                train_labels.append(label)
            else:
                test_instances.append(image_np)
                test_labels.append(label)

        torch.save((train_instances, train_labels),
                   os.path.join(self._root, 'processed', 'train.pth'))
        torch.save((test_instances, test_labels),
                   os.path.join(self._root, 'processed', 'test.pth'))

    def __getitem__(self, index):
        """
        Args:
            index, int: Index.

        Returns:
            instance, PIL.Image: Image of the given index.
            label, int: target of the given index.
        """
        if self._train:
            image, label = (self._train_instances[index],
                            self._train_labels[index])
        else:
            image, label = (self._test_instances[index],
                            self._test_labels[index])
        image = PIL.Image.fromarray(image)
        if self._transform is not None:
            image = self._transform(image)
        if self._target_transform is not None:
            label = self._target_transform(label)
        return image, label

    def __len__(self):
        """Length of the dataset.

        Returns:
            length, int: Length of the dataset.
        """
        if self._train:
            return len(self._train_instances)
        return len(self._test_instances)
