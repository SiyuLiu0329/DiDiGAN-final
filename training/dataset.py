import os
import numpy as np
import PIL.Image
import random
import torch
import cv2

class ADNIDataset3Class(torch.utils.data.Dataset):
    def __init__(self,
                path,
                constraint_res=80,
                 ):
        self.c_size = constraint_res
        self.constraint_res = constraint_res
        self._path = path
        self._zipfile = None
        name = os.path.splitext(os.path.basename(self._path))[0]
        self.name = name
        self.has_labels = True
        self.num_channels = 1

        self._type = 'dir'
        self._all_fnames = {os.path.relpath(os.path.join(
            root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files}

        PIL.Image.init()
        self._image_fnames = sorted(
            fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        self.pos, self.neg = [], []
        self.mid = []
        for f in self._image_fnames:
            if f[:2] == 'ad':
                self.pos.append(f)
            elif f[:2] == 'cn':
                self.neg.append(f)
            elif f[:3] == 'mci':
                self.mid.append(f)
            else:
                raise ValueError
        super().__init__()
    
    @property
    def n_classes(self):
        return 3


    def __len__(self):
        return len(self._image_fnames)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    @property
    def image_shape(self):
        return self._random_sample(0)[0].shape

    @property
    def label_dim(self):
        return 512

    @property
    def resolution(self):
        assert len(self.image_shape) == 3
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    def __getitem__(self, idx):
        tgt = random.choice([0, 1, 2])
        return self._random_sample(tgt)

    def _random_sample(self, target):
        if target == 0:
            s = self.neg
        elif target == 1:
            s = self.mid
        elif target == 2:
            s = self.pos
        else:
            raise ValueError
        f = random.choice(s)

        img = PIL.Image.open(os.path.join(self._path, f))
        img = np.array(img).astype('float32')
        c = PIL.Image.open(os.path.join(self._path, f)).resize(
            (self.c_size, self.c_size), PIL.Image.NEAREST)
        c = np.array(c).astype('float32')
        img = cv2.normalize(img.copy(), None, -1, 1, cv2.NORM_MINMAX)
        c = cv2.normalize(c.copy(), None, -1, 1, cv2.NORM_MINMAX)
    
        return torch.tensor(img[None, ...]).float(), torch.tensor(c[None, ...]).float(), torch.tensor([target])

    def load(self, index):

        f = self._image_fnames[index]
        img = PIL.Image.open(os.path.join(self._path, f))
        img = np.array(img).astype('float32')
        c = PIL.Image.open(os.path.join(self._path, f)).resize(
            (self.c_size, self.c_size), PIL.Image.NEAREST)
        c = np.array(c).astype('float32')

        img = cv2.normalize(img.copy(), None, -1, 1, cv2.NORM_MINMAX)
        c = cv2.normalize(c.copy(), None, -1, 1, cv2.NORM_MINMAX)
    
        return torch.tensor(img[None, ...]).float(), torch.tensor(c[None, ...]).float(), f

    def load_with_label(self, index):
        s = self.neg + self.mid + self.pos
        t = [0] * len(self.neg) + [1] * len(self.mid)  + [2] * len(self.pos)
        target = t[index]
        f = s[index]
        img = PIL.Image.open(os.path.join(self._path, f))
        img = np.array(img).astype('float32')
        c = PIL.Image.open(os.path.join(self._path, f)).resize(
            (self.constraint_res, self.constraint_res), PIL.Image.NEAREST)
        c = np.array(c).astype('float32')

        img = cv2.normalize(img.copy(), None, -1, 1, cv2.NORM_MINMAX)
        c = cv2.normalize(c.copy(), None, -1, 1, cv2.NORM_MINMAX)
    
        return torch.tensor(img[None, ...]).float(), torch.tensor(c[None, ...]).float(), torch.tensor([target])
            

class ADNIDataset(torch.utils.data.Dataset):
    def __init__(self,
                 path,
                 constraint_res=80,
                 variable_constraint_res=False
                 ):
        self.constraint_res = constraint_res
        self._path = path
        self._zipfile = None
        name = os.path.splitext(os.path.basename(self._path))[0]
        self.name = name
        self.has_labels = True
        self.num_channels = 1
        self.variable_constraint_res = variable_constraint_res

        self._type = 'dir'
        self._all_fnames = {os.path.relpath(os.path.join(
            root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files}

        PIL.Image.init()
        self._image_fnames = sorted(
            fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        self.pos, self.neg = [], []
        for f in self._image_fnames:
            if f[:2] == 'ad':
                self.pos.append(f)
            elif f[:2] == 'cn':
                self.neg.append(f)
            else:
                raise ValueError
        super().__init__()

    def __len__(self):
        return len(self._image_fnames)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    @property
    def image_shape(self):
        return self._random_sample(0)[0].shape

    @property
    def label_dim(self):
        return 512

    @property
    def n_classes(self):
        return 2

    @property
    def resolution(self):
        assert len(self.image_shape) == 3
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    def __getitem__(self, idx):
        if random.random() > 0.5:
            return self._random_sample(0)
        else:
            return self._random_sample(1)

    def load_with_label(self, index):
        s = self.neg + self.pos
        t = [0] * len(self.neg) + [1] * len(self.pos)
        target = t[index]
        f = s[index]
        img = PIL.Image.open(os.path.join(self._path, f))
        img = np.array(img).astype('float32')
        c = PIL.Image.open(os.path.join(self._path, f)).resize(
            (self.constraint_res, self.constraint_res), PIL.Image.NEAREST)
        c = np.array(c).astype('float32')

        img = cv2.normalize(img.copy(), None, -1, 1, cv2.NORM_MINMAX)
        c = cv2.normalize(c.copy(), None, -1, 1, cv2.NORM_MINMAX)
    
        return torch.tensor(img[None, ...]).float(), torch.tensor(c[None, ...]).float(), torch.tensor([target])
            

    def _random_sample(self, target):
        if target == 0:
            s = self.neg
        elif target == 1:
            s = self.pos
        else:
            raise ValueError
        f = random.choice(s)

        img = PIL.Image.open(os.path.join(self._path, f))
        img = np.array(img).astype('float32')
        c = PIL.Image.open(os.path.join(self._path, f)).resize(
            (self.constraint_res, self.constraint_res), PIL.Image.NEAREST)
        c = np.array(c).astype('float32')

        img = cv2.normalize(img.copy(), None, -1, 1, cv2.NORM_MINMAX)
        c = cv2.normalize(c.copy(), None, -1, 1, cv2.NORM_MINMAX)
    
        return torch.tensor(img[None, ...]).float(), torch.tensor(c[None, ...]).float(), torch.tensor([target])

    def load(self, index):

        f = self._image_fnames[index]
        img = PIL.Image.open(os.path.join(self._path, f))
        img = np.array(img).astype('float32')
        c = PIL.Image.open(os.path.join(self._path, f)).resize(
            (self.constraint_res, self.constraint_res), PIL.Image.NEAREST)
        c = np.array(c).astype('float32')

        img = cv2.normalize(img.copy(), None, -1, 1, cv2.NORM_MINMAX)
        c = cv2.normalize(c.copy(), None, -1, 1, cv2.NORM_MINMAX)
    
        return torch.tensor(img[None, ...]).float(), torch.tensor(c[None, ...]).float(), f