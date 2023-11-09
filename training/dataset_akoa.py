import torch
import os
from PIL import Image
import numpy as np
import cv2


class AKOADataset(torch.utils.data.Dataset):
    def __init__(self, path, constraint_res=80, variable_constraint_res=False):
        """
        :param path: path to the dataset, structure:
        path
        ├── klg=0.0
        │    ├── img1
        │    ├── img2
        │    ├── ...
        │    └── imgN
        ├── klg=1.0
        │    ├── img1
        │    ├── img2
        │    ├── ...
        │    └── imgN
        ├── ...

        :param constra int_res: constraint resolution
        """
        self.path = path
        self.constraint_res = constraint_res
        self.variable_constraint_res = variable_constraint_res

        # list of list of images, for each kl grade
        self.kl_images = [[], [], [], [], []]
        self.images = []
        self.num_channels = 1

        # read images and put them into the list
        for kl_grade in range(5):
            kl_grade_path = os.path.join(path, "klg={}.0".format(kl_grade))
            for image_name in os.listdir(kl_grade_path):
                image_path = os.path.join(kl_grade_path, image_name)
                self.kl_images[kl_grade].append(image_path)
                self.images.append(image_path)

    @property
    def n_classes(self):
        return len(self.kl_images)

    def __len__(self):
        return len(self.images)

    @property
    def label_dim(self):
        return 512

    @property
    def resolution(self):
        # image resolution
        return self.__getitem__(0)[0].shape[1]

    @property
    def image_shape(self):
        return self.__getitem__(0)[0].shape

    def __getitem__(self, index):
        # randomly sample an image by kl grade, so that the dataset is balanced
        kl_grade = torch.randint(0, 5, (1,)).item()
        image_path = self.kl_images[kl_grade][index % len(self.kl_images[kl_grade])]
        img = Image.open(image_path).convert("L")
        img = np.array(img).astype("float32")

        c = Image.open(image_path).resize(
            (self.constraint_res, self.constraint_res), Image.NEAREST
        )
        c = np.array(c).astype("float32")

        img = cv2.normalize(img.copy(), None, -1, 1, cv2.NORM_MINMAX)
        c = cv2.normalize(c.copy(), None, -1, 1, cv2.NORM_MINMAX)
        return (
            torch.tensor(img[None, ...]).float(),
            torch.tensor(c[None, ...]).float(),
            torch.tensor([kl_grade]),
        )


if __name__ == "__main__":
    dataset = AKOADataset("datasets/OAI/oai_slices_com/imgs")
    print(dataset[0])
