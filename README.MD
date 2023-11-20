# Disease Disentanglement GAN (DiDiGAN)
This is the code for the MICCAI 2023 paper [Style-Based Manifold for Weakly-Supervised Disease Characteristic Discovery](https://link.springer.com/chapter/10.1007/978-3-031-43904-9_36).

## Data
- [ADNI dataset](https://adni.loni.usc.edu/) (requires registration)
- AD/CN labels are provided in `./data/*.csv`


## Preprocessin
- Extract the center 40 coronal slices from each 3D image.
- Zero-pad each image to 256 x 256
- Re-normalise image pixel values using ``cv2`` to [0-255]
- Save files as png. Save all the AD images in one folder (e.g. ``./dataset/adni/ad/{unique-name}.png``) and all the CN images in another folder (e.g. ``./dataset/adni/cn/{unique-name}.png``).

## How to run
- Build Docker image as defined by ``./Dockerfile``
- Run ``pip install umap-learn`` using a container. 
- Run ``python main.py --conf training/ce_2class/config.json`` 

Note: before running, update the data location in the ``config.json`` files:
```
...
    "training_set_kwargs": {
        "class_name": "training.dataset.ADNIDataset",
        "path": "dataset/adni",
        "constraint_res": 64
    },
...
```

## TODO:
- New docker container to include the ``umap-learn`` package
- More detailed code documentation.


## Citation

```
@InProceedings{10.1007/978-3-031-43904-9_36,
author="Liu, Siyu
and Liu, Linfeng
and Engstrom, Craig
and To, Xuan Vinh
and Ge, Zongyuan
and Crozier, Stuart
and Nasrallah, Fatima
and Chandra, Shekhar S.",
editor="Greenspan, Hayit
and Madabhushi, Anant
and Mousavi, Parvin
and Salcudean, Septimiu
and Duncan, James
and Syeda-Mahmood, Tanveer
and Taylor, Russell",
title="Style-Based Manifold for Weakly-Supervised Disease Characteristic Discovery",
booktitle="Medical Image Computing and Computer Assisted Intervention -- MICCAI 2023",
year="2023",
publisher="Springer Nature Switzerland",
address="Cham",
pages="368--378",
}

```



