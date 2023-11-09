"""
Script to generate 2D slices from 3D OAI data. First read the 3D data and the corresponding segmentation masks, then
crop each volume into its cartilage region and save the slices from the cropped volumes as png images. Seperate the
sliced images according to the kl grade of the corresponding volume (in the name of the image file), i.e. each kl grade
has its own folder.
"""

import os
import numpy as np
import nibabel as nib
import tqdm
import cv2
from PIL import Image


# image dir, image format: 'datasets/OAI/out/imgs/id=9003380_klg=1.0_lat=Right_tp=0_cohort=C_version=2_shape=(160, 384, 384)_bbox=[31:140, 117:288, 116:236].nii.gz'
img_dir = "datasets/OAI/out/imgs"
# label dir, label format: 'datasets/OAI/out/segs/id=9003380_klg=1.0_lat=Right_tp=0_cohort=C_version=2_shape=(160, 384, 384)_bbox=[31:140, 117:288, 116:236].nii.gz' (same as image name)
seg_dir = "datasets/OAI/out/segs"


out_dir = "datasets/OAI/oai_slices_cart_touching"
if not os.path.exists(out_dir):
    os.mkdir(out_dir)


def read_image_segmentation_pairs():
    """
    Read the image and segmentation pairs from the image and segmentation directories.
    :return: a list of tuples, each tuple contains the image and segmentation pair
    """
    image_segmentation_pairs = []
    for image_name in os.listdir(img_dir):
        image_path = os.path.join(img_dir, image_name)
        segmentation_path = os.path.join(seg_dir, image_name)
        # verify file existence
        if not os.path.exists(image_path):
            raise ValueError("Image file {} does not exist".format(image_path))
        if not os.path.exists(segmentation_path):
            raise ValueError(
                "Segmentation file {} does not exist".format(segmentation_path)
            )
        image_segmentation_pairs.append((image_path, segmentation_path))
    return image_segmentation_pairs


def centre_crop_by_cartilage(img2d, seg2d, crop_size, use_centre_of_mass=False):
    mask = np.zeros_like(seg2d)
    mask[seg2d != 0] = 1
    cart_pixels = np.where(mask == 1)
    cart_centre = np.array(cart_pixels).mean(axis=1)
    if not use_centre_of_mass: 
        # center is the exact middle between the two cartilage regions (2 and 4)
        cart2_pixels = np.where(seg2d == 2)
        cart4_pixels = np.where(seg2d == 4)
        right_most_cart4 = np.max(cart4_pixels[1])
        left_most_cart2 = np.min(cart2_pixels[1])
        horizontal_mid = (right_most_cart4 + left_most_cart2) // 2
        cart_centre[1] = horizontal_mid

        # cart_centre[0] is set as the middle of the cartilage region 4
        cart_centre[0] = np.median(cart4_pixels[0])
        

    halfsize = crop_size // 2
    # draw centerlines on seg2d
    # seg2d[cart_centre[0], :] = 5
    # seg2d[:, cart_centre[1]] = 5


    # crop around cartilage centre, if it reaches the edge, extend the other side
    if cart_centre[0] < halfsize:
        cart_centre[0] = halfsize
    elif cart_centre[0] > img2d.shape[0] - halfsize:
        cart_centre[0] = img2d.shape[0] - halfsize

    if cart_centre[1] < halfsize:
        cart_centre[1] = halfsize
    elif cart_centre[1] > img2d.shape[1] - halfsize:
        cart_centre[1] = img2d.shape[1] - halfsize

    cart_centre = (int(cart_centre[0]), int(cart_centre[1]))
    # crop image and segmentation
    img2d_crop = img2d[
        cart_centre[0] - halfsize : cart_centre[0] + halfsize,
        cart_centre[1] - halfsize : cart_centre[1] + halfsize,
    ]
    seg2d_crop = seg2d[
        cart_centre[0] - halfsize : cart_centre[0] + halfsize,
        cart_centre[1] - halfsize : cart_centre[1] + halfsize,
    ]

    return img2d_crop, seg2d_crop


def process_img_seg_pairs(
    x,
    y,
    img_dir,
    seg_dir,
    non_cartilage_classes=[1, 3],
    crop_size=200,
    use_centre_of_mass=False,
    min_cart_dist=1,
):
    x = nib.load(x)
    y = nib.load(y)
    img = np.array(x.dataobj)
    seg = np.array(y.dataobj)

    # set everything but cartilage to 0
    for c in non_cartilage_classes:
        seg[seg == c] = 0
    
    touching_img = []
    touching_seg = []
    # for all the slices along dimension 0
    for i in range(img.shape[0]):
        s = seg[i]
        # cartilage 2 indices
        c2 = np.where(s == 2)
        # cartilage 4 indices
        c4 = np.where(s == 4)

        touching = False
        for x2, y2 in zip(*c2):
            for x4, y4 in zip(*c4):
                # there are two cases where the cartilage 2 and 4 with touching distance defined by min_cart_dist
                # case 1: same x, abs(y2 - y4) <= min_cart_dist
                # case 2: same y, abs(x2 - x4) <= min_cart_dist

                # case 1
                if x2 == x4 and abs(y2 - y4) <= min_cart_dist:
                    touching = True
                    break
                # case 2
                if y2 == y4 and abs(x2 - x4) <= min_cart_dist:
                    touching = True
                    break
            if touching:
                break
        
        if touching:
            touching_img.append(img[i])
            touching_seg.append(seg[i])

    # combine all the touching slices into a 3D array
    touching_img = np.array(touching_img)
    touching_seg = np.array(touching_seg)



    img = touching_img
    seg = touching_seg


    index = 0
    for img_slice, seg_slice in zip(img, seg):
        cropped_img, cropped_seg = centre_crop_by_cartilage(
            img_slice, seg_slice, crop_size, use_centre_of_mass
        )

        # if the cropped image is not the same size as 256 resize it to 256
        if cropped_img.shape != (256, 256):
            cropped_img = cv2.resize(cropped_img, (256, 256))
            cropped_seg = cv2.resize(cropped_seg, (256, 256), interpolation=cv2.INTER_NEAREST)

        # map to 0-255
        cropped_img = cv2.normalize(cropped_img, None, 0, 255, cv2.NORM_MINMAX)
        cropped_seg = cropped_seg * 255 / 5

        # save as png
        img_name = f"{img_dir}_{index}.png"
        seg_name = f"{seg_dir}_{index}.png"
        Image.fromarray(cropped_img.astype("uint8")).save(img_name)
        Image.fromarray(cropped_seg.astype("uint8")).save(seg_name)
        index += 1


image_segmentation_pairs = read_image_segmentation_pairs()

# for each image, read the image and segmentation, crop the image according to the segmentation, and save the slices
for x, y in tqdm.tqdm(image_segmentation_pairs):
    # get attributes from the image name
    image_name = os.path.basename(x).replace(".nii.gz", "")

    img_id, klg, lat, tp, cohort, version, shape, bbox = image_name.split("_")
    attributes = {}
    for i, attr in enumerate([img_id, klg, lat, tp, cohort, version, shape, bbox]):
        attr_name, attr_value = attr.split("=")
        if i != 0:
            try:
                attr_value = float(attr_value)
            except ValueError:
                pass
        attributes[attr_name] = attr_value
    odimg = os.path.join(out_dir, "imgs")
    odseg = os.path.join(out_dir, "segs")
    imgd = os.path.join(odimg, f"klg={attributes['klg']}")
    segd = os.path.join(odseg, f"klg={attributes['klg']}")
    if not os.path.exists(imgd):
        os.makedirs(imgd)
        os.makedirs(segd)

    img_dir = os.path.join(imgd, image_name)
    seg_dir = os.path.join(segd, image_name)

    process_img_seg_pairs(
        x,
        y,
        img_dir,
        seg_dir,
        non_cartilage_classes=[1, 3],
        crop_size=192,
        # min_cart_size=800,
        use_centre_of_mass=False,
        min_cart_dist=6,
    )
