import random
import numpy as np
import cv2
import os
import copy
import traceback
import matplotlib.pyplot as plt


def possion_fusion(big_image, small_image, fusion_type_list=[cv2.MONOCHROME_TRANSFER, cv2.NORMAL_CLONE]):
    h, w, c = small_image.shape

    mask = np.ones_like(small_image)[..., 0] * 255
    big_image_h, big_image_w, big_image_c = big_image.shape
    diff_w = big_image_w // 2 - w // 2
    diff_h = big_image_h // 2 - h // 2
    center_x = big_image_w // 2 + int(random.uniform(-diff_w, diff_w))
    center_y = big_image_h // 2 + int(random.uniform(-diff_h, diff_h))

    fusion_type = random.choice(fusion_type_list)
    fusion = cv2.seamlessClone(small_image, big_image, mask, (center_x, center_y), fusion_type)
    return fusion


if __name__ == '__main__':
    test_dir = "/data/giraffe/0_FSL/data/mini_imagenet_tools/processed_images/train"
    fileList = []
    for root, dirs, files in os.walk(test_dir, True):
        for name in files:
            if name.endswith(".jpg") or name.endswith(".png") or name.endswith(".bmp"):
                file_path = os.path.join(root, name)
                fileList.append(file_path)
    random.shuffle(fileList)

    source_dir = "/data/giraffe/0_FSL/data/mini_imagenet_tools/processed_images/test"
    source_fileList = []
    for root, dirs, files in os.walk(test_dir, True):
        for name in files:
            if name.endswith(".jpg") or name.endswith(".png") or name.endswith(".bmp"):
                file_path = os.path.join(root, name)
                source_fileList.append(file_path)
    random.shuffle(source_fileList)

    cv2.namedWindow("targe", 0)
    cv2.namedWindow("source_image", 0)
    cv2.namedWindow("fusion", 0)
    for target_image_path in source_fileList:
        target = cv2.imread(target_image_path)
        cv2.imshow("targe", target)
        cv2.waitKey(1)
        h, w, c = target.shape
        random.shuffle(source_fileList)
        for source_image_path in source_fileList:
            source = cv2.imread(source_image_path)
            cv2.imshow("source_image", source)
            ratio = random.uniform(0.9, 1.)
            resize_source = cv2.resize(source, None, None, ratio, ratio)

            fusion = possion_fusion(target, resize_source, fusion_type_list=[cv2.MONOCHROME_TRANSFER])
            fusion2 = possion_fusion(target, resize_source, fusion_type_list=[cv2.NORMAL_CLONE])
            cv2.imshow("fusion", cv2.hconcat([target, source, fusion, fusion2]))
            ch = cv2.waitKey(0)
            if int(ch) == 27:
                break
