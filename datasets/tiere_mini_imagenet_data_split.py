import os
import csv
from shutil import copytree
from tqdm import tqdm


# os.path.join(base_path, "{}.csv".format(phase))
def merge_sub_class(csv_path=None):
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        lines = [row for row in reader]

    high_level_class_dict = {}
    sub_level_class_dict = {}
    for sub_class, high_level_class in lines:
        if high_level_class not in high_level_class_dict.keys():
            high_level_class_dict[high_level_class] = []
        high_level_class_dict[high_level_class].append(sub_class)

        sub_level_class_dict[sub_class] = high_level_class
    print("merge done. high_level_class num: {}".format(len(high_level_class_dict)))

    return high_level_class_dict, sub_level_class_dict


def genereate_meta_datasets(source_dir, csv_path, phase, out_dir=None):
    data_sets = os.path.join(source_dir, phase)
    high_level_class_dict, sub_level_class_dict = merge_sub_class(csv_path)

    if out_dir is None:
        out_dir = source_dir + "_high_level"

    for sub_class in tqdm(os.listdir(data_sets)):
        sub_class_path = os.path.join(data_sets, sub_class)
        high_level_class = sub_level_class_dict.get(sub_class)
        new_path = os.path.join(out_dir, phase, high_level_class)
        os.makedirs(new_path, exist_ok=True)
        new_path = os.path.join(new_path, sub_class)
        copytree(sub_class_path, new_path, dirs_exist_ok=True)


genereate_meta_datasets("/data/giraffe/0_FSL/data/tiered_imagenet_tools/tiered_imagenet_224",
                        "/data/giraffe/0_FSL/data/tiered_imagenet_tools/tiered_imagenet_split/test.csv",
                        phase="test")
genereate_meta_datasets("/data/giraffe/0_FSL/data/tiered_imagenet_tools/tiered_imagenet_224",
                        "/data/giraffe/0_FSL/data/tiered_imagenet_tools/tiered_imagenet_split/train.csv",
                        phase="train")
genereate_meta_datasets("/data/giraffe/0_FSL/data/tiered_imagenet_tools/tiered_imagenet_224",
                        "/data/giraffe/0_FSL/data/tiered_imagenet_tools/tiered_imagenet_split/val.csv",
                        phase="val")
