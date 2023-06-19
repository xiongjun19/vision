# coding=utf8

import sys
import os
import shutil
from pathlib import Path


def copy(src, dst):
    shutil.copy(src, dst)


def main(in_dir, ann_path, out_dir):
    ann_dict = _read_ann_path(ann_path)
    img_list = list_imgs(in_dir)
    for img in img_list:
        img_key = os.path.basename(img.name)
        img_class = ann_dict.get(img_key, None)
        if img_class:
            sub_dir = os.path.join(out_dir, img_class)
            os.makedirs(sub_dir, exist_ok=True)
            dst_file = os.path.join(sub_dir, img_key) 
            copy(img.absolute(), dst_file)



def _read_ann_path(f_path):
    res = dict()
    with open(f_path) as in_:
        for line in in_:
            line = line.strip()
            line_arr = line.split()
            if len(line_arr) >= 2:
                key = os.path.basename(line_arr[0])
                val = line_arr[1]
                res[key] = val
    return res


def iter_dir(in_dir, pat='*'):
    res = list()
    path = Path(in_dir)
    for p in path.rglob(pat):
        res.append(p)
        # print(p.name)
    return res


def list_imgs(in_dir):
    img_list = iter_dir(in_dir, pat="*.JPEG")
    # for img in img_list:
    #     img_name = img.name
    #     # print(img_name)
    #     # print(img.absolute())
    return img_list


if __name__ == '__main__':
   in_dir = sys.argv[1] 
   out_dir = sys.argv[2]
   ann_f_path = sys.argv[3]
   main(in_dir, ann_f_path, out_dir)


