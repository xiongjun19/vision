# coding=utf8

import sys
import os
import pandas as pd
from collections import OrderedDict


def main(in_path, out_path, idx):
    # import ipdb; ipdb.set_trace()
    lines = _read_file(in_path, idx)
    ly_arr, b_arr = _parse_bytes(lines)
    _save_info(ly_arr, b_arr, out_path)


def _parse_bytes(lines):
    res = OrderedDict() 
    for line  in lines:
        if 'weight' in line or 'bias' in line:
            _arr = line.strip().split(",")
            key = _arr[0].strip().strip(".weight").strip(".bias")
            val = float(_arr[1].strip())
            prev_val = res.get(key, 0)
            res[key] = prev_val + val
    key_arr = []
    res_arr = []
    for key, val in res.items():
        key_arr.append(key)
        res_arr.append(val)
    return key_arr, res_arr


def _save_info(ly_arr, b_arr, out_path):
    res_dict = {
            'layer': ly_arr,
            'bytes': b_arr,
    }
    df = pd.DataFrame.from_dict(res_dict)
    df.to_csv(out_path, index=False)


def _read_file(in_path, idx):
    begin = False
    res = []
    with open(in_path) as _in:
        for line in _in:
            if begin:
                if "iteration number:" in line:
                    break
                else:
                    res.append(line.strip())
            else:
                if "iteration number:" in line:
                    num = int(line.strip().split(":")[1])
                    if num == idx:
                        begin = True
    return res


if __name__ == '__main__':
    in_path = sys.argv[1]
    out_path = sys.argv[2]
    idx = 2  # indicate to parse which iteration
    main(in_path, out_path, idx)
