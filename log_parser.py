# coding=utf8

import sys
import os
import pandas as pd


def main(in_path, out_path, idx):
    # import ipdb; ipdb.set_trace()
    lines = _read_file(in_path, idx)
    fwd_arr = _parse_fwd(lines)
    ig_arr, wg_arr, wg_comm = _parse_bwd(lines)
    _save_info(fwd_arr, ig_arr, wg_arr, wg_comm, out_path)


def _parse_fwd(lines):
    res = dict()
    for line  in lines:
        if 'fwd_' in line:
            info = line.strip().split("|")[1]
            _arr = info.split(":")
            key = int(_arr[0].split("_")[1])
            val = float(_arr[1].strip())
            res[key] = val
    res_arr = [None] * len(res)
    for key, val in res.items():
        res_arr[key] = val
    return res_arr


def _parse_bwd(lines):
    ig_arr = _parse_ig(lines)
    wg_arr = _parse_wg(lines)
    final_bw_time = get_final_bw(lines)
    comm_time = final_bw_time - wg_arr[0]
    _post_process(ig_arr, wg_arr)
    return ig_arr, wg_arr, comm_time


def _parse_ig(lines):
    sep_key = ' _sep_ '
    prefix_key = 'ig_bwd_'
    return _parse_impl(lines, sep_key, prefix_key)


def _parse_wg(lines):
    sep_key = ' _sep_ '
    prefix_key = 'wg_bwd_'
    return _parse_impl(lines, sep_key, prefix_key)


def _parse_impl(lines, sep_key, prefix_key):
    res_dict = dict()
    for line in lines:
        if sep_key in line and  prefix_key in line:
            _arr = line.split(sep_key)
            key_str = _arr[0]
            key = int(key_str.strip().strip(prefix_key))
            val_str = _arr[1].split(":")[-1].strip()
            val = float(val_str)
            res_dict[key] = val
    res = [None] * len(res_dict)
    for key, val in res_dict.items():
        res[key] = val
    return res


def get_final_bw(lines):
    sep_key = ' _sep_ '
    for line in lines:
        if 'backward' in line and sep_key not in line:
            val = line.split(":")[-1].strip()
            return float(val)
    return None


def _post_process(ig_arr, wg_arr):
    for i in range(len(ig_arr)):
        wg_arr[i] = wg_arr[i] - ig_arr[i]
    for i in range(len(ig_arr) - 1):
        ig_arr[i] = ig_arr[i] - ig_arr[i+1]


def _save_info(fwd_arr, ig_arr, wg_arr, wg_comm, out_path):
    layer_arr = list(range(len(fwd_arr)))
    comm_arr = [None] * len(fwd_arr)
    comm_arr[0] = wg_comm
    res_dict = {
            'layer': layer_arr,
            'forward': fwd_arr,
            'ig': ig_arr,
            'wg': wg_arr,
            'weight_comm': comm_arr,
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
    idx = 4  # indicate to parse which iteration
    main(in_path, out_path, idx)
