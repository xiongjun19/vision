# coding=utf8

import argparse
from collections import OrderedDict
import pandas as pd


def main(args):
    flops_path = args.input
    flops_dict  = _read_flops(flops_path)
    print(flops_dict)
    ly_df = _read_layer_time(args.time_path)
    fin_df = _merge_and_calc(ly_df, flops_dict, args.max_flops)
    fin_df.to_csv(args.output, index=False)


def _read_flops(f_path):
    res = OrderedDict()
    is_data = False
    with open(f_path) as _in:
        for line in _in:
            if not is_data:
                if 'module' in line and 'Flops' in line:
                    is_data = True
            else:
                if line.startswith('total'):
                    is_data=False
                else:
                    line_arr = line.strip().split("  ")
                    filter_arr = [x for x in line_arr if len(x) > 0]
                    name = filter_arr[1]
                    val = filter_arr[-5].replace(",", "")
                    val = float(val)
                    res[name] = val
    return res


def _read_layer_time(f_path):
    df = pd.read_csv(f_path)
    df['bwd'] = df['ig'] + df['wg']
    return df

def _merge_and_calc(ly_df, flops_dict, max_flops):
    flops_df = _dict_2_df(flops_dict, max_flops)
    merged_df = ly_df.join(flops_df)
    merged_df['fwd_flops_ratio'] = merged_df['flops'] * 1000 / ( 
            merged_df['max_flops'] * merged_df['forward'])
    merged_df['bwd_flops_ratio'] =  2 * merged_df['flops'] * 1000 / ( 
            merged_df['max_flops'] * merged_df['bwd'])
    return merged_df


def _dict_2_df(flops_dict, max_flops):
    res_dict = {
            'layer_name': [],
            'flops': [],
    }
    key_arr = ['conv', 'downsample.0', 'fc']
    for k, v in flops_dict.items():
        if any([x in k for x in key_arr ]):
            res_dict['layer_name'].append(k)
            res_dict['flops'].append(v * 64)
    res_dict['max_flops'] = [max_flops * (1024 ** 4)] * len(res_dict['flops'])
    df = pd.DataFrame.from_dict(res_dict)
    return df


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default=None)
    parser.add_argument('-it', '--time_path', type=str, default=None)
    parser.add_argument('-c', '--max_flops', type=float, default=31.2,
        help='the maximun flops of the card, (Unit is TF)')
    parser.add_argument('-o', '--output', type=str, default=None)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    t_args = get_args()
    main(t_args)
