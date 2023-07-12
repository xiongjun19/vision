# coding=utf8

import pandas as pd


def main(comm_time_file, trans_byte_file, output_file, band_width):
    comm_df = _read_comm_time(comm_time_file)
    name_arr, byte_arr = _read_bytes(trans_byte_file)
    info = _merge_and_calc(comm_df, name_arr, byte_arr, band_width)
    df = pd.DataFrame.from_dict(info)
    df.to_csv(output_file, index=False)


def _read_comm_time(f_path):
    df = pd.read_csv(f_path)
    return df


def _read_bytes(f_path):
    name_arr = []
    byte_arr = []
    with open(f_path) as _in:
        for line in _in:
            line_arr = line.strip().split(",")
            if len(line_arr) > 1:
                name = line_arr[0].strip()
                byte_size = int(line_arr[1].strip())
                name_arr.append(name)
                byte_arr.append(byte_size)
    return name_arr, byte_arr


def _merge_and_calc(comm_df, name_arr, byte_arr, band_width):
    time_arr = list(comm_df['time'])
    time_arr = time_arr[:len(name_arr)]
    band_arr = []
    band_ratio_arr = []
    for time, byte in zip(time_arr, byte_arr):
        band = (byte * 1e+9) / (time * (1024 **3))
        ratio = band / band_width
        band_arr.append(band)
        band_ratio_arr.append(ratio)
    res = {
        'weight_name': name_arr,
        'byte': byte_arr,
        'fact_band': band_arr,
        'band_ratio': band_ratio_arr,
    }
    return res


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, help='input time of kernels of nccl time')
    parser.add_argument('-ib', '--byte_file', type=str, help='input bytes of kernels of nccl ')
    parser.add_argument('-o', '--output', type=str, help='path to output file')
    parser.add_argument('-b', '--bandwidth', type=float, help='bandwith of the test')
    args = parser.parse_args()
    main(args.input, args.byte_file, args.output, args.bandwidth)
