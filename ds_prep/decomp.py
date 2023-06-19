# coding=utf8


import os
import sys
import subprocess
import shlex
from subprocess import Popen


def exe_cmd_sync(cmd_line):
    print(cmd_line)
    # call is blocking:
    cmd_args = shlex.split(cmd_line)
    subprocess.call(cmd_args)


def main(in_dir, out_dir):
    f_arr = os.listdir(in_dir)
    for f in f_arr:
        if f.endswith('.tar'):
            f_name = f.split(".")[0]
            sub_out_dir = os.path.join(out_dir, f_name)
            os.makedirs(sub_out_dir, exist_ok=True)
            f_path = os.path.join(in_dir, f)
            cmd = f'tar xvf {f_path} --directory {sub_out_dir}'
            exe_cmd_sync(cmd)


if __name__ == '__main__':
    in_dir = sys.argv[1]
    out_dir = sys.argv[2]
    main(in_dir, out_dir)
