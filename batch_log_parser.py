# coding=utf8

import os
import sys
import log_parser

def main(in_dir, out_dir):
    f_names = os.listdir(in_dir)
    for f_name in f_names:
        if f_name.endswith('.txt'):
            out_path = os.path.join(out_dir, f_name)
            in_path = os.path.join(in_dir, f_name)
            log_parser.main(in_path, out_path, 4)

        
if __name__ == '__main__':
    in_path = sys.argv[1]
    out_path = sys.argv[2]
    main(in_path, out_path)
