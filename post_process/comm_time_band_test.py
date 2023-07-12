# coding=utf8

import sqlite3
import pandas as pd


def do_parse(f_path, dev, out_path):
    mydb = sqlite3.connect(f_path)
    cursor = mydb.cursor()
    nccl_time_info = _parse_kernel(cursor, dev)
    _save_info(nccl_time_info, out_path)
    cursor.close()
    mydb.close()


def _save_info(nccl_time_info, out_path):
    _dict = {
        'start_time': [],
        'time': [],
        'op_name': [],
    }
    for item in nccl_time_info:
        s, t, o = item
        _dict['start_time'].append(s)
        _dict['time'].append(t)
        _dict['op_name'].append(o)
    df = pd.DataFrame.from_dict(_dict)
    df.to_csv(out_path, index=False)


def _parse_kernel(cursor, device=0):
    sql = f"select start, end - start as dur, s.value from CUPTI_ACTIVITY_KIND_KERNEL as k join StringIds as s ON k.ShortName=s.id "\
          f" where k.deviceId={device} AND Shortname in (select id from StringIds where  value like '%nccl%');"
    nccl_time = exec_and_parse(cursor, sql)
    return nccl_time


def exec_query(cursor, sql):
    cursor.execute(sql)
    return cursor.fetchall()


def exec_and_parse(cursor, sql):
    try:
        cursor.execute(sql)
        items = cursor.fetchall()
        return items
    except sqlite3.OperationalError as e:
        print(e)
        return None 


def test(f_path, device=0, out_path=None):
    res = do_parse(f_path, device, out_path)
    print(res)


if __name__ == '__main__':
    import sys
    device_num = 0
    t_path = sys.argv[1]
    o_path = sys.argv[2]
    if len(sys.argv) > 3:
        device_num = sys.argv[3]
    test(t_path, device_num, o_path)
