# coding: utf-8

import os

input_path = "data/school/school.txt"
backup_path = "data/school/school_original.txt"
tmp_path = "data/school/school_tmp.txt"


def convert_time(t):
    t = float(t)
    if 0 <= t < 0.2:
        return "1"
    elif 0.2 <= t < 0.4:
        return "2"
    elif 0.4 <= t < 0.6:
        return "3"
    elif 0.6 <= t < 0.8:
        return "4"
    elif t >= 0.8:
        return "5"
    else:
        return "0"


if not os.path.exists(backup_path):
    os.rename(input_path, backup_path)

with open(backup_path, "r") as fin, open(tmp_path, "w") as fout:
    for line in fin:
        parts = line.strip().split()
        if len(parts) != 3:
            continue
        source, target, time = parts
        new_time = convert_time(time)
        fout.write(f"{source} {target} {new_time}\n")

os.replace(tmp_path, input_path)
