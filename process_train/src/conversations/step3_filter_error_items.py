import argparse

import os.path
import sys
from pathlib import Path

# 将src路径添加到环境变量中
sys.path.insert(0, str(Path(os.path.abspath(__file__)).parents[1]))
from utils import *

ERROR_KEYWORDS = ["GPT4-ERROR"]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data_path", type=str,
                        default="data/train.jsonl")
    args = parser.parse_args()
    return args


def run_filter(input_path, output_path):
    def _check_error_keywords(string):
        for keyword in ERROR_KEYWORDS:
            if keyword in string:
                return True
        return False

    raw_data = load_jsonline(input_path)
    error_list = []
    for i, data_item in enumerate(raw_data):
        if _check_error_keywords(data_item["response"]):
            error_list.append(i)
    print(f"number of error items: {len(error_list)}")

    for error_idx in error_list[::-1]:
        del raw_data[error_idx]

    dump_jsonline(output_path, raw_data)


if __name__ == '__main__':
    args = get_args()
    run_filter(args.raw_data_path, args.raw_data_path)
