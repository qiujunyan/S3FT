import argparse
import os

from utils import load_json, dump_json


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="result_cl")
    parser.add_argument("--model1", type=str, default="v2")
    parser.add_argument("--model2", type=str, default="cl")
    parser.add_argument("--model1_path", type=str, default="data-v2-base-chatglm6b-1k.json")
    parser.add_argument("--model2_path", type=str, default="data-cl-base-chatglm6b-1k.json")
    args = parser.parse_args()
    return args


def load_data():
    args = get_args()
    # data_v2 = load_json(os.path.join(args.data_dir, args.v2_path))
    data1 = load_json(os.path.join(args.data_dir, args.model2_path))
    data2 = load_json(os.path.join(args.data_dir, args.model1_path))

    return data1, data2


def gather(args):
    data1, data2 = load_data()
    merged_data = []
    for item_id, (item_ft, item_raw) in enumerate(zip(data1, data2)):
        system = item_raw["system"].split("\n\n")[-1]
        merged_data.append({"prompt_id": item_raw["prompt_id"] if "prompt_id" in item_raw.keys() else "",
                            "system": system,
                            "query": item_raw["prompt"],
                            "gpt4": item_raw["response"],
                            "response1": item_raw["target"],
                            "response2": item_ft["target"],
                            "model1": args.model1,
                            "model2": args.model2})
    return merged_data


def try_merge():
    args = get_args()
    file_path = os.path.join(args.data_dir, "merged.json")
    merged = gather(args)
    dump_json(file_path, merged)
    return merged


if __name__ == '__main__':
    try_merge()
