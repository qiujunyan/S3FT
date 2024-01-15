import copy
import os
import os.path
import sys
from pathlib import Path

# 将src路径添加到环境变量中
sys.path.insert(0, str(Path(os.path.abspath(__file__)).parents[1]))

import random
import time
from datetime import timedelta

import utils
from config import args
from utils import *


def resume_from_ckpt(ckpt_path):
    print(f"loading from {ckpt_path}")
    if os.path.exists(ckpt_path):
        total_len, utils.NUM_CALL, utils.NUM_ERROR, init_conv_id, init_prompt_id, total_time = load_pickle(ckpt_path)
    else:
        total_len, init_conv_id, init_prompt_id, total_time = 0, 0, 0, 0
    print(
        f"init_len: {total_len}, init_conv_id: {init_conv_id}, init_prompt_id:{init_prompt_id}, init_time:{total_time}")
    return total_len, init_conv_id, init_prompt_id, total_time


def save_data(args, data, *argv):
    with open(args.dump_path, "a") as fp:
        for record in data:
            json_record = json.dumps(record, ensure_ascii=False)
            fp.write(json_record + "\n")
    dump_pickle(args.ckpt_path, argv)


def filter_error_data(data, pid):
    error_index = []
    for i, data_item in enumerate(data):
        if "【GPT4-ERROR】" in data_item["response"]:
            error_index.append(i)
        else:
            data_item["gpt4"] = data_item["response"]
            data_item["pid"] = pid
            del data_item["response"]
    for index in error_index[::-1]:
        del data[index]
    return data


def reformate_data(data):
    formed_data = []
    for i, data_item in enumerate(data):
        messages = data_item["message"]
        assert messages[0]["role"] == "system"
        history = []
        for index, uttr_dict in enumerate(messages[1:-1:2]):
            index = index * 2 + 1
            assert messages[index]["role"] == "user" and \
                   messages[index + 1]["role"] == "assistant"
            history.append({"prompt": messages[index]["content"],
                            "response": messages[index + 1]["content"]})

        assert messages[-1]["role"] == "user"
        item = {"pid": data_item["pid"],
                "system": messages[0]["content"],
                "history": history,
                "prompt": messages[-1]["content"],
                "response": data_item["gpt4"]}
        formed_data.append(item)
    return formed_data


def generate_data_with_gpt4(args):
    raw_data = load_json(args.input_cached_path)
    start_time = time.time()
    total_len, init_conv_id, init_prompt_id, total_time_init = resume_from_ckpt(args.ckpt_path)
    if total_len >= args.num_gen > 0:
        print("reaching the maximum number, early stopping...")
        return

    for conv_id, data_info in enumerate(raw_data[init_conv_id:]):
        conv_id += init_conv_id
        # resume prompt id
        if init_prompt_id != 0:
            start = init_prompt_id
            init_prompt_id = 0
        else:
            start = 0

        for prompt_id, sys_prompt in enumerate(data_info["system_list"][start:]):
            if random.random() < args.skip_prob:
                continue
            prompt_id += start
            # system = Instruction_zh.format(sys_prompt["prompt"])
            system = sys_prompt["prompt"]
            incremental_conversations = []
            context = [{"role": "system", "content": system}]
            for conv in data_info["conversation"]:
                context.append({"role": "user", "content": conv["prompt"]})
                if random.random() >= args.skip_prob:
                    incremental_conversations.append(copy.deepcopy(context))
                context.append({"role": "assistant", "content": conv["response"]})

            # generate and process data
            data = parallel_call_gpt4(incremental_conversations)
            pure_data = filter_error_data(data, sys_prompt["pid"])
            data = reformate_data(pure_data)
            total_len += len(data)

            end_time = time.time()
            total_time = end_time - start_time + total_time_init

            print(f"conv_id: {conv_id}, prompt_id: {prompt_id}, total_time: {timedelta(seconds=total_time)}, "
                  f"total_num: {total_len}, num_call: {utils.NUM_CALL}, num_error:{utils.NUM_ERROR}")

            save_data(args, data, total_len, utils.NUM_CALL, utils.NUM_ERROR,
                      conv_id, prompt_id, total_time)

            if 0 < args.num_gen < total_len:
                print("reaching the maximum number, early stopping...")
                return


if __name__ == '__main__':
    print_args(args)
    generate_data_with_gpt4(args)
