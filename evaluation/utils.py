import argparse
import json
import os.path
import pickle

from transformers import AutoTokenizer, AutoModel


def print_args(args):
    # 设定总宽度和黑点字符
    total_width = 100
    dot_char = '•'

    # 计算最长的参数名的长度
    max_arg_length = max(len(arg) for arg in vars(args))

    # 打印格式化的参数
    for arg in vars(args):
        value = getattr(args, arg)
        # 格式化左侧文本
        left_text = f'{arg}'.ljust(max_arg_length)
        # 格式化右侧文本（左对齐）
        right_text = f'{value}'
        # 计算需要填充的黑点数量
        dots_count = total_width - len(left_text) - len(right_text)
        # 生成填充的黑点字符串
        dots = dot_char * max(dots_count, 1)  # 确保至少有一个黑点
        # 打印格式化的行
        print(f'{left_text} {dots} {right_text}')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--finetune_iters", type=int, default=0,
                        help="number of iterations to finetune the model, 0 means not finetune")
    parser.add_argument("--ckpt_file_path", type=str,
                        default="/workspace/qiujunyan/checkpoints/chatglm/iter_2k_cl_ordered")
    parser.add_argument("--data_file_path", type=str,
                        default="/workspace/qiujunyan/system_prompt/data/ContrastiveConv/output_data/ordered_test_data.jsonl")

    parser.add_argument("--result_dir", type=str,
                        default="/workspace/qiujunyan/system_prompt/evaluation/result_cl_ordered")
    parser.add_argument("--result_name", type=str, default="iter_2k_cl_ordered")
    parser.add_argument("--eval_num", type=int, default=-1)

    # parser.add_argument("--model_name", type=str, default="gpt2")

    args = parser.parse_args()

    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir, exist_ok=True)
    # args.ckpt_file_path = os.path.join(args.ckpt_file_dir,
    #                                    f"iter_{args.finetune_iters:07d}")
    # args.ckpt_file_path = "/workspace/qiujunyan/checkpoints/chatglm/iter_0000000_with_sft"

    print_args(args)
    return args


def print_conversation(conv_id, conversation):
    print("*" * 20 + f"conversation {conv_id + 1}" + "*" * 20)
    print(f"【system_prompt:】 {conversation['system']}")
    for turn_id, conv_dict in enumerate(conversation["history"]):
        print(f"【user {turn_id + 1}:】{conv_dict['prompt']}")
        print(f"【assistant {turn_id + 1}:】{conv_dict['response']}")
    print(f"【user {len(conversation['history']) + 1}:】{conversation['prompt']}")
    print(f"【assistant {len(conversation['history']) + 1}:】{conversation['response']}")
    print(f"【target:】{conversation['target']}")
    print("\n")


def load_chatglm(args):
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt_file_path,
                                              trust_remote_code=True,
                                              local_files_only=True,
                                              cache_dir=args.ckpt_file_path)
    model = AutoModel.from_pretrained(args.ckpt_file_path,
                                      trust_remote_code=True,
                                      local_files_only=True,
                                      cache_dir=args.ckpt_file_path).half().cuda()
    return model, tokenizer


def call_chatglm(model, tokenizer, data):
    history = [{"role": "system",
                "content": data["system"]}]
    for conv in data["history"]:
        history.extend([{"role": "user",
                         "content": conv["prompt"]},
                        {"role": "assistant",
                         "content": conv["response"]}])
    try:
        response, _ = model.chat(tokenizer, data["prompt"], history=history)
    except Exception as e:
        response = "ChatGLM-ERROR"
    data["target"] = response
    return data


def load_data(args):
    datas = []
    with open(args.data_file_path, "r") as fp:
        for line in fp.readlines():
            if 0 < args.eval_num <= len(datas):
                break
            if len(line) > 8092:
                continue
            data_line = json.loads(line)
            datas.append(data_line)
    return datas


def load_jsonline(file_path):
    json_lines = []
    with open(file_path) as fp:
        for i, line in enumerate(fp.readlines()):
            json_lines.append(json.loads(line))
    return json_lines


def dump_jsonline(file_path, data):
    with open(file_path, "w", encoding="utf-8") as fp:
        for line in data:
            fp.write(json.dumps(line, ensure_ascii=False) + "\n")


def load_json(file_path):
    with open(file_path) as fp:
        json_data = json.load(fp)
    return json_data


def dump_json(file_path, data):
    with open(file_path, "w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False)


def load_pickle(file_path):
    with open(file_path, "rb") as fp:
        data = pickle.load(fp)
    return data


def dump_pickle(file_path, data):
    with open(file_path, "wb") as fp:
        pickle.dump(data, fp)
