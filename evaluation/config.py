import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--finetune_iters", type=int, default=0,
                        help="number of iterations to finetune the model, 0 means not finetune")
    parser.add_argument("--ckpt_file_dir", type=str,
                        default="/workspace/qiujunyan/ChatGLM/ckpt/")
    parser.add_argument("--data_file_path", type=str,
                        default="/workspace/qiujunyan/system_prompt/data/RawConv/test_data_v2.jsonl")

    args = parser.parse_args()
    return args
