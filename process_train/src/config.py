import argparse

# 创建顶级解析器
parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(dest='command')

# stage1 parser
parser_a = subparsers.add_parser('stage1_construct_prompt_pool',
                                 help='parser for constructing prompt pool')
parser_a.add_argument('--input_prompt_path', type=str, default="../../data/prompt/prompts.json")

parser_a.add_argument('--num_prompts_for_each_type', type=int, default=300)
parser_a.add_argument('--num_examples', type=int, default=3)
parser_a.add_argument('--parallel_size', type=int, default=10)

# stage2 parser
parser_b = subparsers.add_parser('stage2_construct_sft_data',
                                 help='parser for constructing sft data')
# step1 sample system for conversational data
parser_b.add_argument('--input_prompt_path', type=str, default="data/prompts.json")
parser_b.add_argument("--input_conv_path", type=str, default="data/raw_conv.jsonl")
parser_b.add_argument("--cached_train_path", type=str, default="data/train.jsonl",
                      help="Path containing conversation data and sampled prompts. " \
                           "This data is used for further processing to generate training datasets.")
parser_b.add_argument("--cached_test_path", type=str, default="data/test.jsonl")

# step2: construct sft data
parser_b.add_argument("--input_cached_path", type=str, default="data/train.jsonl")
parser_b.add_argument("--ckpt_path", type=str, default="data/train_ckpt.pkl",
                      help="Path to the checkpoint file. This file contains the saved state "
                           "of the data generation process, allowing for the resumption of the "
                           "process from the last saved checkpoint in case of interruption.")
parser_b.add_argument("--dump_path", type=str, default="../../data/conv/output/train.jsonl")
parser_b.add_argument("--skip_prob", type=float, default=0,
                      help="Specifies the probability (between 0.0 and 1.0) for randomly skipping "
                           "the generation process. ")
parser_b.add_argument("--num_gen", type=int, default=-1,
                      help="Specifies the number of samples to generate. ")

# stage2, step4: sample data for training discriminator
parser_c = subparsers.add_parser('stage2-4_discriminator_data', help='parser for constructing discriminator data')
parser_c.add_argument("--input_path", type=str, default="data/annotation/train.jsonl")
parser_c.add_argument("--output_sft_path", type=str, default="datat/train_sft.jsonl")
parser_c.add_argument("--output_discriminator_path", type=str,
                      default="../../data/conv/output/train_discriminator.jsonl")
parser_c.add_argument("--num_discriminator_data", type=int, default=2000)

# stage2, step5: train discriminator
parser_d = subparsers.add_parser('stage2-5_train_discriminator', help='parser for training discriminator')
parser_d.add_argument("--input_path", type=str, default="data/annotation/train.jsonl")
parser_d.add_argument("--output_path", type=str, default="data/output/train.jsonl")

# 解析命令行参数
args = parser.parse_args()
