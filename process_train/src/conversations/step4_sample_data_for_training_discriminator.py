import os.path
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(os.path.abspath(__file__)).parents[1]))

from config import args
from utils import *


def split_data(args):
    all_data = load_jsonline(args.input_path)
    random.shuffle(all_data)

    data_discriminator = all_data[:args.num_discriminator_data]
    data_sft = all_data[args.num_discriminator_data:]

    dump_jsonline(args.output_discriminator_path, data_discriminator)
    print("Discriminator data saved to {}".format(os.path.abspath(args.output_discriminator_path)))
    dump_jsonline(args.output_sft_path, data_sft)
    print("SFT data saved to {}".format(os.path.abspath(args.output_sft_path)))


if __name__ == '__main__':
    split_data(args)
