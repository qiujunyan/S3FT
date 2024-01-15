import os
import random

from config import args
from utils import *


class SystemPrompts(object):
    def __init__(self, args):
        super(SystemPrompts, self).__init__()
        self.args = args
        self.prompts = load_json(self.args.input_prompt_path)
        self.train_prompts, self.test_prompts = self.split_train_test_prompts()

    def split_train_test_prompts(self):
        train_prompts, test_prompts = {}, {}
        for type_id, (prompt_type, prompt_list) in enumerate(self.prompts.items()):
            prompt_list = [{"pid": f"{type_id + 1}-{i + 1}",
                            "prompt": prompt} for i, prompt in enumerate(prompt_list)]

            random.shuffle(prompt_list)
            num = len(prompt_list)
            train_prompts[prompt_type] = prompt_list[:int(num * 0.9)]
            test_prompts[prompt_type] = prompt_list[int(num * 0.9):]
        return train_prompts, test_prompts

    @staticmethod
    def sample_prompts(prompt_pool):
        prompts = []
        for prompt_list in prompt_pool.values():
            prompts.append(random.sample(prompt_list, 1)[0])
        return prompts

    def __getitem__(self, key):
        return self.train_prompts if key == "train" else self.test_prompts


class ConvLoader(object):
    def __init__(self, args):
        super(ConvLoader, self).__init__()
        self.train_convs, self.test_convs = self.load_and_split_conversations(args.input_conv_path)

    def __getitem__(self, key):
        return self.train_convs if key == "train" else self.test_convs

    @staticmethod
    def load_and_split_conversations(input_conv_path):
        conversations = load_jsonline(input_conv_path)
        random.shuffle(conversations)
        return conversations[:2000], conversations[2000:]

    def sample_prompts_for_data_split(self, prompt_obj, mode, path):
        prompt_pool = prompt_obj[mode]
        conversations = self.__getitem__(mode)
        for conv_id, conv_dict in enumerate(conversations):
            conv_id += 1
            conv_dict["conv_id"] = conv_id
            conv_dict["system_list"] = prompt_obj.sample_prompts(prompt_pool)
        dump_json(path, conversations)


def main():
    if os.path.exists(args.cached_train_path):
        return

    prompt_obj = SystemPrompts(args)
    conv_obj = ConvLoader(args)
    conv_obj.sample_prompts_for_data_split(prompt_obj, "train", args.cached_train_path)
    conv_obj.sample_prompts_for_data_split(prompt_obj, "test", args.cached_test_path)


if __name__ == '__main__':
    main()
