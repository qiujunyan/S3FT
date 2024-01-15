import random

import utils
from config import args
from utils import *

instruction = """
你是一个专业的提示工程师，能够根据给出的例子，生成相同类型的prompt。

你生成的prompt不应该为AI指定特定的任务，而是给模型一个基本的设定。

请用中文编写prompt。
"""


def generate_prompt(args, prompt_pool, prompt_type):
    def _construct_parallel_inputs():
        parallel_inputs = []
        while len(parallel_inputs) < args.parallel_size:
            prompt = "请根据以下例子生成一条prompt:\n"
            sampled_prompts = random.sample(prompts, num_example)
            for i, sample in enumerate(sampled_prompts):
                prompt += f"{sample}\n\n"
            parallel_inputs.append([{"role": "system", "content": instruction},
                                    {"role": "user", "content": prompt}])
        return parallel_inputs

    prompts = prompt_pool[prompt_type]
    num_example = min(len(prompts), args.num_examples)
    while len(prompts) < args.num_prompts_for_each_type:
        parallel_inputs = _construct_parallel_inputs()
        responses = parallel_call_gpt4(parallel_inputs)
        responses = [item["response"] for item in responses]
        responses = list(filter(lambda x: "【GPT4-ERROR】" not in x, responses))
        for i, response in enumerate(responses):
            print(f"num {len(prompts) + i + 1}: {response}")
        prompts.extend(responses)

        if utils.NUM_CALL >= 100:
            dump_json(args.input_prompt_path, prompt_pool)
        if utils.NUM_ERROR >= 100:
            raise EnvironmentError("GPT4 error, system exit...")


def main():
    print_args(args)
    prompt_pool = load_json(args.input_prompt_path)
    for prompt_type, prompts in prompt_pool.items():
        print(f"generating type: {prompt_type}, current length: {len(prompts)}")
        generate_prompt(args, prompt_pool, prompt_type)


if __name__ == '__main__':
    main()
