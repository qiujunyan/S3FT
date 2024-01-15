import os.path
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(os.path.abspath(__file__)).parents[1]))

from config import args
from utils import *

output_dict = {"good": "【这是一个优秀的回复】",
               "bad_context": "【该回复不符合上下文】",
               "bad_system": "【该回复未遵循系统指令】",
               "all_bad": "【以上条件均不满足】",
               "out_of_place": "【不适合】"}

instruction = """
为了评估一条系统回复是否符合预期标准，请根据以下两个关键维度进行判断：
1. 回复的连贯性：检查该<系统回复>是否与之前的<对话上下文>及<用户提问>形成流畅且自然的衔接。
2. 对<背景及设定>的遵循：分析该回复是否系统给出的<背景及设定>。

若你认为回复不适合遵守系统指令，则输出"{}"，否则进行以下判断。
若你认为回复在这两个方面都做得很好，请输出【这是一个优秀的回复】。如果回复未能满足这些标准，则根据具体情况进行如下细分：
* 如果回复与上下文不连贯，请输出【回复不符合上下文】。
* 如果回复未遵循系统指定的背景及设定，请输出'【该回复未遵循系统指令】'。
* 如果回复同时违背了这两个条件，请输出【以上条件均不满足】。

请注意，你的输出只能是'{}'，'{}'，'{}'以及'{}'中的其中一个。
在任何情况下都不要违反这一规则。
""".format(output_dict["out_of_place"], output_dict["good"], output_dict["bad_context"],
           output_dict["bad_system"], output_dict["all_bad"])

input_template = """
<背景及设定>：\n{}

<对话上下文>：\n{}

<对话上下文>结束

<用户提问>：\n{}

<系统回复>：\n{}
"""


def reformat_data_to_fit_ChatGLM(args):
    def _convert_history_into_string(history):
        content = ""
        for turn_dict in history:
            content += "<用户提问>：\n{}\n<系统回复>：\n{}，\n".format(turn_dict["prompt"], turn_dict["response"])
        return content if len(content) > 0 else "无对话历史"

    raw_data = load_jsonline(args.input_path)
    all_datas = []
    label_dict = defaultdict(int)
    for i, data_item in enumerate(raw_data):
        pid = data_item["payload"]["pid"]
        prompt, system = data_item["prompt"].split("\"system\":")
        history = _convert_history_into_string(data_item["history"])
        response = data_item["responses"][0]["response"]
        label = f'【{data_item["labels"][0]["result"]["label"]}】'
        assert label in output_dict.values()

        input_content = input_template.format(system, history, prompt, response)
        all_datas.append({"pid": pid,
                          "system": instruction,
                          "history": [],
                          "prompt": input_content,
                          "response": label})
        label_dict[label] += 1

    for key, val in label_dict.items():
        print(f"{key}: {val}")
    dump_jsonline(args.output_path, all_datas)


def main():
    print_args(args)
    reformat_data_to_fit_ChatGLM(args)


if __name__ == '__main__':
    main()
