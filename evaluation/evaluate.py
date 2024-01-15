from tqdm import tqdm

from utils import *


def try_cache(args):
    dump_file = os.path.join(args.result_dir, f"{args.result_name}.json")
    if os.path.exists(dump_file):
        with open(dump_file) as fp:
            processed_datas = json.load(fp)
    else:
        processed_datas = []
        datas = load_data(args)
        model, tokenizer = load_chatglm(args)
        for conv_id, data in tqdm(enumerate(datas), total=len(datas)):
            item = call_chatglm(model, tokenizer, data)
            processed_datas.append(item)
            # print_conversation(conv_id, data)
        with open(dump_file, 'w', encoding="utf-8") as fp:
            json.dump(processed_datas, fp, ensure_ascii=False)
    return processed_datas


def main():
    args = get_args()
    try_cache(args)


if __name__ == '__main__':
    main()
