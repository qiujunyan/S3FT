export PYTHONPATH=$PYTHONPATH:src

cmd="stage2_construct_sft_data"

step1(){
  echo "stage2: constructing sft data..."
  echo "step1: sample system prompts for each conversation, and split train and test data"
  python src/conversations/step1_sample_system_for_conv_data.py ${cmd} \
  --input_prompt_path="data/prompt/prompts.json" \
  --input_conv_path="data/conv/raw_conv/raw_conv_3k.jsonl" \
  --cached_train_path="data/conv/cached/train.jsonl" \
  --cached_test_path="data/conv/cached/test.jsonl"
  echo -e "step1 finished\n\n"
}

step2_1(){
  echo "step2-1: generate data for test"
  python src/conversations/step2_construct_sft_data.py ${cmd} \
  --input_cached_path="data/conv/cached/test.jsonl" \
  --ckpt_path="data/conv/output/test_ckpt.pkl" \
  --dump_path="data/conv/output/test.jsonl" \
  --skip_prob=0.7 \
  --num_gen=200
  echo -e "step2-1 finished\n\n"
}

step2_2() {
  echo "step2-2: generate data for train"
  python src/conversations/step2_construct_sft_data.py ${cmd} \
  --input_cached_path="data/conv/cached/train.jsonl" \
  --ckpt_path="data/conv/output/train_ckpt.pkl" \
  --dump_path="data/conv/output/train.jsonl" \
  --skip_prob=0 \
  --num_gen=-1
  echo  -e "step2-2 finished\n\n"
}

step_3(){
  bash scripts/step3_filter_error_items.sh
}

tasks=$1
IFS=',' read -ra ADDR <<< "$tasks"
for i in "${ADDR[@]}"; do
    case $i in
        1)
            step1
            ;;
        2-1)
            step2_1
            ;;
        # 可以添加更多任务的调用
        2-2)
            step2_2
            ;;
        3)
            step3
            ;;
    esac
done