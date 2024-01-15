CKPT_DIR="ckpt_dir/"
DATA_FILE_PATH="data/test_sft.jsonl"
RESULT_DIR="result"
RESULT_NAME="data-discriminator-base-chatglm6b-40k"
EVAL_NUM=2000

CUDA_VISIBLE_DEVICES="0" \
python evaluate.py \
--ckpt_file_path=${CKPT_DIR} \
--data_file_path=${DATA_FILE_PATH} \
--result_dir=${RESULT_DIR} \
--result_name=${RESULT_NAME} \
--eval_num=${EVAL_NUM}