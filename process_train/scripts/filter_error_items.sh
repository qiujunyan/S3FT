echo "step3: filter error items"
python src/conversations/step3_filter_error_items.py \
--raw_data_path="data/conv/output/train.jsonl"

python src/conversations/step3_filter_error_items.py \
--raw_data_path="data/conv/output/test.jsonl"
echo  -e "step3 finished \n\n"