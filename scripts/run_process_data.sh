# cd your_path/system_prompt
ln -s prompt/data/prompts.json data

# sample prompts for train and test conversational data
echo -e "sample prompts for train data\n"
python process_train/sample_prompts_for_conv_data.py sample_prompts_for_conv_data \
--input_prompt_path=data/prompts.json \
--cached_path=data/cached_train.jsonl \
--split=train

echo -e "sample prompts for test data\n"
python process_train/sample_prompts_for_conv_data.py sample_prompts_for_conv_data \
--input_prompt_path=data/prompts.json \
--cached_path=data/cached_test.jsonl \
--split=test

# construct sft data
echo -e "construct sft data"
python process_train/construct_sft_data.py construct_sft_data \
--input_cached_path="data/cached_train.jsonl" \
--ckpt_path="data/train.ckpt" \
--dump_path="data/train.jsonl" \
--num_gen=-1 \
--skip_prob=0

echo -e "construct test data"
python process_train/construct_sft_data.py construct_sft_data \
--input_cached_path="data/cached_test.jsonl" \
--ckpt_path="data/test.ckpt" \
--dump_path="data/test.jsonl" \
--num_gen=200 \
--skip_prob=0.7


echo "construct discriminator data"
python process_train/sample_discriminator_data.py construct_discriminator_data \
--input_path="data/train.jsonl" \
--output_sft_path="data/train_sft.jsonl" \
--output_discriminator_path="data/train_discriminator.jsonl" \
--num_discriminator_data=2000