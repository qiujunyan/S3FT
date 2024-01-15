input_path="data/conv/output/train.jsonl"
output_sft_path="data/conv/output/train_sft.jsonl"
output_discriminator_path="data/conv/output/train_discriminator.jsonl"
num_discriminator_data=2000

echo "step4: sample_data_for_training_discriminator"
python src/conversations/step4_sample_data_for_training_discriminator.py stage2-4_discriminator_data \
--input_path=${input_path} \
--output_sft_path=${output_sft_path} \
--output_discriminator_path=${output_discriminator_path} \
--num_discriminator_data=${num_discriminator_data}