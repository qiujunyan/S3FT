echo "step5: construct data for training discriminator"
echo "construct train data"
python src/discriminator/data_process.py stage2-5_train_discriminator \
--input_path="data/train.jsonl" \
--output_path="data/annotation/train.jsonl"
echo -e "training data construction finished\n\n"

echo "construct test data"
python src/discriminator/data_process.py stage2-5_train_discriminator \
--input_path="data/test.jsonl" \
--output_path="data/annotation/test.jsonl"
echo -e "test data construction finished\n\n"