export PYTHONPATH=$PYTHONPATH:src

input_prompt_path="data/prompts.json"
num_prompts_for_each_type=300
num_examples=3
parallel_size=10

echo "stage1:constructing prompt pool"
python src/prompt/construct_prompt_pool.py stage1_construct_prompt_pool \
--input_prompt_path=${input_prompt_path} \
--num_prompts_for_each_type=${num_prompts_for_each_type} \
--num_examples=${num_examples} \
--parallel_size=${parallel_size}
