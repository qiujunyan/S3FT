# Training Large Language Models to Follow System Prompt with Self-Supervised Fine-tuning

This repository contains the code and resources for the paper "Training Large Language Models to Follow System Prompt with Self-Supervised Fine-tuning"

### Abstract
In the realm of artificial intelligence, system prompts stand as directives or requests aimed at guiding systems, such as programming environments or AI models, to execute specific tasks or operations. Typically positioned at the commencement of input sequences in large language models, these prompts play a pivotal role in shaping the model's response and guiding its interaction flow. However, a notable challenge emerges during multi-turn dialogues, where these models gradually diverge from adhering to the initial system prompt, leading to inconsistencies in the dialogue. In this paper, we present a scalable framework facilitating the adherence of language models to system prompts through automated data construction. Our approach, termed \textsc{Self-Supervised System Prompt Fine-tuning} (S3FT), begins by prompting a language model to modify real dialogue responses to fit a specific system prompt, using stylized translation. Subsequently, we select a small sample of these responses for human preference annotation. This annotated data is utilized to train the language model to act as a discriminator, identifying high-quality examples that are then employed in further supervised fine-tuning. Experimental results on several datasets demonstrate that applying our method to LlaMA2 and ChatGLM promotes human preference rates by over 50\%, and outperforms ChatGPT and GPT4 by a consideratble margin.

### Code Structure
The repository is organized as follows:
* `data` This directory contains data that is used for training and evaluation. Download from [Google Driver](https://drive.google.com/file/d/165_AUHe-A1KekJcILqWNn9cyJxE78LKS/view?usp=drive_link), and unzip to the `data` directory.
  * `train_discriminator.jsonl`: Training data for the discriminator.
  * `test_discriminator.jsonl`: Test data for the discriminator.'
  * `train_sft.jsonl`: Training data for SSFT.
* `doc`: Containing description of annotation rule for labelling discriminator data.
* `process_train` This directory contains source code related to data processing.
* `evaluation` This directory contains source code related to evaluation.

### Construct Training Data
* construct system prompt pool
```shell
bash process_train/scripts/run_construct_prompt_pool.sh
```
* construct data for discriminator
```shell
bash process_train/scripts/construct_data_for_training_discriminator.sh
```

### Run Evaluation
* Fine-tune the models using data generated in previous steps (kindly refer to the provided fine-tuning code in [ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B)). Save the checkpoints in ckpt_dir.
* Run the evaluation script
 ```shell
bash evaluation/run_eval.sh
```

