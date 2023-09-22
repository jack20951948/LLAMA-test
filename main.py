import torch
import argparse
import bitsandbytes as bnb
from datasets import load_dataset
from functools import partial
import os
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, Trainer, TrainingArguments, BitsAndBytesConfig, \
    DataCollatorForLanguageModeling, TrainingArguments, EarlyStoppingCallback, IntervalStrategy
from datasets import load_dataset, concatenate_datasets
from trl import SFTTrainer
import random

print("Is Cuda available: ", torch.cuda.is_available())

def load_model(model_name, bnb_config):
    n_gpus = torch.cuda.device_count()
    max_memory = f'{12288}MB'

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto", # dispatch efficiently the model on the available ressources
        max_memory = {i: max_memory for i in range(n_gpus)},
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)

    # Needed for LLaMA tokenizer
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

# Load the dataset from Hugging Face
dataset_abilities = load_dataset("changjacHp/lol_champions_abilities", split="train")

print(f'Number of prompts: {len(dataset_abilities)}')
print(f'Column names are: {dataset_abilities.column_names}')

dataset_tips = load_dataset("changjacHp/lol_champion_top3_tips", split="train")

print(f'Number of prompts: {len(dataset_tips)}')
print(f'Column names are: {dataset_tips.column_names}')
# print(dataset)

def create_prompt_formats_tips(sample):
    # Randomly select one of three different formats for the prompt
    format_choice = random.choice([1, 2, 3])

    # Format instruction based on the given scenario
    instruction = f"I'm using {sample['champion']} and the champion I need to fight with is {sample['target_hero']}. Can you give me some advice or suggestions? Considered both champion's abilities and let me know my opponent is a Threat or Synergy to me and also provide some further detail if you got some."

    # Determine the appropriate phrasing based on 'category'
    if sample['category'] == 'Threat':
        if format_choice == 1:
            interaction_phrase = f"{sample['target_hero']} is a {sample['level']} threat to {sample['champion']}."
        elif format_choice == 2:
            interaction_phrase = f"{sample['target_hero']} is a dangerous opponent for {sample['champion']}."
        else:
            interaction_phrase = f"{sample['target_hero']} poses a significant challenge to {sample['champion']}."
    else:  # Assuming 'Synergy'
        if format_choice == 1:
            interaction_phrase = f"{sample['target_hero']} has a {sample['level']} synergy with {sample['champion']}."
        elif format_choice == 2:
            interaction_phrase = f"{sample['target_hero']} complements {sample['champion']}'s abilities well."
        else:
            interaction_phrase = f"{sample['target_hero']} and {sample['champion']} make a great team."

    comment = sample['comment']

    # Combine the instruction, interaction phrase, and comment into the formatted prompt
    formatted_prompt = f"{instruction}\n\n{interaction_phrase} {comment}"

    # Add the formatted prompt to the 'text' field of the sample
    sample["text"] = formatted_prompt

    return sample

def create_prompt_formats_abilities(sample):
    # Randomly select one of three different formats for the prompt
    format_choice = random.choice([1, 2, 3])

    # Combine the instruction, interaction phrase, and comment into the formatted prompt
    if format_choice == 1:
        formatted_prompt = f"{sample['champion_name']}, {sample['champion_alias']}, has '{sample['ability_type']}' ability called {sample['ability_name']}, which range is {sample['ability_range']} and {sample['ability_description']}"
    elif format_choice == 2:
        formatted_prompt = f"{sample['champion_name']}'s {sample['ability_name']} ({sample['ability_type']}) has a range of {sample['ability_range']} and {sample['ability_description']}"
    else:
        formatted_prompt = f"{sample['champion_alias']} ({sample['champion_name']}) has an ability called {sample['ability_name']} ({sample['ability_type']}) with a range of {sample['ability_range']} that {sample['ability_description']}"

    # Add the formatted prompt to the 'text' field of the sample
    sample["text"] = formatted_prompt

    return sample

def get_max_length(model):
    conf = model.config
    max_length = None
    for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
        max_length = getattr(model.config, length_setting, None)
        if max_length:
            print(f"Found max lenth: {max_length}")
            break
    if not max_length:
        max_length = 1024
        print(f"Using default max length: {max_length}")
    return max_length


def preprocess_batch(batch, tokenizer, max_length):
    """
    Tokenizing a batch
    """
    return tokenizer(
        batch["text"],
        max_length=max_length,
        truncation=True,
    )

def preprocess_dataset_tips(tokenizer: AutoTokenizer, max_length: int, seed, dataset: str):
    """Format & tokenize it so it is ready for training
    :param tokenizer (AutoTokenizer): Model Tokenizer
    :param max_length (int): Maximum number of tokens to emit from tokenizer
    """

    # Add prompt to each sample
    print("Preprocessing dataset...")
    dataset = dataset.map(create_prompt_formats_tips)#, batched=True)

    # Apply preprocessing to each batch of the dataset & and remove 'author', 'author_link', 'date' fields
    _preprocessing_function = partial(preprocess_batch, max_length=max_length, tokenizer=tokenizer)
    dataset = dataset.map(
        _preprocessing_function,
        batched=True,
        remove_columns=['author', 'author_link', 'date', 'category', 'target_hero', 'level', 'comment'],
    )

    # Filter out samples that have input_ids exceeding max_length
    dataset = dataset.filter(lambda sample: len(sample["input_ids"]) < max_length)

    # Shuffle dataset
    dataset = dataset.shuffle(seed=seed)

    return dataset

# SOURCE https://github.com/databrickslabs/dolly/blob/master/training/trainer.py
def preprocess_dataset_abilities(tokenizer: AutoTokenizer, max_length: int, seed, dataset: str):
    """Format & tokenize it so it is ready for training
    :param tokenizer (AutoTokenizer): Model Tokenizer
    :param max_length (int): Maximum number of tokens to emit from tokenizer
    """

    # Add prompt to each sample
    print("Preprocessing dataset...")
    dataset = dataset.map(create_prompt_formats_abilities)#, batched=True)

    # Apply preprocessing to each batch of the dataset & and remove 'author', 'author_link', 'date' fields
    _preprocessing_function = partial(preprocess_batch, max_length=max_length, tokenizer=tokenizer)
    dataset = dataset.map(
        _preprocessing_function,
        batched=True,
        remove_columns=['champion_alias', 'champion_name', 'ability_type', 'ability_name', 'ability_description', 'ability_range'],
    )

    # Filter out samples that have input_ids exceeding max_length
    dataset = dataset.filter(lambda sample: len(sample["input_ids"]) < max_length)

    # Shuffle dataset
    dataset = dataset.shuffle(seed=seed)

    return dataset

def create_bnb_config():
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        # load_in_4bit=True,
        # bnb_4bit_use_double_quant=True,
        # bnb_4bit_quant_type="nf4",
        # bnb_4bit_compute_dtype=torch.bfloat16,
    )

    return bnb_config

def create_peft_config(modules):
    """
    Create Parameter-Efficient Fine-Tuning config for your model
    :param modules: Names of the modules to apply Lora to
    """
    config = LoraConfig(
        r=16,  # dimension of the updated matrices
        lora_alpha=64,  # parameter for scaling
        target_modules=modules,
        lora_dropout=0.1,  # dropout probability for layers
        bias="none",
        task_type="CAUSAL_LM",
    )

    return config

def find_all_linear_names(model):
    cls = bnb.nn.Linear8bitLt #if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def print_trainable_parameters(model, use_4bit=False):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    if use_4bit:
        trainable_params /= 2
    print(
        f"all params: {all_param:,d} || trainable params: {trainable_params:,d} || trainable%: {100 * trainable_params / all_param}"
    )

# Load model from HF with user's token and with bitsandbytes config

model_name = "meta-llama/Llama-2-7b-hf"

bnb_config = create_bnb_config()

model, tokenizer = load_model(model_name, bnb_config)

## Preprocess dataset

max_length = get_max_length(model)

dataset_abilities = preprocess_dataset_abilities(tokenizer, max_length, random.seed(10), dataset_abilities)

dataset_tips = preprocess_dataset_tips(tokenizer, max_length, random.seed(10), dataset_tips)

combined_dataset = concatenate_datasets([dataset_abilities, dataset_tips])

# print(combined_dataset['text'][0])

def train(model, tokenizer, dataset, output_dir):
    # Apply preprocessing to the model to prepare it by
    # 1 - Enabling gradient checkpointing to reduce memory usage during fine-tuning
    model.gradient_checkpointing_enable()

    # 2 - Using the prepare_model_for_kbit_training method from PEFT
    model = prepare_model_for_kbit_training(model)

    # Get lora module names
    modules = find_all_linear_names(model)

    # Create PEFT config for these modules and wrap the model to PEFT
    peft_config = create_peft_config(modules)
    model = get_peft_model(model, peft_config)

    # Print information about the percentage of trainable parameters
    print_trainable_parameters(model)

    max_seq_length = 512

    # Training parameters
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            warmup_steps=2,
            max_steps=50,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=1,
            output_dir="outputs",
            # optim="paged_adamw_8bit",
        ),
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        #data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    model.config.use_cache = False  # re-enable for inference to speed up predictions for similar inputs

    ### SOURCE https://github.com/artidoro/qlora/blob/main/qlora.py
    # Verifying the datatypes before training

    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes: dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    total = 0
    for k, v in dtypes.items(): total+= v
    for k, v in dtypes.items():
        print(k, v, v/total)

    do_train = True

    # Launch training
    print("Training...")

    if do_train:
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        print(metrics)

    ###

    # Saving model
    print("Saving last checkpoint of the model...")
    os.makedirs(output_dir, exist_ok=True)
    trainer.model.save_pretrained(output_dir)

    # Free memory for merging weights
    del model
    del trainer
    torch.cuda.empty_cache()


output_dir = "results/llama2/final_checkpoint"
train(model, tokenizer, combined_dataset, output_dir)

# model = AutoPeftModelForCausalLM.from_pretrained(output_dir, device_map="auto")
# model = model.merge_and_unload()

# output_merged_dir = "results/llama2/final_merged_checkpoint"
# os.makedirs(output_merged_dir, exist_ok=True)
# model.save_pretrained(output_merged_dir, safe_serialization=True)

# # save tokenizer for easy inference
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# tokenizer.save_pretrained(output_merged_dir)