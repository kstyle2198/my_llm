import streamlit as st
from datasets import load_dataset

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import transformers



#Download the finetuning dataset
train_dataset = load_dataset('gem/viggo', split='train')
eval_dataset = load_dataset('gem/viggo', split='validation')
test_dataset = load_dataset('gem/viggo', split='test')


# Loading the model and tokenizer
model_id = "EleutherAI/gpt-neox-20b"

#configuration object "bnb_config" for the "BitsAndBytes" (bnb) library.
#enable 4-bit quantization for a deep learning model,
#specifically using the "torch" library with a bfloat16 compute data type
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

#Load the pretarined tokenizer for the pretrained Model
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

def tokenize(prompt):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=512,
        padding="max_length",
    )
    result["labels"] = result["input_ids"].copy()
    return result

def generate_and_tokenize_prompt(data_point):
    full_prompt =f"""Given a target sentence construct the underlying meaning representation of
                     the input sentence as a single function with attributes and attribute values.
                     This function should describe the target string accurately and the 
                     function must be one of the following 
                     ['inform', 'request', 'give_opinion', 'confirm', 'verify_attribute', 
                     'suggest', 'request_explanation', 'recommend', 'request_attribute'].
                      The attributes must be one of the following: 
                      ['name', 'exp_release_date', 'release_year', 'developer', 
                      'esrb', 'rating', 'genres', 'player_perspective', 'has_multiplayer', 
                      'platforms', 'available_on_steam', 'has_linux_release', 'has_mac_release', 'specifier']


                    ### Target sentence:
                    {data_point["target"]}


                    ### Meaning representation:
                    {data_point["meaning_representation"]}
                    """
    return tokenize(full_prompt)

tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt)
tokenized_val_dataset = eval_dataset.map(generate_and_tokenize_prompt)


#loads the pre-trained causal language model and apply quantization
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto", low_cpu_mem_usage=True)

#Sets the pad_token to be the same as the eos_token, 
#ensuring consistency in model input and output.
# tokenizer.pad_token = tokenizer.eos_token

model.gradient_checkpointing_enable()

'''This method wraps the entire protocol for preparing a model before running a training.
This includes:
1- Cast the layernorm in fp32 to ensures accurate normalization of layer outputs for smoother training
2- making output embedding layer require grads allows the model to learn better word representations during training.
3- Add the upcasting of the lm head to fp32 nhances precision in the language modeling component, leading to better predictions.
#'''
model = prepare_model_for_kbit_training(model) 
config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["query_key_value"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
#Returns a Peft model object from a model and a config.
model = get_peft_model(model, config)
# print_trainable_parameters(model)

trainer = transformers.Trainer(
    model=model,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=2,
        max_steps=10,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        output_dir="outputs",
        optim="paged_adamw_8bit"
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False  

if st.button("Train"):
    trainer.train()


    model.save_pretrained("qloraFinetuned_Model")