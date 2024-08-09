from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType, AutoPeftModelForSequenceClassification
from datasets import load_dataset, DatasetDict, Dataset
import numpy as np
import pandas as pd
import torch
import sys



#NOTE REMEMBER: HuggingFace's Trainer expects the target labels for classifications to be named 'labels' and not 'label'
#     With that said, will need to rename "profession" to "labels"


print("##################################################################")
print("##################################################################")
print("##########################   WELCOME!   ##########################")
print("##################################################################")
print("##################################################################")
print("##########################  Developer:  ##########################")
print("##########################   Zacharie   ##########################")
print("##########################   Happel     ##########################")
print("##################################################################")
print("##################################################################")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n\nUsing device: {device}")
if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(torch.cuda.current_device())
    print(f"Using GPU: {device_name}")
else:
    print("CUDA is not available. Using CPU.")



splits = ["train", "test", "dev"]
dataset_name = "LabHC/bias_in_bios"

print(f"Loading Dataset [{dataset_name}]...", flush=True)
dataset = {split: load_dataset(dataset_name, split=split) for split in splits}

print("\n\n##################################################################")
print("####################### Dataset Operations #######################")
print("##################################################################")


print("\nSplitting Data...")

dev_dataset = dataset['dev'] # Will be unused

train_data_split = dataset["train"].train_test_split(test_size=0.9) # Keep only 10%, Discard the rest
training_dataset = train_data_split["train"]

test_dataset_split = dataset['test'].train_test_split(test_size=0.9) # Keep only 10%, split into 'train' and 'test' where 'train' is the 10%
halved_test_dataset = test_dataset_split["train"].train_test_split(0.5) # Split into halves, half to be used for pre-training, half for post-training
pre_training_dataset = halved_test_dataset["train"] # Pre Training
post_training_dataset = halved_test_dataset["test"] # Post Training

'''
print("train data set split: ", train_data_split)
print("train data: ", training_dataset)
print("halved_test_dataset: ", halved_test_dataset)
print("pre_testing_dataset: ", pre_training_dataset)
print("post_testing_dataset: ", post_training_dataset)
'''

#print("\nData - Train - [Row Count: ", training_dataset.num_rows, " | Features: ", training_dataset.features, "]")
#print("Data - Pre-Training - [Row Count: ", pre_training_dataset.num_rows, " | Features: ", pre_training_dataset.features, "]")
#print("Data - Post-Training - [Row Count: ", post_training_dataset.num_rows, " | Features: ", post_training_dataset.features, "]")


# Standard dict to hold all Dataset objects
dataset_dictionary = {
    "training": training_dataset,
    "pre_training": pre_training_dataset,
    "post_training": post_training_dataset
}

print("\n\nDataset: ", dataset_dictionary)


def analyze_data(dataset_dictionary, target_label):
    label_counts = {}
    percentages = {}
    
    for subset in dataset_dictionary:
        print(f"\n   Subset \"{subset}\": ")

        for label in dataset_dictionary[subset][target_label]:
            if label not in label_counts: label_counts[label] = 0 #set default
            label_counts[label] += 1 #create new
        
        for label, count in label_counts.items():
            percentages[label] = (count / dataset_dictionary[subset].num_rows) * 100
            
        for label, percentage in percentages.items():
            print(f'   [Label: {label}]  Subset respresentation: { round(percentage, 2)}%')

        label_counts = {}
        percentages = {}

print('\nAnalyze Data - Subset Information: ')
analyze_data(dataset_dictionary, target_label='gender')



print("\n\n##################################################################")
print("#######################     Tokenizing     #######################")
print("##################################################################")

tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
tokenized_ds = {}


def preprocess_function(examples):
    return tokenizer(examples['hard_text'], padding=True, truncation=True)

def convert_to_tensors_and_move_to_device(tokenized_inputs):
    tensor_inputs = {}
    for key, value in tokenized_inputs.items():
        #print("key: ", key)
        if isinstance(value, list) and all(isinstance(i, int) for i in value):
            tensor_inputs[key] = torch.tensor(value).to(device)
        elif isinstance(value, list) and len(value) > 0:
            if isinstance(value[0], list):
                #print("Embedded list, value[0]")
                #print("List: ", value[0])
                tensor_inputs[key] = torch.tensor(value).to(device)
            else:
                pass
                #print("Embedded value is not list...")    
        else:
            pass
            #print(f"Skipping {key} as it contains non-numeric or invalid data: Type - {type(value)}, Example - {value[:5] if isinstance(value, list) else value}")
    return tensor_inputs



print("\nTokenizing dataset...")
for split in ["training", "pre_training", "post_training"]:
    
    print(f" - Using preprocessor to process split: {split}")
    tokenized_split = dataset_dictionary[split].map(preprocess_function, batched=True)
    
    #NOTE Uncomment if you want to see an example printed from each subset that gets processed
    #print(f"Tokenized {split} example: {tokenized_split[0]}")
    tokenized_split = tokenized_split.rename_column("gender", "labels") #For HF Trainer expectations
    
    # Convert tokenized outputs to tensors and move to GPU
    tokenized_ds[split] = tokenized_split.map(
        lambda x: convert_to_tensors_and_move_to_device(x), 
        batched=True
    )


print("\n\n\n##################################################################")
print("######################   Preparing Model   #######################")
print("##################################################################")

model_name = 'gpt2'

print(f"\nLoading model [{model_name}]...")
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
    id2label={0: "MALE", 1: "FEMALE"},
    label2id={"MALE": 0, "FEMALE": 1}
)


print(f'\nMoving model to [{device}]...')
model = model.to(device)

print("Configuring padding token\nand freezing base model parameters...")
model.config.pad_token_id = tokenizer.pad_token_id
for param in model.base_model.parameters():
    param.requires_grad = False

print("\n~~~~~~~~~~~ Model info ~~~~~~~~~~~")
print(model)
print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")



print("##################################################################")
print("##################   Pre-Training Evaluation   ###################")
print("##################################################################")


training_args = TrainingArguments(
    output_dir="./model_output",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

print("Preparing data collator")
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {"accuracy": (predictions == labels).mean()}

print("Configuring trainer...")
print("Training arguments:")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["training"],
    eval_dataset=tokenized_ds["pre_training"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("Performing pre-training evaluation...")
pretrain_results = trainer.evaluate()

print("\n\nEvaluation results prior to training:\n", pretrain_results, "\n\n\n")



#### PEFT
print("##################################################################")
print("##########   Parameter Efficient Fine-Tuning (PEFT)   ############")
print("### Using Low Rank Adapatation of Large Language Models (LoRA) ###")
print("##################################################################")




lora_saveas = "gpt2-lora-bias_in_bios" # Name to save lora as
lora_savedir = "./lora_model_output-bias_in_bios"

config = LoraConfig(
                    r=8, # Rank
                    lora_alpha=32,
                    target_modules=['c_attn', 'c_proj'],
                    lora_dropout=0.1,
                    bias="none",
                    task_type=TaskType.SEQ_CLS #Sequence classification
                )

print("\nRetrieving model...")
print("PEFT model configuration:\n", config)
peft_model = get_peft_model(model, config)

print("\nTrainable parameters:")
peft_model.print_trainable_parameters()


print("\nConfiguring trainer...")
trainer = Trainer(
    model=peft_model,
    args=TrainingArguments(
        output_dir=lora_savedir,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_dir='./logs',
        #fp16=True, 
        #dataloader_num_workers=4, 
    ),
    train_dataset=tokenized_ds["training"],
    eval_dataset=tokenized_ds["pre_training"],
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer, padding=True, max_length=512),
    compute_metrics=compute_metrics,
)

## Printing
print("\nTraining arguments:")
print(f"Output Directory: {trainer.args.output_dir}")
print(f"Learning Rate: {trainer.args.learning_rate}")
print(f"Per Device Train Batch Size: {trainer.args.per_device_train_batch_size}")
print(f"Per Device Eval Batch Size: {trainer.args.per_device_eval_batch_size}")
print(f"Number of Train Epochs: {trainer.args.num_train_epochs}")
print(f"Weight Decay: {trainer.args.weight_decay}")
print(f"Evaluation Strategy: {trainer.args.evaluation_strategy}")
print(f"Save Strategy: {trainer.args.save_strategy}")
print(f"Load Best Model at End: {trainer.args.load_best_model_at_end}")
print(f"Logging Directory: {trainer.args.logging_dir}")
print(f"FP16: {trainer.args.fp16}")
print(f"DataLoader Num Workers: {trainer.args.dataloader_num_workers}")
##

print("\n\nPerforming LoRA training...\n\n")
trainer.train()

print(f"\nSaving LoRA, as [{lora_saveas}]")
peft_model.save_pretrained(lora_saveas)



print("##################################################################")
print("#########################   Inference   ##########################")
print("##################################################################")

'''
Notes:
Our "lora_model" is the model with the base GPT-2 weights and our LoRA-adapted parameters

Approach: 
Using this approach, we do not need to interact with the initial base model and the LoRA separately
(unlike, for instance, Stable Diffusion's usage of LoRAs when using Comfy or Automatic1111)
'''

lora_model = AutoPeftModelForSequenceClassification.from_pretrained(lora_saveas, num_labels=2, ignore_mismatched_sizes=True).to(device)
lora_model.config.pad_token_id = tokenizer.pad_token_id

inference_savedir = "./data/comparison-bias_in_bios"

training_args = TrainingArguments(
    output_dir=inference_savedir, 
    learning_rate=2e-5,
    per_device_train_batch_size=16, 
    per_device_eval_batch_size=16, 
    num_train_epochs=1, 
    weight_decay=0.01, 
    eval_strategy="epoch", 
    save_strategy="epoch", 
    load_best_model_at_end=True, 
)

evaluation_trainer = Trainer(
    model=lora_model,
    args=training_args,
    eval_dataset=tokenized_ds["post_training"], 
    tokenizer=tokenizer, 
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics,
)

print(f"Performing [{lora_saveas}] inference evaluation...")
evaluation_results = evaluation_trainer.evaluate()

print("\n\n Evaluation of the adapted model after performing lightweight tuning using PEFT/LoRA:\n", evaluation_results)

sys.exit()
