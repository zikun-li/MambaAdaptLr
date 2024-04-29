from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from torch.optim import Adam, SGD
import torch
import argparse
import wandb
import os
from torch import nn
from torch.utils.data import Dataset

parser = argparse.ArgumentParser()
parser.add_argument('-a', '--adapt_lr',  action="store_true", default=False,
                        help='A boolean to adapt learning rate')
parser.add_argument('-s', '--sqrt',  action="store_true", default=False,
                        help='Sqrt the scale of learning rate')
parser.add_argument('-g', '--gpu',  type=str, required=True, default='0',
                        help='The gpu to use to train the model')
parser.add_argument('--seed',  type=int, default=0,
                        help='The gpu to use to train the model')
# Parse the arguments
args = parser.parse_args()
adapt_lr = args.adapt_lr
sqrt = args.sqrt
seed = args.seed
gpu = args.gpu

# wandb.init(mode="disabled")
os.environ["WANDB_PROJECT"] = "mamba-adapt-lr"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu

class MMLUAnswerPredictionDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        """
        Initialize the dataset for answer prediction.

        :param data: List of dictionaries with 'question', 'choices', and 'answer' keys.
        :param tokenizer: Instance of a tokenizer compatible with the model.
        :param max_length: Maximum length of the tokenized input sequences.
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        """
        Return the number of items in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]['train']
        question = entry['question']
        choices = entry['choices']
        answer = entry['answer']
        answer_text = chr(answer + 65)

        # Format input to prompt model to generate the answer
        prompt = f"{question} Choices: {' '.join([f'{chr(65+i)}: {choice}' for i, choice in enumerate(choices)])} Answer:"
        
        # Encode input and the start of the answer
        encoding = self.tokenizer(prompt, return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_length)

        input_ids = encoding['input_ids'].squeeze(0)  # Remove the batch dimension
        attention_mask = encoding['attention_mask'].squeeze(0)

        input_length = input_ids.size(0)

        # The model should learn to predict the answer text, so the labels are the input_ids shifted
        labels = self.tokenizer(answer_text, add_special_tokens=False, return_tensors='pt').input_ids.squeeze(0)
        labels = torch.cat([torch.tensor([-100] * (input_length - 1)), labels])  # Ignore loss for the prompt part

        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-370m-hf")
model = AutoModelForCausalLM.from_pretrained("state-spaces/mamba-370m-hf")
dataset = load_dataset("cais/mmlu", "auxiliary_train", split="train")
dataset = MMLUAnswerPredictionDataset(dataset, tokenizer)
run_name = f'mamba-370m-hf_adapt_lr={adapt_lr}_sqrt={sqrt}_seed={seed}'

training_args = TrainingArguments(
    output_dir="results_" + run_name,
    num_train_epochs=1,
    per_device_train_batch_size=18,
    logging_dir="logs_" + run_name,
    logging_steps=20,
    # learning_rate=2e-3,
    lr_scheduler_type='linear', # 'constant', 'constant_with_warmup'
    report_to='wandb',
    run_name=run_name,
    save_strategy='epoch',
    seed=seed,
    remove_unused_columns = False,
)


if adapt_lr:
    # Model specs
    lr = 1e-3
    d_model = 1024
    d_inner = 160
    d_state = 16
    conv_kernel = 4
    dt_rank = d_model / 16

    A_log = []
    D = []
    conv1d_weight = []
    conv1d_bias = []
    in_proj_weight = []
    x_proj_weight = []
    dt_proj_weight = []
    dt_proj_bias = []
    out_proj_weight = []
    other = []

    for name, param in model.named_parameters():
        if 'A_log' in name:
            A_log.append(param)
        elif 'mixer.D' in name:
            D.append(param)
        elif 'conv1d.weight' in name:
            conv1d_weight.append(param)
        elif 'conv1d.bias' in name:
            conv1d_bias.append(param)
        elif 'in_proj.weight' in name:
            in_proj_weight.append(param)
        elif 'x_proj.weight' in name:
            x_proj_weight.append(param)
        elif 'dt_proj.weight' in name:
            dt_proj_weight.append(param)
        elif 'dt_proj.bias' in name:
            dt_proj_bias.append(param)
        elif 'out_proj.weight' in name:
            out_proj_weight.append(param)
        else:
            other.append(param)
    if sqrt:
        # To make the geo mean of the highest and lowest learning rate equal to lr
        lr *= (d_model ** 0.25)
        optimizer = Adam([
            {'params': A_log, 'lr': lr / d_state ** 0.5},
            {'params': D, 'lr': lr},
            {'params': conv1d_weight, 'lr': lr / (conv_kernel * d_inner) ** 0.5},
            {'params': conv1d_bias, 'lr': lr},
            {'params': in_proj_weight, 'lr': lr / d_model ** 0.5},
            {'params': x_proj_weight, 'lr': lr / d_inner ** 0.5},
            {'params': dt_proj_weight, 'lr': lr / dt_rank ** 0.5},
            {'params': dt_proj_bias, 'lr': lr },
            {'params': out_proj_weight, 'lr': lr / d_inner ** 0.5},
            {'params': other, 'lr': lr},
        ])
    else:
        # To make the geo mean of the highest and lowest learning rate equal to lr
        lr *= (d_model ** 0.5)
        optimizer = Adam([
            {'params': A_log, 'lr': lr / d_state},
            {'params': D, 'lr': lr},
            {'params': conv1d_weight, 'lr': lr / conv_kernel / d_inner},
            {'params': conv1d_bias, 'lr': lr},
            {'params': in_proj_weight, 'lr': lr / d_model},
            {'params': x_proj_weight, 'lr': lr / d_inner },
            {'params': dt_proj_weight, 'lr': lr / dt_rank},
            {'params': dt_proj_bias, 'lr': lr },
            {'params': out_proj_weight, 'lr': lr / d_inner},
            {'params': other, 'lr': lr},
        ])
else:
    optimizer = Adam(model.parameters(), lr=1e-3)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,  # Ensure your dataset is correctly formatted
    optimizers=(optimizer, None),
)

trainer.train()