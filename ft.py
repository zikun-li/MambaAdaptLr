from datasets import load_dataset
from trl import SFTTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from torch.optim import Adam, SGD
import torch
import argparse
import wandb
import os

parser = argparse.ArgumentParser()
parser.add_argument('-a', '--adapt_lr',  action="store_true", default=False,
                        help='A boolean to adapt learning rate')
parser.add_argument('-s', '--sqrt',  action="store_true", default=False,
                        help='Sqrt the scale of learning rate')
parser.add_argument('-g', '--gpu',  type=str, required=True, default='0',
                        help='The gpu to use to train the model')
# Parse the arguments
args = parser.parse_args()

os.environ["WANDB_PROJECT"] = "mamba-adapt-lr"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

def main(adapt_lr: bool, sqrt: bool):
    tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-370m-hf")
    model = AutoModelForCausalLM.from_pretrained("state-spaces/mamba-370m-hf")
    dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    training_args = TrainingArguments(
        output_dir=f"./results_adapt_lr={adapt_lr}_sqrt={sqrt}",
        num_train_epochs=3,
        per_device_train_batch_size=12,
        logging_dir=f"./logs_adapt_lr={adapt_lr}_sqrt={sqrt}",
        logging_steps=10,
        # learning_rate=2e-3,
        lr_scheduler_type='linear', # 'constant', 'constant_with_warmup'
        report_to='wandb',
        run_name=f'mamba-370m-hf_adapt_lr={adapt_lr}_sqrt={sqrt}',
    )


    if adapt_lr:
        # Model specs
        lr = 2e-3
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
        optimizer = Adam(model.parameters(), lr=2e-3)

    # device = torch.device("cuda:0")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
        # dataset_text_field="quote",
        optimizers=(optimizer, None),
    )

    # model.to(device)
    trainer.train()

main(args.adapt_lr, args.sqrt)