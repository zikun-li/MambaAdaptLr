from datasets import load_dataset
from trl import SFTTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from torch.optim import Adam, SGD
import argparse
import wandb

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', type=str, required=True,
                        help='Path to the dataset')
parser.add_argument('--adapt_lr', type=bool, required=True,
                        help='A boolean to adapt learning rate')

def main(dataset: str = "Abirate/english_quotes", adapt_lr: bool = False):
    tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-370m-hf")
    model = AutoModelForCausalLM.from_pretrained("state-spaces/mamba-370m-hf")
    dataset = load_dataset(dataset, split="train")
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        logging_dir="./logs",
        logging_steps=10,
        # learning_rate=2e-3,
        lr_scheduler_type='linear', # 'constant', 'constant_with_warmup'
        report_to='wandb',
        run_name=f'mamba-370m-hf_{dataset}_adapt_lr={adapt_lr}',
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

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
        dataset_text_field="quote",
        optimizers=(optimizer, None),
    )
    trainer.train()

main()