#!/bin/bash
python ft.py --dataset "lukaemon/mmlu" --adapt_lr --sqrt &
python ft.py --dataset "lukaemon/mmlu" --adapt_lr &
python ft.py --dataset "lukaemon/mmlu" &
