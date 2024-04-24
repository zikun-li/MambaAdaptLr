#!/bin/bash
python ft.py --adapt_lr --sqrt --gpu 5 &
python ft.py --adapt_lr --gpu 6 &
python ft.py --gpu 7 &
