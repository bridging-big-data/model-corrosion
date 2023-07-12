#!/bin/bash

echo "run python nohup"

nohup \
 python -W ignore corrosion.py \
 train \
 --dataset=./../../dataset/Corrosion_Condition_State_Classification/processed_512/ \
 --weights=coco \
 --logs=./../../logs/corrosion \
&
