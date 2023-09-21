#!/bin/bash

prompts=(
    #"leaning tower of pisa on moon"
    #"rhino dressed suit and tie sitting at a table in a bar with a bar stools, award winning photography"
    "astronaut in times square"
)

ress=(
    "block0-res0"
    "block0-res1"
    "block0-res2"
    "block1-res0"
    "block1-res1"
    "block1-res2"
    "block2-res0"
    "block2-res1"
    "block2-res2"
)
export CUDA_VISIBLE_DEVICES=1
for prompt in "${prompts[@]}"; do
    for res in "${ress[@]}"; do
        python attention_xl.py \
            --visualization_every 5 \
            --prompt "$prompt" \
            --res_layer "$res"
    done
done
