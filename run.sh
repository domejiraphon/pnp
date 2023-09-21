#!/bin/bash

prompts=(
    "leaning tower of pisa on moon"
    "rhino dressed suit and tie sitting at a table in a bar with a bar stools, award winning photography"
    "astronaut in times square"
)

ress=(
    "block0-attn2-trans0"
    "block0-attn2-trans9"
    "block1-attn2-trans1"
)
export CUDA_VISIBLE_DEVICES=1
for prompt in "${prompts[@]}"; do
    for res in "${ress[@]}"; do
        python pnp_xl.py \
            --visualization_every 5 \
            --prompt "$prompt" \
            --attn_layer "$res"
    done
done
