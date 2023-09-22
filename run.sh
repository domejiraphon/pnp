#!/bin/bash

prompts=(
    "leaning tower of pisa on moon"
    #"rhino dressed suit and tie sitting at a table in a bar with a bar stools, award winning photography"
    # "astronaut in times square"
)

ress=(
    "block1-attn0-trans0"
)
export CUDA_VISIBLE_DEVICES=1
for prompt in "${prompts[@]}"; do
    for res in "${ress[@]}"; do
        python pnp_xl.py \
            --visualization_every 5 \
            --prompt "$prompt" \
            --attn_layer "$res" \
            --deformed_at_t=0.5 \
            --deformed
    done
done
