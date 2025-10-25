#!/bin/bash
set -e

INPUT=../your_input.csv

python src/cli_pipeline_minilm_only.py \
  --input_csv $INPUT \
  --output_dir ./outputs_minilm \
  --recall_target 0.95

python src/cli_pipeline_light.py \
  --input_csv $INPUT \
  --output_dir ./outputs_light \
  --recall_target 0.95 \
  --weights 0.5,0.3,0.2 \
  --domain_terms "glymphatic,perivascular,pvs,csf,diffusion,asl,flair,spectroscopy"

python src/cli_pipeline_heavy.py \
  --input_csv $INPUT \
  --output_dir ./outputs_heavy \
  --recall_target 0.95 \
  --alpha 0.7 \
  --max_len 256 \
  --epochs 4 \
  --batch_size 8 \
  --lr 2e-5 \
  --warmup_ratio 0.1 \
  --calibrate_platt \
  --domain_terms "glymphatic,perivascular,pvs,csf,diffusion,asl,flair,spectroscopy"
