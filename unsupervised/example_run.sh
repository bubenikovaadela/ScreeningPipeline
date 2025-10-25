#!/bin/bash
set -e

INPUT=../your_input.csv

python src/cli_pipeline_unsupervised.py \
  --input_csv $INPUT \
  --output_dir ./outputs_unsup \
  --domain_terms "glymphatic,perivascular,pvs,csf,diffusion,asl,flair,spectroscopy" \
  --weights 0.5,0.5 \
  --top_k 200

python src/cli_pipeline_weaksup.py \
  --input_csv $INPUT \
  --output_dir ./outputs_weaksup \
  --pos_keywords "glymphatic|perivascular|perivascular space|pvs|csf circulation|phase-contrast mri|4d flow|asl|flair" \
  --neg_keywords "cardiac|renal|liver" \
  --top_k 200
