#!/bin/sh

# Table 3
CUDA_VISIBLE_DEVICES=0 python ./infer.py --data_dir "zh_en" --encoder "gcn" --features "stru"  --sim "cosine" --scoreop "none" --match "sm"
CUDA_VISIBLE_DEVICES=0 python ./infer.py --data_dir "ja_en" --encoder "gcn" --features "stru"  --sim "cosine" --scoreop "none" --match "sm"
CUDA_VISIBLE_DEVICES=0 python ./infer.py --data_dir "fr_en" --encoder "gcn" --features "stru"  --sim "cosine" --scoreop "none" --match "sm"
CUDA_VISIBLE_DEVICES=0 python ./infer.py --data_dir "en_fr_15k_V1" --encoder "gcn" --features "stru"  --sim "cosine" --scoreop "none" --match "sm"
CUDA_VISIBLE_DEVICES=0 python ./infer.py --data_dir "en_de_15k_V1" --encoder "gcn" --features "stru"  --sim "cosine" --scoreop "none" --match "sm"
CUDA_VISIBLE_DEVICES=0 python ./infer.py --data_dir "dbp_wd_15k_V1" --encoder "gcn" --features "stru"  --sim "cosine" --scoreop "none" --match "sm"
CUDA_VISIBLE_DEVICES=0 python ./infer.py --data_dir "dbp_yg_15k_V1" --encoder "gcn" --features "stru"  --sim "cosine" --scoreop "none" --match "sm"

CUDA_VISIBLE_DEVICES=0 python ./infer.py --data_dir "zh_en" --encoder "rrea" --features "stru"  --sim "cosine" --scoreop "none" --match "sm"
CUDA_VISIBLE_DEVICES=0 python ./infer.py --data_dir "ja_en" --encoder "rrea" --features "stru"  --sim "cosine" --scoreop "none" --match "sm"
CUDA_VISIBLE_DEVICES=0 python ./infer.py --data_dir "fr_en" --encoder "rrea" --features "stru"  --sim "cosine" --scoreop "none" --match "sm"
CUDA_VISIBLE_DEVICES=0 python ./infer.py --data_dir "en_fr_15k_V1" --encoder "rrea" --features "stru"  --sim "cosine" --scoreop "none" --match "sm"
CUDA_VISIBLE_DEVICES=0 python ./infer.py --data_dir "en_de_15k_V1" --encoder "rrea" --features "stru"  --sim "cosine" --scoreop "none" --match "sm"
CUDA_VISIBLE_DEVICES=0 python ./infer.py --data_dir "dbp_wd_15k_V1" --encoder "rrea" --features "stru"  --sim "cosine" --scoreop "none" --match "sm"
CUDA_VISIBLE_DEVICES=0 python ./infer.py --data_dir "dbp_yg_15k_V1" --encoder "rrea" --features "stru"  --sim "cosine" --scoreop "none" --match "sm"


## Table 7
CUDA_VISIBLE_DEVICES=0 python ./infer.py --mode "unm" --data_dir "zh_en" --encoder "gcn" --features "stru"  --sim "cosine" --scoreop "none" --match "sm"
CUDA_VISIBLE_DEVICES=0 python ./infer.py --mode "unm" --data_dir "ja_en" --encoder "gcn" --features "stru"  --sim "cosine" --scoreop "none" --match "sm"
CUDA_VISIBLE_DEVICES=0 python ./infer.py --mode "unm" --data_dir "fr_en" --encoder "gcn" --features "stru"  --sim "cosine" --scoreop "none" --match "sm"
#
## Table 8
CUDA_VISIBLE_DEVICES=0 python ./infer.py --mode "mul" --data_dir "mul" --encoder "gcn" --features "stru"  --sim "cosine" --scoreop "none" --match "sm"