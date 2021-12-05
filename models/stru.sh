#!/bin/sh

CUDA_VISIBLE_DEVICES=0 python gcn.py --data_dir "zh_en"
CUDA_VISIBLE_DEVICES=0 python gcn.py --data_dir "ja_en"
CUDA_VISIBLE_DEVICES=0 python gcn.py --data_dir "fr_en"
CUDA_VISIBLE_DEVICES=0 python gcn.py --data_dir "en_fr_15k_V1"
CUDA_VISIBLE_DEVICES=0 python gcn.py --data_dir "en_de_15k_V1"
CUDA_VISIBLE_DEVICES=0 python gcn.py --data_dir "dbp_yg_15k_V1"
CUDA_VISIBLE_DEVICES=0 python gcn.py --data_dir "dbp_wd_15k_V1"
CUDA_VISIBLE_DEVICES=0 python gcn.py --data_dir "mul"

CUDA_VISIBLE_DEVICES=0 python rrea.py --data_dir "zh_en"
CUDA_VISIBLE_DEVICES=0 python rrea.py --data_dir "ja_en"
CUDA_VISIBLE_DEVICES=0 python rrea.py --data_dir "fr_en"
CUDA_VISIBLE_DEVICES=0 python rrea.py --data_dir "en_fr_15k_V1"
CUDA_VISIBLE_DEVICES=0 python rrea.py --data_dir "en_de_15k_V1"
CUDA_VISIBLE_DEVICES=0 python rrea.py --data_dir "dbp_yg_15k_V1"
CUDA_VISIBLE_DEVICES=0 python rrea.py --data_dir "dbp_wd_15k_V1"
CUDA_VISIBLE_DEVICES=0 python rrea.py --data_dir "mul"

CUDA_VISIBLE_DEVICES=0 python gcn.py --data_dir "dbp_wd_100"
CUDA_VISIBLE_DEVICES=0 python gcn.py --data_dir "dbp_yg_100"

