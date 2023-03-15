#!/bin/sh
#done
#algorithm dinf
#echo "only name:--algorithm dinf --mode 1-to-1 --encoder rrea --features name --data_dir:DBP15K/D-Z,DBP15K/D-J,DBP15K/D-F,SRPRS/S-F,SRPRS/S-D:"
#CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm dinf --mode 1-to-1 --encoder rrea --features name --data_dir "../data/zh_en"
#CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm dinf --mode 1-to-1 --encoder rrea --features name --data_dir "../data/ja_en"
#CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm dinf --mode 1-to-1 --encoder rrea --features name --data_dir "../data/fr_en"
#CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm dinf --mode 1-to-1 --encoder rrea --features name --data_dir "../data/en_fr_15k_V1"
#CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm dinf --mode 1-to-1 --encoder rrea --features name --data_dir "../data/en_de_15k_V1"


#echo "name and structural:--algorithm dinf --mode 1-to-1 --encoder rrea --features struname --data_dir:DBP15K/D-Z,DBP15K/D-J,DBP15K/D-F,SRPRS/S-F,SRPRS/S-D:"
#CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm dinf --mode 1-to-1 --encoder rrea --features struname --data_dir "../data/zh_en"
#CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm dinf --mode 1-to-1 --encoder rrea --features struname --data_dir "../data/ja_en"
#CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm dinf --mode 1-to-1 --encoder rrea --features struname --data_dir "../data/fr_en"
#CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm dinf --mode 1-to-1 --encoder rrea --features struname --data_dir "../data/en_fr_15k_V1"
#CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm dinf --mode 1-to-1 --encoder rrea --features struname --data_dir "../data/en_de_15k_V1"



#algorithm csls

echo "only name:--algorithm csls --mode 1-to-1 --encoder rrea --features name --data_dir:DBP15K/D-Z,DBP15K/D-J,DBP15K/D-F,SRPRS/S-F,SRPRS/S-D:"
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm csls --mode 1-to-1 --encoder rrea --features name --data_dir "../data/zh_en"
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm csls --mode 1-to-1 --encoder rrea --features name --data_dir "../data/ja_en"
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm csls --mode 1-to-1 --encoder rrea --features name --data_dir "../data/fr_en"
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm csls --mode 1-to-1 --encoder rrea --features name --data_dir "../data/en_fr_15k_V1"
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm csls --mode 1-to-1 --encoder rrea --features name --data_dir "../data/en_de_15k_V1"


echo "name and structural:--algorithm csls --mode 1-to-1 --encoder rrea --features struname --data_dir:DBP15K/D-Z,DBP15K/D-J,DBP15K/D-F,SRPRS/S-F,SRPRS/S-D:"
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm csls --mode 1-to-1 --encoder rrea --features struname --data_dir "../data/zh_en"
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm csls --mode 1-to-1 --encoder rrea --features struname --data_dir "../data/ja_en"
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm csls --mode 1-to-1 --encoder rrea --features struname --data_dir "../data/fr_en"
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm csls --mode 1-to-1 --encoder rrea --features struname --data_dir "../data/en_fr_15k_V1"
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm csls --mode 1-to-1 --encoder rrea --features struname --data_dir "../data/en_de_15k_V1"



#algorithm rinf

echo "only name:--algorithm rinf --mode 1-to-1 --encoder rrea --features name --data_dir:DBP15K/D-Z,DBP15K/D-J,DBP15K/D-F,SRPRS/S-F,SRPRS/S-D:"
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm rinf --mode 1-to-1 --encoder rrea --features name --data_dir "../data/zh_en"
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm rinf --mode 1-to-1 --encoder rrea --features name --data_dir "../data/ja_en"
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm rinf --mode 1-to-1 --encoder rrea --features name --data_dir "../data/fr_en"
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm rinf --mode 1-to-1 --encoder rrea --features name --data_dir "../data/en_fr_15k_V1"
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm rinf --mode 1-to-1 --encoder rrea --features name --data_dir "../data/en_de_15k_V1"


echo "name and structural:--algorithm rinf --mode 1-to-1 --encoder rrea --features struname --data_dir:DBP15K/D-Z,DBP15K/D-J,DBP15K/D-F,SRPRS/S-F,SRPRS/S-D:"
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm rinf --mode 1-to-1 --encoder rrea --features struname --data_dir "../data/zh_en"
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm rinf --mode 1-to-1 --encoder rrea --features struname --data_dir "../data/ja_en"
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm rinf --mode 1-to-1 --encoder rrea --features struname --data_dir "../data/fr_en"
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm rinf --mode 1-to-1 --encoder rrea --features struname --data_dir "../data/en_fr_15k_V1"
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm rinf --mode 1-to-1 --encoder rrea --features struname --data_dir "../data/en_de_15k_V1"


#algorithm sinkhorn

echo "only name:--algorithm sinkhorn --mode 1-to-1 --encoder rrea --features name --data_dir:DBP15K/D-Z,DBP15K/D-J,DBP15K/D-F,SRPRS/S-F,SRPRS/S-D:"
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm sinkhorn --mode 1-to-1 --encoder rrea --features name --data_dir "../data/zh_en"
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm sinkhorn --mode 1-to-1 --encoder rrea --features name --data_dir "../data/ja_en"
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm sinkhorn --mode 1-to-1 --encoder rrea --features name --data_dir "../data/fr_en"
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm sinkhorn --mode 1-to-1 --encoder rrea --features name --data_dir "../data/en_fr_15k_V1"
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm sinkhorn --mode 1-to-1 --encoder rrea --features name --data_dir "../data/en_de_15k_V1"


echo "name and structural:--algorithm sinkhorn --mode 1-to-1 --encoder rrea --features struname --data_dir:DBP15K/D-Z,DBP15K/D-J,DBP15K/D-F,SRPRS/S-F,SRPRS/S-D:"
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm sinkhorn --mode 1-to-1 --encoder rrea --features struname --data_dir "../data/zh_en"
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm sinkhorn --mode 1-to-1 --encoder rrea --features struname --data_dir "../data/ja_en"
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm sinkhorn --mode 1-to-1 --encoder rrea --features struname --data_dir "../data/fr_en"
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm sinkhorn --mode 1-to-1 --encoder rrea --features struname --data_dir "../data/en_fr_15k_V1"
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm sinkhorn --mode 1-to-1 --encoder rrea --features struname --data_dir "../data/en_de_15k_V1"


#algorithm hun

echo "only name:--algorithm hun --mode 1-to-1 --encoder rrea --features name --data_dir:DBP15K/D-Z,DBP15K/D-J,DBP15K/D-F,SRPRS/S-F,SRPRS/S-D:"
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm hun --mode 1-to-1 --encoder rrea --features name --data_dir "../data/zh_en"
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm hun --mode 1-to-1 --encoder rrea --features name --data_dir "../data/ja_en"
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm hun --mode 1-to-1 --encoder rrea --features name --data_dir "../data/fr_en"
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm hun --mode 1-to-1 --encoder rrea --features name --data_dir "../data/en_fr_15k_V1"
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm hun --mode 1-to-1 --encoder rrea --features name --data_dir "../data/en_de_15k_V1"


echo "name and structural:--algorithm hun --mode 1-to-1 --encoder rrea --features struname --data_dir:DBP15K/D-Z,DBP15K/D-J,DBP15K/D-F,SRPRS/S-F,SRPRS/S-D:"
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm hun --mode 1-to-1 --encoder rrea --features struname --data_dir "../data/zh_en"
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm hun --mode 1-to-1 --encoder rrea --features struname --data_dir "../data/ja_en"
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm hun --mode 1-to-1 --encoder rrea --features struname --data_dir "../data/fr_en"
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm hun --mode 1-to-1 --encoder rrea --features struname --data_dir "../data/en_fr_15k_V1"
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm hun --mode 1-to-1 --encoder rrea --features struname --data_dir "../data/en_de_15k_V1"


#algorithm sm
echo "only name:--algorithm sm --mode 1-to-1 --encoder rrea --features name --data_dir:DBP15K/D-Z,DBP15K/D-J,DBP15K/D-F,SRPRS/S-F,SRPRS/S-D:"
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm sm --mode 1-to-1 --encoder rrea --features name --data_dir "../data/zh_en"
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm sm --mode 1-to-1 --encoder rrea --features name --data_dir "../data/ja_en"
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm sm --mode 1-to-1 --encoder rrea --features name --data_dir "../data/fr_en"
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm sm --mode 1-to-1 --encoder rrea --features name --data_dir "../data/en_fr_15k_V1"
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm sm --mode 1-to-1 --encoder rrea --features name --data_dir "../data/en_de_15k_V1"


echo "name and structural:--algorithm sm --mode 1-to-1 --encoder rrea --features struname --data_dir:DBP15K/D-Z,DBP15K/D-J,DBP15K/D-F,SRPRS/S-F,SRPRS/S-D:"
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm sm --mode 1-to-1 --encoder rrea --features struname --data_dir "../data/zh_en"
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm sm --mode 1-to-1 --encoder rrea --features struname --data_dir "../data/ja_en"
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm sm --mode 1-to-1 --encoder rrea --features struname --data_dir "../data/fr_en"
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm sm --mode 1-to-1 --encoder rrea --features struname --data_dir "../data/en_fr_15k_V1"
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm sm --mode 1-to-1 --encoder rrea --features struname --data_dir "../data/en_de_15k_V1"



#algorithm rl
echo "only name:--algorithm rl --mode 1-to-1 --encoder rrea --features name --data_dir:DBP15K/D-Z,DBP15K/D-J,DBP15K/D-F,SRPRS/S-F,SRPRS/S-D:"
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm rl --mode 1-to-1 --encoder rrea --features name --data_dir "../data/zh_en"
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm rl --mode 1-to-1 --encoder rrea --features name --data_dir "../data/ja_en"
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm rl --mode 1-to-1 --encoder rrea --features name --data_dir "../data/fr_en"
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm rl --mode 1-to-1 --encoder rrea --features name --data_dir "../data/en_fr_15k_V1"
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm rl --mode 1-to-1 --encoder rrea --features name --data_dir "../data/en_de_15k_V1"


echo "name and structural:--algorithm rl --mode 1-to-1 --encoder rrea --features struname --data_dir:DBP15K/D-Z,DBP15K/D-J,DBP15K/D-F,SRPRS/S-F,SRPRS/S-D:"
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm rl --mode 1-to-1 --encoder rrea --features struname --data_dir "../data/zh_en"
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm rl --mode 1-to-1 --encoder rrea --features struname --data_dir "../data/ja_en"
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm rl --mode 1-to-1 --encoder rrea --features struname --data_dir "../data/fr_en"
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm rl --mode 1-to-1 --encoder rrea --features struname --data_dir "../data/en_fr_15k_V1"
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm rl --mode 1-to-1 --encoder rrea --features struname --data_dir "../data/en_de_15k_V1"
