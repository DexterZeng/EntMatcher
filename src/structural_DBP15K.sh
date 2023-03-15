#!/bin/sh
#done
#algorithm dinf
echo "--algorithm dinf --mode unm --encoder gcn --features stru --data_dir "DBP15K+-D-Z,D-J,D-F:""
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm dinf --mode unm --encoder gcn --features stru --data_dir "../data/zh_en"
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm dinf --mode unm --encoder gcn --features stru --data_dir "../data/ja_en"
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm dinf --mode unm --encoder gcn --features stru --data_dir "../data/fr_en"

echo "--algorithm dinf --mode unm --encoder rrea --features stru --data_dir "DBP15K+-D-Z,D-J,D-F:""
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm dinf --mode unm --encoder rrea --features stru --data_dir "../data/zh_en"
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm dinf --mode unm --encoder rrea --features stru --data_dir "../data/ja_en"
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm dinf --mode unm --encoder rrea --features stru --data_dir "../data/fr_en"




#algorithm csls

echo "--algorithm csls --mode unm --encoder gcn --features stru --data_dir "DBP15K+-D-Z,D-J,D-F:""
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm csls --mode unm --encoder gcn --features stru --data_dir "../data/zh_en"
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm csls --mode unm --encoder gcn --features stru --data_dir "../data/ja_en"
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm csls --mode unm --encoder gcn --features stru --data_dir "../data/fr_en"

echo "--algorithm csls --mode unm --encoder rrea --features stru --data_dir "DBP15K+-D-Z,D-J,D-F:""
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm csls --mode unm --encoder rrea --features stru --data_dir "../data/zh_en"
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm csls --mode unm --encoder rrea --features stru --data_dir "../data/ja_en"
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm csls --mode unm --encoder rrea --features stru --data_dir "../data/fr_en"





#algorithm rinf


echo "--algorithm rinf --mode unm --encoder gcn --features stru --data_dir "DBP15K+-D-Z,D-J,D-F:""
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm rinf --mode unm --encoder gcn --features stru --data_dir "../data/zh_en"
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm rinf --mode unm --encoder gcn --features stru --data_dir "../data/ja_en"
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm rinf --mode unm --encoder gcn --features stru --data_dir "../data/fr_en"

echo "--algorithm rinf --mode unm --encoder rrea --features stru --data_dir "DBP15K+-D-Z,D-J,D-F:""
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm rinf --mode unm --encoder rrea --features stru --data_dir "../data/zh_en"
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm rinf --mode unm --encoder rrea --features stru --data_dir "../data/ja_en"
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm rinf --mode unm --encoder rrea --features stru --data_dir "../data/fr_en"



#algorithm sinkhorn

echo "--algorithm sinkhorn --mode unm --encoder gcn --features stru --data_dir "DBP15K+-D-Z,D-J,D-F:""
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm sinkhorn --mode unm --encoder gcn --features stru --data_dir "../data/zh_en"
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm sinkhorn --mode unm --encoder gcn --features stru --data_dir "../data/ja_en"
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm sinkhorn --mode unm --encoder gcn --features stru --data_dir "../data/fr_en"

echo "--algorithm sinkhorn --mode unm --encoder rrea --features stru --data_dir "DBP15K+-D-Z,D-J,D-F:""
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm sinkhorn --mode unm --encoder rrea --features stru --data_dir "../data/zh_en"
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm sinkhorn --mode unm --encoder rrea --features stru --data_dir "../data/ja_en"
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm sinkhorn --mode unm --encoder rrea --features stru --data_dir "../data/fr_en"





#algorithm hun
echo "--algorithm hun --mode unm --encoder gcn --features stru --data_dir "DBP15K+-D-Z,D-J,D-F:""
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm hun --mode unm --encoder gcn --features stru --data_dir "../data/zh_en"
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm hun --mode unm --encoder gcn --features stru --data_dir "../data/ja_en"
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm hun --mode unm --encoder gcn --features stru --data_dir "../data/fr_en"

echo "--algorithm hun --mode unm --encoder rrea --features stru --data_dir "DBP15K+-D-Z,D-J,D-F:""
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm hun --mode unm --encoder rrea --features stru --data_dir "../data/zh_en"
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm hun --mode unm --encoder rrea --features stru --data_dir "../data/ja_en"
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm hun --mode unm --encoder rrea --features stru --data_dir "../data/fr_en"




#algorithm sm
echo "--algorithm sm --mode unm --encoder gcn --features stru --data_dir "DBP15K+-D-Z,D-J,D-F:""
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm sm --mode unm --encoder gcn --features stru --data_dir "../data/zh_en"
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm sm --mode unm --encoder gcn --features stru --data_dir "../data/ja_en"
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm sm --mode unm --encoder gcn --features stru --data_dir "../data/fr_en"

echo "--algorithm sm --mode unm --encoder rrea --features stru --data_dir "DBP15K+-D-Z,D-J,D-F:""
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm sm --mode unm --encoder rrea --features stru --data_dir "../data/zh_en"
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm sm --mode unm --encoder rrea --features stru --data_dir "../data/ja_en"
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm sm --mode unm --encoder rrea --features stru --data_dir "../data/fr_en"



#algorithm rl
echo "--algorithm rl --mode unm --encoder gcn --features stru --data_dir "DBP15K+-D-Z,D-J,D-F:""
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm rl --mode unm --encoder gcn --features stru --data_dir "../data/zh_en"
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm rl --mode unm --encoder gcn --features stru --data_dir "../data/ja_en"
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm rl --mode unm --encoder gcn --features stru --data_dir "../data/fr_en"

echo "--algorithm rl --mode unm --encoder rrea --features stru --data_dir "DBP15K+-D-Z,D-J,D-F:""
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm rl --mode unm --encoder rrea --features stru --data_dir "../data/zh_en"
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm rl --mode unm --encoder rrea --features stru --data_dir "../data/ja_en"
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm rl --mode unm --encoder rrea --features stru --data_dir "../data/fr_en"
