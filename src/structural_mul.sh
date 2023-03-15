#!/bin/sh
#done
#algorithm dinf
echo "--algorithm dinf --mode mul --encoder gcn --features stru --data_dir "../data/mul:""
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm dinf --mode mul --encoder gcn --features stru --data_dir "../data/mul"


echo "--algorithm dinf --mode mul --encoder rrea --features stru --data_dir "../data/mul:""
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm dinf --mode mul --encoder rrea --features stru --data_dir "../data/mul"





#algorithm csls

echo "--algorithm csls --mode mul --encoder gcn --features stru --data_dir "../data/mul:""
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm csls --mode mul --encoder gcn --features stru --data_dir "../data/mul"

echo "--algorithm csls --mode mul --encoder rrea --features stru --data_dir "../data/mul:""
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm csls --mode mul --encoder rrea --features stru --data_dir "../data/mul"





#algorithm rinf


echo "--algorithm rinf --mode mul --encoder gcn --features stru --data_dir "../data/mul:""
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm rinf --mode mul --encoder gcn --features stru --data_dir "../data/mul"

echo "--algorithm rinf --mode mul --encoder rrea --features stru --data_dir "../data/mul:""
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm rinf --mode mul --encoder rrea --features stru --data_dir "../data/mul"




#algorithm sinkhorn

echo "--algorithm sinkhorn --mode mul --encoder gcn --features stru --data_dir "../data/mul:""
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm sinkhorn --mode mul --encoder gcn --features stru --data_dir "../data/mul"

echo "--algorithm sinkhorn --mode mul --encoder rrea --features stru --data_dir "../data/mul:""
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm sinkhorn --mode mul --encoder rrea --features stru --data_dir "../data/mul"





#algorithm hun
echo "--algorithm hun --mode mul --encoder gcn --features stru --data_dir "../data/mul:""
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm hun --mode mul --encoder gcn --features stru --data_dir "../data/mul"


echo "--algorithm hun --mode mul --encoder rrea --features stru --data_dir "../data/mul:""
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm hun --mode mul --encoder rrea --features stru --data_dir "../data/mul"




#algorithm sm
echo "--algorithm sm --mode mul --encoder gcn --features stru --data_dir "../data/mul:""
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm sm --mode mul --encoder gcn --features stru --data_dir "../data/mul"


echo "--algorithm sm --mode mul --encoder rrea --features stru --data_dir "../data/mul:""
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm sm --mode mul --encoder rrea --features stru --data_dir "../data/mul"



#algorithm rl
echo "--algorithm rl --mode mul --encoder gcn --features stru --data_dir "../data/mul:""
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm rl --mode mul --encoder gcn --features stru --data_dir "../data/mul"

echo "--algorithm rl --mode mul --encoder rrea --features stru --data_dir "../data/mul:""
CUDA_VISIBLE_DEVICES=0 python embed_matching.py --algorithm rl --mode mul --encoder rrea --features stru --data_dir "../data/mul"


