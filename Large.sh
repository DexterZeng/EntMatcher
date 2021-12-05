#!/bin/sh

# Table 5
python ./infer.py --data_dir "dbp_wd_100" --encoder "gcn" --features "stru"  --sim "cosine" --scoreop "none" --match "greedy"
python ./infer.py --data_dir "dbp_yg_100" --encoder "gcn" --features "stru"  --sim "cosine" --scoreop "none" --match "greedy"

python ./infer.py --data_dir "dbp_wd_100" --encoder "gcn" --features "stru"  --sim "cosine" --scoreop "csls" --match "greedy"
python ./infer.py --data_dir "dbp_yg_100" --encoder "gcn" --features "stru"  --sim "cosine" --scoreop "csls" --match "greedy"

python ./infer.py --data_dir "dbp_wd_100" --encoder "gcn" --features "stru"  --sim "cosine" --scoreop "rinf" --match "greedy"
python ./infer.py --data_dir "dbp_yg_100" --encoder "gcn" --features "stru"  --sim "cosine" --scoreop "rinf" --match "greedy"

python ./infer.py --data_dir "dbp_wd_100" --encoder "gcn" --features "stru"  --sim "cosine" --scoreop "sinkhorn" --match "greedy"
python ./infer.py --data_dir "dbp_yg_100" --encoder "gcn" --features "stru"  --sim "cosine" --scoreop "sinkhorn" --match "greedy"

python ./infer.py --data_dir "dbp_wd_100" --encoder "gcn" --features "stru"  --sim "cosine" --scoreop "none" --match "hun"
python ./infer.py --data_dir "dbp_yg_100" --encoder "gcn" --features "stru"  --sim "cosine" --scoreop "none" --match "hun"

python ./infer.py --data_dir "dbp_wd_100" --encoder "gcn" --features "stru"  --sim "cosine" --scoreop "none" --match "rl"
python ./infer.py --data_dir "dbp_yg_100" --encoder "gcn" --features "stru"  --sim "cosine" --scoreop "none" --match "rl"