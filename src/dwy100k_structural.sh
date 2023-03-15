#!/bin/sh

#algorithm dinf
echo "--algorithm dinf --mode 1-to-1 --encoder gcn --features stru --data_dir:DWY100K/D-W,DWY100K/D-Y"
python embed_matching.py --algorithm dinf --mode 1-to-1 --encoder gcn --features stru --data_dir "../data/dbp_wd_100"
python embed_matching.py --algorithm dinf --mode 1-to-1 --encoder gcn --features stru --data_dir "../data/dbp_yg_100"



#algorithm csls

echo "--algorithm csls --mode 1-to-1 --encoder gcn --features stru --data_dir:DWY100K/D-W,DWY100K/D-Y"
python embed_matching.py --algorithm csls --mode 1-to-1 --encoder gcn --features stru --data_dir "../data/dbp_wd_100"
python embed_matching.py --algorithm csls --mode 1-to-1 --encoder gcn --features stru --data_dir "../data/dbp_yg_100"



#algorithm rinf

echo "--algorithm dinf --mode 1-to-1 --encoder gcn --features stru --data_dir:DWY100K/D-W,DWY100K/D-Y"
python embed_matching.py --algorithm dinf --mode 1-to-1 --encoder gcn --features stru --data_dir "../data/dbp_wd_100"
python embed_matching.py --algorithm dinf --mode 1-to-1 --encoder gcn --features stru --data_dir "../data/dbp_yg_100"



#algorithm sinkhorn
#
echo "--algorithm dinf --mode 1-to-1 --encoder gcn --features stru --data_dir:DWY100K/D-W,DWY100K/D-Y"
python embed_matching.py --algorithm dinf --mode 1-to-1 --encoder gcn --features stru --data_dir "../data/dbp_wd_100"
python embed_matching.py --algorithm dinf --mode 1-to-1 --encoder gcn --features stru --data_dir "../data/dbp_yg_100"



#algorithm hun

echo "--algorithm dinf --mode 1-to-1 --encoder gcn --features stru --data_dir:DWY100K/D-W,DWY100K/D-Y"
python embed_matching.py --algorithm dinf --mode 1-to-1 --encoder gcn --features stru --data_dir "../data/dbp_wd_100"
python embed_matching.py --algorithm dinf --mode 1-to-1 --encoder gcn --features stru --data_dir "../data/dbp_yg_100"


#algorithm sm
echo "--algorithm dinf --mode 1-to-1 --encoder gcn --features stru --data_dir:DWY100K/D-W,DWY100K/D-Y"
python embed_matching.py --algorithm dinf --mode 1-to-1 --encoder gcn --features stru --data_dir "../data/dbp_wd_100"
python embed_matching.py --algorithm dinf --mode 1-to-1 --encoder gcn --features stru --data_dir "../data/dbp_yg_100"



#algorithm rl
echo "--algorithm dinf --mode 1-to-1 --encoder gcn --features stru --data_dir:DWY100K/D-W,DWY100K/D-Y"
python embed_matching.py --algorithm dinf --mode 1-to-1 --encoder gcn --features stru --data_dir "../data/dbp_wd_100"
python embed_matching.py --algorithm dinf --mode 1-to-1 --encoder gcn --features stru --data_dir "../data/dbp_yg_100"
