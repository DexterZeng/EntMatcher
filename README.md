# EntMatcher
[![language-python3](https://img.shields.io/badge/Language-Python3-blue.svg?style=flat-square)](https://www.python.org/)
[![made-with-Pytorch](https://img.shields.io/badge/Made%20with-pytorch-orange.svg?style=flat-square)](https://www.pytorch.org/)
[![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg?style=flat-square)](https://github.com/DexterZeng/EntMatcher/issues)

## Contents
1. [Overview](#overview)
2. [Getting started](#getting-started)
3. [Datasets](#datasets)
4. [Experiments and Results](#experiments-and-results)

## Overview

<p>
  <img width="50%" src="https://github.com/DexterZeng/EntMatcher/blob/main/framework.png" />
</p>

The architecture of EntMatcher library is presented in the blue block of figure above, which takes as input unified entity embeddings and produces the matched entity pairs. 
It has the following three major features:

* **Loosely-coupled design**. There are three independent modules in EntMatcher, and we have implemented the representative methods in each module. Users are free to combine the techniques in each module to develop new approaches, or to implement their new designs by following the templates in modules. 

* **Reproduction of existing approaches**. We re-implement all existing embedding matching algorithms by using EntMatcher. 
For instance, the combination of cosine similarity, CSLS, and Greedy algorithm reproduces the CSLS algorithm; and the combination of cosine similarity, None, and Hungarian reproduces the Hungarian algorithm. 

* **Flexible integration with other modules in EA**. EntMatcher is highly flexible, which can be directly called during the development of standalone EA approaches. 
Besides, users may also use EntMatcher as the backbone and call other modules. 
For instance, to conduct the experimental evaluations, we implemented the representation learning and auxiliary information modules to generate the unified entity embeddings, as shown in the white blocks of figure above. 
Finally, EntMatcher is also compatible with existing open-source EA libraries (that mainly focus on representation learning) such as [OpenEA](https://github.com/nju-websoft/OpenEA) and [EAkit](https://github.com/THU-KEG/EAkit). 

## Getting started

### Code organization
```
data/: datasets
models/: generating the input unified entity embeddings using existing representation learning methods
src/
|-- entmatcher/
|	|--algorithms/: package of the standalone algorithms
|	|--extras/: package of the extra modules
|	|--modules/: package of the main modules
|	|--embed_matching.py: implementaion of calling the standalone algorithms
|	|--example.py: implementaion of calling the modules
```

### Dependencies
* Python=3.6
* Tensorflow-gpu=1.13.1
* Pytorch=1.4.0
* Scipy
* Numpy
* Scikit-learn

### Installation
conda create -n entmatcher python=3.7
conda activate entmatcher
conda install pytorch==1.x torchvision==0.x torchaudio==0.x cudatoolkit=xxx -c pytorch
conda install scipy
conda install tensorflow-gpu=2.6.0

Then, EntMatcher can be installed using pip with the following steps:
```
git clone https://github.com/DexterZeng/EntMatcher.git EntMatcher
cd EntMatcher
pip install EntMatcher-0.1.tar.gz
```
### Usage
The following is an example about how to use EntMatcher in Python (We assume that you have already downloaded our datasets)

First, you need to generate vectors from the EA model and save them to an npy file named after the model. Like the follows:
```
python rrea.py --data_dir "zh_en"
```
Then, you can use these vectors to select the appropriate algorithm for matching calculations.
```
python embed_matching.py --data_dir ../data/zh_en --encoder rrea --algorithm csls
```
A detail example can be seen as follows:
```
import entmatcher as em

model = rrea
args = load_args("hyperparameter file folder")
kgs = read_kgs_from_folder("data folder")
dataset = em.extras.Datasets(args)
algorithms = em.algorithms.csls
se_vec = np.load(args.data_dir + '/' + args.encoder + '.npy')
name_vec = dataset.loadNe()
algorithms.match([se_vec, name_vec], dataset)
```
#### 1. Generate input unified entity embeddings
```
cd models
python gcn.py –data_dir "zh_en"
python rrea.py –data_dir "zh_en"
```
The data_dir could be chosen from the directories of these datasets. Or you can directly run:
```
bash stru.sh
```
As for the auxiliary information, we obtain the entity name embeddings from EAE, which can also be found here. 
#### 2. Matching KGs in entity embedding spaces
To call different algorithms, you can run
```
cd src
python embed_matching.py
```
where you can set ```--algorithm``` to ```dinf, csls, rinf, sinkhorn, hun, sm, rl```

Other configurations:
```--mode``` can be chosen from ```1-to-1, mul, unm```; ```--encoder``` can be chosen from ```gcn, rrea```; ```--features``` can be chosen from ```stru, name, struname```; ```--data_dir``` can be chosen from the dataset directories.

Or you can explore different modules, and design new strategies by following ```exanples.py```
Main configurations:
* Similarity metric ```--sim``` can be chosen from ```cosine, euclidean, manhattan```;
* Score optimization ```--scoreop``` can be chosen from ```csls, sinkhorn, rinf none```;
* Matching constraint ```--match``` can be chosen from ```hun, sm, rl, greedy```;


## Datasets

#### Existing Datasets
The original datasets are obtained from [DBP15K dataset](https://github.com/nju-websoft/BootEA),  [GCN-Align](https://github.com/1049451037/GCN-Align) and [JAPE](https://github.com/nju-websoft/JAPE).

Take the dataset DBP15K (ZH-EN) as an example, the folder "zh_en" contains:
* ent_ids_1: ids for entities in source KG (ZH);
* ent_ids_1_trans_goo: entities in source KG (ZH) with translated names (only for cross-lingual datasets);
* ent_ids_2: ids for entities in target KG (EN);
* ill_ent_ids: all labeled entity links;
* ref_ent_ids: entity links for testing;
* val_ent_ids: entity links for validation;
* sup_ent_ids: entity links for training;
* triples_1: relation triples encoded by ids in source KG (ZH);
* triples_2: relation triples encoded by ids in target KG (EN);

#### Non 1-to-1 Alignment Dataset
We also offer our constructed non 1-to-1 alignment dataset FB_DBP_MUL (shortened as mul), which adopts the same format.

#### Usage
Unzip the data.zip. For the usage of auxiliary information, obtain the name embedding files and place them under corresponding dataset directories.


## Experiments and Results
To reproduce the experimental results in the paper, you can first download the unified structural embeddings and the name embeddings. 
Then put the files under the corresponding directories. 

Next, you can run 
```
cd src
python embed_matching.py --algorithm dinf --mode 1-to-1 --encoder gcn --features stru --data_dir "../data/zh_en"
```
and varying the parameter settings.

> Due to the instability of embedding-based methods, it is acceptable that the results fluctuate a little bit when running code repeatedly.

> If you have any questions about reproduction, please feel free to email to zengweixin13@nudt.edu.cn.
