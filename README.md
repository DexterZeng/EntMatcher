# EntMatcher: An Open-source Library
[![language-python3](https://img.shields.io/badge/Language-Python3-blue.svg?style=flat-square)](https://www.python.org/)
[![made-with-Pytorch](https://img.shields.io/badge/Made%20with-pytorch-orange.svg?style=flat-square)](https://www.pytorch.org/)
[![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg?style=flat-square)](https://github.com/DexterZeng/EntMatcher/issues)

> Entity alignment (EA) identifies equivalent entities that locate in different knowledge graphs (KGs), and has attracted growing research interests over the last few years with the advancement of KG embedding techniques. Although a pile of embedding-based EA frameworks have been developed, they mainly focus on improving the performance of entity representation learning, while largely overlook the subsequent stage that matches KGs in entity embedding spaces. Nevertheless, accurately matching entities based on learned entity representations is crucial to the overall alignment performance, as it coordinates individual alignment decisions and determines the global matching result. Hence, it is essential to understand how well existing solutions for matching KGs in entity embedding spaces perform on present benchmarks, as well as their strengths and weaknesses. To this end, in this article we provide a comprehensive survey and evaluation of matching algorithms for KGs in entity embedding spaces in terms of effectiveness and efficiency on both classic settings and new scenarios that better mirror real-life challenges. Based on in-depth analysis, we provide useful insights into the design trade-offs and good paradigms of existing works, and suggest promising directions for future development. 

## Contents
- [EntMatcher: An Open-source Library](#entmatcher-an-open-source-library)
  - [Contents](#contents)
  - [Overview](#overview)
  - [Getting started](#getting-started)
    - [Code organization](#code-organization)
    - [Dependencies](#dependencies)
    - [Installation](#installation)
    - [Usage](#usage)
      - [1. Generate input unified entity embeddings](#1-generate-input-unified-entity-embeddings)
      - [2. Matching KGs in entity embedding spaces](#2-matching-kgs-in-entity-embedding-spaces)
      - [3. Example](#3-example)
  - [Datasets](#datasets)
    - [Existing Datasets statistics](#existing-datasets-statistics)
    - [Dataset Description](#dataset-description)
    - [Non 1-to-1 Alignment Dataset](#non-1-to-1-alignment-dataset)
    - [Usage](#usage-1)
  - [Experiments and Results](#experiments-and-results)
    - [Experiment Settings](#experiment-settings)
      - [Hardware configuration and hyper-parameter setting](#hardware-configuration-and-hyper-parameter-setting)
      - [Representation learning models](#representation-learning-models)
      - [Auxiliary information for alignment](#auxiliary-information-for-alignment)
      - [Similarity metric](#similarity-metric)
    - [Detailed Results](#detailed-results)

## Overview

<p>
  <img width="75%" src="https://github.com/DexterZeng/EntMatcher/blob/main/framework1.png" />
</p>

We use [Python](https://www.python.org/), [Pytorch](https://www.pytorch.org/) and [Tensorflow](https://www.tensorflow.org/) to develop an open-source library, namely **EntMatcher**.

The architecture of EntMatcher library is presented in the blue block of figure above, which takes as input unified entity embeddings and produces the matched entity pairs. 
It has the following three major features:

* **Loosely-coupled design**. There are three independent modules in EntMatcher, and we have implemented the representative methods in each module. Users are free to combine the techniques in each module to develop new approaches, or to implement their new designs by following the templates in modules. 

* **Reproduction of existing approaches**. We re-implement all existing embedding matching algorithms by using EntMatcher. 
For instance, the combination of cosine similarity, CSLS, and Greedy algorithm reproduces the CSLS algorithm; and the combination of cosine similarity, None, and Hungarian reproduces the Hungarian algorithm. 

* **Flexible integration with other modules in EA**. EntMatcher is highly flexible, which can be directly called during the development of standalone EA approaches. 
Besides, users may also use EntMatcher as the backbone and call other modules. 
For instance, to conduct the experimental evaluations, we implemented the representation learning and auxiliary information modules to generate the unified entity embeddings, as shown in the white blocks of figure above. 
Finally, EntMatcher is also compatible with existing open-source EA libraries (that mainly focus on representation learning) such as [OpenEA](https://github.com/nju-websoft/OpenEA) and [EAkit](https://github.com/THU-KEG/EAkit). 

Currently, EntMatcher Library (with additional modules) has integrated the following modules, and the approaches in modules can be combined arbitrarily:
* **Representation Learning Module**.
    1. **GCN**: [Cross-lingual Knowledge Graph Alignment via Graph Convolutional Networks](https://www.aclweb.org/anthology/D18-1032). EMNLP 2018.
    2. **RREA**: [Relational reflection entity alignment](https://arxiv.org/pdf/2008.07962.pdf). CIKM 2022.
    3. **...**(such as [OpenEA](https://github.com/nju-websoft/OpenEA))
* **EntMatcher Module**.
    1. **CSLS**: [Word translation without parallel data](https://arxiv.org/pdf/1710.04087.pdf). ICLR 2018.
    2. **RInf**: [On entity alignment at scale](https://dl.acm.org/doi/abs/10.1007/s00778-021-00703-3). VLDB J 2021.
    3. **Sinkhorn**: [Clusterea: Scalable entity alignment with stochastic training and normalized mini-batch similarities](https://arxiv.org/pdf/2205.10312.pdf). SIGKDD 2022.
    4. **DInf**: [Relation-aware entity alignment for heterogeneous knowledge graphs](https://arxiv.org/pdf/1908.08210.pdf). IJCAI 2019.
    5. **Greedy**: [ Deep graph matching consensus](https://arxiv.org/pdf/2001.09621.pdf). ICLR 2020.
    6. **SMat**: [Collective entity alignment via adaptive features](https://arxiv.org/pdf/1912.08404.pdf). ICDE 2020.
    7.  **Hun.**: [From alignment to assignment: Frustratingly simple unsupervised entity alignment](https://aclanthology.org/2021.emnlp-main.226.pdf). EMNLP 2021.
    8.  **RL**: [Reinforcement learning-based collective entity alignment with adaptive features](https://arxiv.org/pdf/2101.01353.pdf). ACM Trans. Inf. Syst. 2021.
* **Auxiliary Information Module**.
    1. **Name**.
    2. **Description**.
   
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
* Python>=3.7 (tested on Python=3.8.10)
* Tensorflow-gpu=2.x (tested on Tensorflow-gpu=2.6.0)
* Pytorch=1.7.1
* Scipy=1.7
* Keras=2.6.0
* Numpy
* Scikit-learn
* fml


### Installation
We recommend creating a new conda environment to install and run EntMatcher. 
```
conda create -n entmatcher python=3.8.10
conda activate entmatcher
conda install pytorch==1.x torchvision==0.x torchaudio==0.x cudatoolkit=xxx -c pytorch
conda install scipy
conda install tensorflow-gpu==2.6.0
conda install Keras==2.6.0
```

Then, EntMatcher can be installed using pip with the following steps:
```
git clone https://github.com/DexterZeng/EntMatcher.git EntMatcher
cd EntMatcher
pip install EntMatcher-0.1.tar.gz
```
### Usage

#### 1. Generate input unified entity embeddings
```
cd models
python gcn.py --data_dir "zh_en"
python rrea.py --data_dir "zh_en"
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

Or you can explore different modules, and design new strategies by following ```examples.py```
Main configurations:
* Similarity metric ```--sim``` can be chosen from ```cosine, euclidean, manhattan```;
* Score optimization ```--scoreop``` can be chosen from ```csls, sinkhorn, rinf none```;
* Matching constraint ```--match``` can be chosen from ```hun, sm, rl, greedy```;


#### 3. Example
The following is an example about how to use EntMatcher in Python (We assume that you have already downloaded our [datasets](#datasets) and know how to [use](#Usage) it)

First, you need to generate vectors from the EA model and save them to an npy file named after the model.
```
python rrea.py --data_dir "zh_en"
```
Then, you can use these vectors to select the appropriate algorithm for matching calculations.
```
import entmatcher as em

model = args.encoder
args = load_args("hyperparameter file folder")
kgs = read_kgs_from_folder("data folder")
dataset = em.extras.Datasets(args)
algorithms = em.algorithms.csls
se_vec = np.load(args.data_dir + '/' + args.encoder + '.npy')
name_vec = dataset.loadNe()
algorithms.match([se_vec, name_vec], dataset)
```
For a more convenient use, You can use the code we prepared and just adjust the parameters to run： 
```
python embed_matching.py --data_dir ../data/zh_en --encoder rrea --algorithm csls --features stru
```

## Datasets

### Existing Datasets statistics
We used popular EA benchmarks for evaluation: 
* **DBP15K**, which comprises three multilingual KG pairs extracted from DBpedia: English to Chinese (DZ), English to Japanese (D-J), and English to French (D-F).
* **SRPRS**, which is a sparser dataset that follows reallife entity distribution, including two multilingual KG pairs extracted from DBpedia: English to French (S-F) and English to German (S-D), and two mono-lingual KG pairs: DBpedia to Wikidata (S-W) and DBpedia to YAGO (S-Y).
* **DWY100K**, a larger dataset consisting of two monolingual KG pairs: DBpedia to Wikidata (D-W) and DBpedia to YAGO (D-Y). 

The detailed statistics can be found in Table, where the numbers of entities, relations, triples, gold links, and the average entity degree are reported.
<p>
  <img width="75%" src="https://github.com/DexterZeng/EntMatcher/blob/main/Dataset_statistics.png" />
</p>
The original datasets are obtained from [DBP15K dataset](https://github.com/nju-websoft/BootEA),  [GCN-Align](https://github.com/1049451037/GCN-Align) and [JAPE](https://github.com/nju-websoft/JAPE):

### Dataset Description
Regarding the gold alignment links, we adopted 70% as test set, 20% for training,and 10% for validation.The folder names of the datasets used by the code are as follows:
* **DBP15K/D-Z**: ```data/zh_en```
* **DBP15K/D-J**: ```data/ja_en```
* **DBP15K/D-F**: ```data/fr_en```
* **SRPRS/S-F**: ```data/en_fr_15k_V1```
* **SRPRS/S-D**: ```data/en_de_15k_V1```
* **SRPRS/S-W**: ```data/dbp_wd_15k_V1```
* **SRPRS/S-Y**: ```data/dbp_yg_15k_V1```
* **DWY100K/D-W**: ```data/dbp_wd_100```
* **DWY100K/D-Y**: ```data/dbp_yg_100```
* **FB_DBP_MUL**: ```data/mul```

Take the dataset **DBP15K (ZH-EN)** as an example, the folder ```data/zh_en``` contains:
* ent_ids_1: ids for entities in source KG (ZH);
* ent_ids_1_trans_goo: entities in source KG (ZH) with translated names (only for cross-lingual datasets);
* ent_ids_2: ids for entities in target KG (EN);
* ill_ent_ids: all labeled entity links;
* ref_ent_ids: entity links for testing;
* val_ent_ids: entity links for validation;
* sup_ent_ids: entity links for training;
* triples_1: relation triples encoded by ids in source KG (ZH);
* triples_2: relation triples encoded by ids in target KG (EN);

### Non 1-to-1 Alignment Dataset
We also offer our constructed non 1-to-1 alignment dataset FB_DBP_MUL (shortened as **mul**), which adopts the same format.

### Usage
Unzip the ```data.zip```. For the usage of auxiliary information, obtain the [name embeddings] and [structural embeddings] files and place them under corresponding dataset directories.


## Experiments and Results
To reproduce the experimental results in the paper, you can first download the unified [structural embeddings] and [the name embeddings]. 
Then put the files under the corresponding directories. 
### Experiment Settings
#### Hardware configuration and hyper-parameter setting
We followed the configurations presented in the original papers of these algorithms, and tuned the hyper-parameters on the validation set. 
* Specifically, for **CSLS**, we set k to 1, except on the non 1-to-1 setting where we set it to 5. 
* Similarly, regarding **RInf**, we changed the maximum operation in Equation 2 to top-k average operation on the non 1-to-1 setting, where k is set to 5. 
* As to **Sink**., we set l to 100. 
* For **RL**, we found the hyper-parameters in the original paper could already produce the best results, and directly adopted them. 
* The rest of the approaches, i.e., **DInf**, **Hun.**, and **SMat**, do not contain hyperparameters.

#### Representation learning models
Since representation learning is not the focus of this work, we adopted two frequently used models, i.e., **RREA**  and **GCN** . 
* Concretely, **GCN** is one of the simplest models, which uses graph convolutional networks to learn the structural embeddings.
* **RREA** is one of the best-performing solutions, which leverages relational reflection transformation to obtain relation-specific entity embeddings.

#### Auxiliary information for alignment
Although EA underlines the use of graph structure for alignment([An experimental study of state-of-the-art entity alignment approaches,IEE TKDE 2020](https://ieeexplore.ieee.org/document/9174835)), for a more comprehensive evaluation, we examined the influence of auxiliary information on the matching results by following previous works and using entity name embeddings to facilitate alignment. We also combined these two channels of information with equal weights to generate the fused similarity matrix for matching.

#### Similarity metric
After obtaining the unified entity representations E, a similarity metric is required to produce pairwise scores and generate the similarity matrix S. Frequent choices include the cosine similarity, the Euclidean distance and the Manhattan distance.In this work, we followed mainstream works and adopted the cosine similarity.
Next, you can run 
```
cd src
python embed_matching.py --algorithm dinf --mode 1-to-1 --encoder gcn --features stru --data_dir "../data/zh_en"
```
and varying the parameter settings.
### Detailed Results
> Due to the instability of embedding-based methods, it is acceptable that the results fluctuate a little bit when running code repeatedly.

> If you have any questions about reproduction, please feel free to email to zengweixin13@nudt.edu.cn.
