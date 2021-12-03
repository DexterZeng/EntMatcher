# EntMatcher

## Dependencies

* Python=3.6
* Tensorflow-gpu=1.13.1
* Pytorch=1.4.0
* Scipy
* Numpy
* Scikit-learn

## Datasets

### Existing Datasets
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

### Non 1-to-1 Alignment Dataset
We also offer our constructed non 1-to-1 alignment dataset FB_DBP_MUL (shortened as mul), which adopts the same format. 

### Auxiliary Information
Regarding the Semantic Information, we obtain the entity name embeddings from EAE, which can also be found here. 

### Usage
Unzip the data.zip. For the usage of auxiliary information, obtain the name embedding files and place them under corresponding dataset directories.

## Running
* First generate input unified entity embeddings. 
```
cd models
python gcn.py –data_dir "zh_en"
python rrea.py –data_dir "zh_en"
```
The data_dir could be chosen from the directories of these datasets. 
* Then run 
```
cd ..
python infer.py --lan "fr_en"
```
* You may also directly run
```
bash auto.sh
```
> Due to the instability of embedding-based methods, it is acceptable that the results fluctuate a little bit  when running code repeatedly.

> If you have any questions about reproduction, please feel free to email to zengweixin13@nudt.edu.cn.
