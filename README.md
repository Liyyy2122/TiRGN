# TiRGN

This is the code of TiRGN.


### Installation
```
conda create -n tirgn python=3.7

conda activate tirgn

pip install -r requirement.txt
```



## How to run

#### Process data

For all the datasets, the following command can be used to get the history of their entities and relations.
```
cd src
python get_history.py --dataset ICEWS14
```



#### Train models

Then the following commands can be used to train TiRGN.

Train models

```
python main.py -d ICEWS14 --history-rate 0.3 --train-history-len 9 --test-history-len 9 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction --relation-prediction --add-static-graph --angle 14 --discount 1 --task-weight 0.7 --gpu 0 --save checkpoint
```



#### Evaluate models

The following commands can be used to evaluate TiRGN (add `--test` only).

###### Test with ground truth history:

```
python main.py -d ICEWS14 --history-rate 0.3 --train-history-len 9 --test-history-len 9 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction --relation-prediction --add-static-graph --angle 14 --discount 1 --task-weight 0.7 --gpu 0 --save checkpoint --test 
```

###### Test without ground truth history:

```
python main.py -d ICEWS14 --history-rate 0.3 --train-history-len 9 --test-history-len 9 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction --relation-prediction --add-static-graph --angle 14 --discount 1 --task-weight 0.7 --gpu 0 --save checkpoint --test --multi-step --topk 0
```



#### Detailed hyperparameters

The following commands and trained models ([google drive](https://drive.google.com/drive/folders/1-nQXpdofg-SXSsBOUMv6DS4XtDXGGl8X?usp=sharing)) can be used to get the entity prediction results reported in the paper (remove `--test` to train new models).

###### ICEWS14

~~~
python main.py -d ICEWS14 --history-rate 0.3 --train-history-len 9 --test-history-len 9 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction --relation-prediction --add-static-graph --angle 14 --discount 1 --task-weight 0.7 --gpu 0 --save checkpoint --test 
~~~

###### ICEWS14s

~~~
python main.py -d ICEWS14s --history-rate 0.3 --train-history-len 9 --test-history-len 9 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction --relation-prediction --add-static-graph --angle 14 --discount 1 --task-weight 0.7 --gpu 0 --save checkpoint --test
~~~

###### ICEWS18

~~~
python main.py -d ICEWS18 --history-rate 0.3 --train-history-len 10 --test-history-len 10 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction --relation-prediction --add-static-graph --angle 10 --discount 1 --task-weight 0.7 --gpu 0 --save checkpoint --test
~~~

###### ICEWS05-15

~~~
python main.py -d ICEWS05-15 --history-rate 0.3 --train-history-len 15 --test-history-len 15 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction --relation-prediction --add-static-graph --angle 10 --discount 1 --task-weight 0.7 --gpu 0 --save checkpoint --test
~~~

###### WIKI

~~~
python main.py -d WIKI --history-rate 0.3 --train-history-len 2 --test-history-len 2 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction --relation-prediction --angle 10 --discount 1 --task-weight 0.7 --gpu 0 --save checkpoint --test
~~~

###### YAGO

~~~
python main.py -d YAGO --history-rate 0.3 --train-history-len 1 --test-history-len 1 --dilate-len 1 --lr 0.001 --n-layers 1 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction --relation-prediction --angle 10 --discount 1 --task-weight 0.7 --gpu 0 --save checkpoint --test
~~~

###### GDELT

~~~
python main.py -d GDELT --history-rate 0.3 --train-history-len 7 --test-history-len 7 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction --relation-prediction --angle 10 --discount 1 --task-weight 0.7 --gpu 0 --save checkpoint --test
~~~



### Cite

Please cite as:
~~~
@inproceedings{DBLP:conf/ijcai/LiS022,
  author    = {Yujia Li and Shiliang Sun and Jing Zhao},
  title     = {TiRGN: Time-Guided Recurrent Graph Network with Local-Global Historical Patterns for Temporal Knowledge Graph Reasoning},
  booktitle = {Proceedings of the Thirty-First International Joint Conference on Artificial Intelligence, {IJCAI} 2022, Vienna, Austria, 23-29 July 2022},
  pages     = {2152--2158},
  year      = {2022},
}
~~~
