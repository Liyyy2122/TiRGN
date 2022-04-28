## quick Start


### Environment variables & dependencies
```
conda create -n tirgn python=3.7

conda activate tirgn

pip install -r requirement.txt
```


### Process data
For all the datasets, get the history of their entities and relations.
```
cd src
python get_history.py --dataset ICEWS14
```


### Train models
Then the following commands can be used to train the proposed models. By default, dev set evaluation results will be printed when training terminates.

Train models
```
python main.py -d ICEWS14 --history-rate 0.3 --train-history-len 9 --test-history-len 9 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --gpu=0 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --weight 0.5  --entity-prediction --relation-prediction --add-static-graph --angle 14 --discount 1 --task-weight 0.7 --gpu 0 --save one
```


### Evaluate models
To generate the evaluation results of a pre-trained model, simply add the `--test` flag in the commands above. 

For example, the following command performs single-step inference and prints the evaluation results (with ground truth history).
```
python main.py -d ICEWS14 --history-rate 0.3 --train-history-len 9 --test-history-len 9 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --gpu=0 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --weight 0.5  --entity-prediction --relation-prediction --add-static-graph --angle 14 --discount 1 --task-weight 0.7 --gpu 0 --test --save one
```

The following command performs multi-step inference and prints the evaluation results (without ground truth history).
```
python main.py -d ICEWS14 --history-rate 0.3 --train-history-len 9 --test-history-len 9 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --gpu=0 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --weight 0.5  --entity-prediction --relation-prediction --add-static-graph --angle 14 --discount 1 --task-weight 0.7 --gpu 0 --test  --save one --multi-step --topk 0
```

Train:
ICEWS14
python main.py -d ICEWS14 --history-rate 0.3 --train-history-len 9 --test-history-len 9 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --gpu=0 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --weight 0.5  --entity-prediction --relation-prediction --add-static-graph --angle 14 --discount 1 --task-weight 0.7 --gpu 0 --save name

ICEWS14s
python main.py -d ICEWS14s --history-rate 0.3 --train-history-len 9 --test-history-len 9 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --gpu=0 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --weight 0.5  --entity-prediction --relation-prediction --add-static-graph --angle 14 --discount 1 --task-weight 0.7 --gpu 0 --save name

ICEWS18
python main.py -d ICEWS18 --history-rate 0.3 --train-history-len 10 --test-history-len 10 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --gpu=0 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --weight 0.5  --entity-prediction --relation-prediction --add-static-graph --angle 10 --discount 1 --task-weight 0.7 --gpu 0 --save name

ICEWS05-15
python main.py -d ICEWS05-15 --history-rate 0.3 --train-history-len 15 --test-history-len 15 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --gpu=0 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --weight 0.5  --entity-prediction --relation-prediction --add-static-graph --angle 10 --discount 1 --task-weight 0.7 --gpu 0 --save name

WIKI
python main.py -d WIKI --history-rate 0.3 --train-history-len 2 --test-history-len 2 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --gpu=0 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --weight 0.5  --entity-prediction --relation-prediction --angle 10 --discount 1 --task-weight 0.7 --gpu 0 --save name

YAGO
python main.py -d YAGO --history-rate 0.3 --train-history-len 1 --test-history-len 1 --dilate-len 1 --lr 0.001 --n-layers 1 --evaluate-every 1 --gpu=0 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --weight 0.5  --entity-prediction --relation-prediction --angle 10 --discount 1 --task-weight 0.7 --gpu 0 --save name

GDELT
python main.py -d GDELT --history-rate 0.3 --train-history-len 7 --test-history-len 7 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --gpu=0 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --weight 0.5  --entity-prediction --relation-prediction --angle 10 --discount 1 --task-weight 0.7 --gpu 0 --save name




Test:
ICEWS14
python main.py -d ICEWS14 --history-rate 0.3 --train-history-len 9 --test-history-len 9 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --gpu=0 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction --relation-prediction --add-static-graph --angle 14 --discount 1 --task-weight 0.7 --gpu 0 --test --save checkpoint

ICEWS14s
python main.py -d ICEWS14s --history-rate 0.3 --train-history-len 9 --test-history-len 9 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --gpu=0 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction --relation-prediction --add-static-graph --angle 14 --discount 1 --task-weight 0.7 --gpu 0 --test --save checkpoint

ICEWS18
python main.py -d ICEWS18 --history-rate 0.3 --train-history-len 10 --test-history-len 10 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --gpu=0 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction --relation-prediction --add-static-graph --angle 10 --discount 1 --task-weight 0.7 --gpu 0 --test --save checkpoint

ICEWS05-15
python main.py -d ICEWS05-15 --history-rate 0.3 --train-history-len 15 --test-history-len 15 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --gpu=0 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction --relation-prediction --add-static-graph --angle 10 --discount 1 --task-weight 0.7 --gpu 0 --test --save checkpoint

WIKI
python main.py -d WIKI --history-rate 0.3 --train-history-len 2 --test-history-len 2 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --gpu=0 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction --relation-prediction --angle 10 --discount 1 --task-weight 0.7 --gpu 0 --test --save checkpoint

YAGO
python main.py -d YAGO --history-rate 0.3 --train-history-len 1 --test-history-len 1 --dilate-len 1 --lr 0.001 --n-layers 1 --evaluate-every 1 --gpu=0 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction --relation-prediction --angle 10 --discount 1 --task-weight 0.7 --gpu 0 --test --save checkpoint

GDELT
python main.py -d GDELT --history-rate 0.3 --train-history-len 7 --test-history-len 7 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --gpu=0 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction --relation-prediction --angle 10 --discount 1 --task-weight 0.7 --gpu 0 --test --save checkpoint
