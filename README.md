# BERT-ProtoNet
> This reposity implements ProtoNet for Few-shot Relation Extraction aided with a BERT encoder.
> Data comes from [FewRel](https://github.com/thunlp/FewRel)
## Requirements
Install python packages in `requirements.txt`.
## Usage
Simply train or test a model with `python main.py --train/test`. Other parameters can be specified:
```bash
$ python main.py -h
usage: main.py [-h] [--dataset DATASET] [--trainN TRAINN] [--N N] [--K K]
               [--Q Q] [--batch_size BATCH_SIZE] [--train_iter TRAIN_ITER]
               [--val_iter VAL_ITER] [--test_iter TEST_ITER]
               [--eval_every EVAL_EVERY] [--max_length MAX_LENGTH] [--lr LR]
               [--weight_decay WEIGHT_DECAY] [--dropout DROPOUT]
               [--na_rate NA_RATE] [--grad_iter GRAD_ITER]
               [--hidden_size HIDDEN_SIZE] [--load_model LOAD_MODEL] [--train]
               [--test] [--dot]

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     name of dataset
  --trainN TRAINN       N in train
  --N N                 N way
  --K K                 K shot
  --Q Q                 Num of query per class
  --batch_size BATCH_SIZE
                        batch size
  --train_iter TRAIN_ITER
                        num of iters in training
  --val_iter VAL_ITER   num of iters in validation
  --test_iter TEST_ITER
                        num of iters in testing
  --eval_every EVAL_EVERY
                        evaluate after training how many iters
  --max_length MAX_LENGTH
                        max length
  --lr LR               learning rate
  --weight_decay WEIGHT_DECAY
                        weight decay
  --dropout DROPOUT     dropout rate
  --na_rate NA_RATE     NA rate (NA = Q * na_rate)
  --grad_iter GRAD_ITER
                        accumulate gradient every x iterations
  --hidden_size HIDDEN_SIZE
                        hidden size
  --load_model LOAD_MODEL
                        where to load trained model
  --train               whether do training
  --test                whether do testing
  --dot                 use dot instead of L2 distance for proto
```
## Performance
Currently only available for `wiki` dataset, acc=... (==TODO==)
Trained model can be downloaded [here]()(==TODO==).
