#!/bin/bash

cd ./gae
python run.py -eval_each

cd ../gae_ssc
python run.py -eval_each

cd ../daegc
python run.py -eval_each

cd ../sdcn
python run.py -eval_each

cd ../dfcn
python run.py -eval_each

cd ../dcrn
python run.py -eval_each

cd ../agcdrr
python run.py -eval_each

cd ../dgcluster
python run.py -eval_each

cd ../hsan
python run.py -eval_each

cd ../ccgc
python run.py -eval_each

cd ../magi
python run.py -eval_each

cd ../ns4gc
python run.py -eval_each