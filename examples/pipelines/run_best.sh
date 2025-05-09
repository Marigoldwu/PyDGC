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

cd ./gae
python run_robust.py
python run_robust.py -eval_each

cd ../gae_ssc
python run_robust.py
python run_robust.py -eval_each

cd ../daegc
python run_robust.py
python run_robust.py -eval_each

cd ../sdcn
python run_robust.py
python run_robust.py -eval_each

cd ../dfcn
python run_robust.py
python run_robust.py -eval_each

cd ../dcrn
python run_robust.py
python run_robust.py -eval_each

cd ../agcdrr
python run_robust.py
python run_robust.py -eval_each

cd ../dgcluster
python run_robust.py
python run_robust.py -eval_each

cd ../hsan
python run_robust.py
python run_robust.py -eval_each

cd ../ccgc
python run_robust.py
python run_robust.py -eval_each

cd ../magi
python run_robust.py
python run_robust.py -eval_each