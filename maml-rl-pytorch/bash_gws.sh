#!/bin/sh

for i in {20..49}
do
  echo "KY, Looping ... number $i"
  python train.py --config configs/maml/ky.yaml --output-folder results_ky --seed "$i" --num-workers 8 
done

for i in {20..49}
do
  echo "IT, Looping ... number $i"
  python train.py --config configs/maml/it10.yaml --output-folder results_it --seed "$i" --num-workers 8 
done

for i in {20..49}
do
  echo "GW10, Looping ... number $i"
  python train.py --config configs/maml/gw10Two1.yaml --output-folder results_gw10Two1 --seed "$i" --num-workers 8
done

for i in {20..49}
do
  echo "GW20, Looping ... number $i"
  python train.py --config configs/maml/gw20Three1.yaml --output-folder results_gw20Three1 --seed "$i" --num-workers 8
done
