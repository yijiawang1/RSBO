#!/bin/sh

for i in {20..49}
do
  echo "Looping ... number $i"
  python train.py --config configs/maml/mc.yaml --output-folder results_mc --seed "$i" --num-workers 8 
done
