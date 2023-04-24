#!/bin/sh

for i in {0..49}
do
  echo "Looping ... number $i"
  python train.py --config configs/maml/ky_2.yaml --output-folder results_ky10One_2 --seed "$i" --num-workers 8 
done
