for i in {0..49}
do
  echo "IT, Looping ... number $i"
  python train.py --config configs/maml/it10_4.yaml --output-folder results_it10_4 --seed "$i" --num-workers 8 
done
