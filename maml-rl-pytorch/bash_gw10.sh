for i in {6..49}
do
  echo "GW10, Looping ... number $i"
  python train.py --config configs/maml/gw10Two1_2.yaml --output-folder results_gw10Two1_2 --seed "$i" --num-workers 8
done
