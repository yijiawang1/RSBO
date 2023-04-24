for i in {50..69}
do
  echo "GW20, Looping ... number $i"
  python train.py --config configs/maml/gw20Three1_2.yaml --output-folder results_gw20Three1_2 --seed "$i" --num-workers 8
done
