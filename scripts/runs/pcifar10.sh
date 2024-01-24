
# for run in {1..10}
# do 
python src/train.py experiment=Finetuning_2 trainer=gpu
# done

# for run in {1..10}
# do 
python src/train.py experiment=HAT_2 trainer=gpu
# done

# for run in {1..10}
# do 
python src/train.py experiment=AdaHAT_2 trainer=gpu
# done
