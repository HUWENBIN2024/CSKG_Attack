# this script file needs putting under \CSKB_Population
# for large model: lr small, bs = 32
# for base model: lr large, bs = 64

python models/KGBertSageTrain.py \
--device cuda:6 \
--train_csv_path data/train.tsv \
--evaluation_file_path data/dev.tsv \
--test_file_path data/test.tsv \
--epochs 15 \
--batch_size 64 \
--eval_every 500 \
--save_best_model \
--ptlm bert-large-uncased \
--output_dir result_clean \
--lr 5e-5 \
--num_neighbour 4 \
--seed 3407 \
