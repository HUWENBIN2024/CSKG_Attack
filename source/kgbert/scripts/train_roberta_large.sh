# this script file needs putting under \CSKB_Population
# for large model: lr = 5e-5
# for base model: lr = 1e-4

python models/train_kgbert_baseline.py \
  --device cuda:0 \
  --batch_size 32 \
  --train_csv_path cskg_backdoor_data/train.tsv \
  --evaluation_file_path cskg_backdoor_data/dev.tsv \
  --test_file_path cskg_backdoor_data/test.tsv \
  --epochs 50 \
  --eval_every 300 \
  --save_best_model \
  --ptlm roberta-large \
  --output_dir result_clean \
  --lr 5e-5