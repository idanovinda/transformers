export CUDA_VISIBLE_DEVICES=$1

BASE_DIR="/datasets/huggingface/wikipedia/wikipedia_100"
FILENAME="wikipedia_100.txt"
BINARIZED_FILENAME="binarized_wikipedia_100"

python scripts/binarized_data.py \
    --file_path $BASE_DIR/$FILENAME \
    --tokenizer_type bert \
    --tokenizer_name bert-base-uncased \
    --dump_file $BASE_DIR/$BINARIZED_FILENAME

python scripts/token_counts.py \
    --data_file $BASE_DIR/$BINARIZED_FILENAME.bert-base-uncased.pickle \
    --token_counts_dump $BASE_DIR/token_counts.bert-base-uncased.pickle \
    --vocab_size 30522

python train.py \
    --student_type distilbert \
    --student_config training_configs/distilbert-base-uncased.json \
    --teacher_type bert \
    --teacher_name bert-base-uncased \
    --alpha_ce 5.0 --alpha_mlm 2.0 --alpha_cos 1.0 --alpha_clm 0.0 --mlm \
    --freeze_pos_embs \
    --dump_path /logs/distilbert/wikipedia_100 \
    --data_file $BASE_DIR/$BINARIZED_FILENAME.bert-base-uncased.pickle \
    --token_counts $BASE_DIR/token_counts.bert-base-uncased.pickle \
    --force
