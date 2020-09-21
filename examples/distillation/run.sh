export CACHE_DIR=/pre-trained-transformers
export BASE_DIR=$1
export FILENAME=$2
export CUDA_VISIBLE_DEVICES=$3

python scripts/binarized_data.py \
    --file_path $BASE_DIR/$FILENAME.txt \
    --tokenizer_type bert \
    --tokenizer_name bert-base-uncased \
    --cache_dir $CACHE_DIR \
    --dump_file $BASE_DIR/binarized_$FILENAME

python scripts/token_counts.py \
    --data_file $BASE_DIR/binarized_$FILENAME.bert-base-uncased.pickle \
    --token_counts_dump $BASE_DIR/token_counts.bert-base-uncased.pickle \
    --vocab_size 30522

python train.py \
    --student_type distilbert \
    --student_config training_configs/distilbert-base-uncased.json \
    --teacher_type bert \
    --teacher_name bert-base-uncased \
    --cache_dir $CACHE_DIR \
    --alpha_ce 5.0 --alpha_mlm 2.0 --alpha_cos 1.0 --alpha_clm 0.0 --mlm \
    --freeze_pos_embs \
    --dump_path /logs/distilbert/$FILENAME \
    --data_file $BASE_DIR/binarized_$FILENAME.bert-base-uncased.pickle \
    --token_counts $BASE_DIR/token_counts.bert-base-uncased.pickle \
    --force
