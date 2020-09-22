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

export NODE_RANK=0
export N_NODES=1

export N_GPU_NODE=$4
export WORLD_SIZE=$4

pkill -f 'python -u train.py'

python -m torch.distributed.launch \
    --nproc_per_node=$N_GPU_NODE \
    --nnodes=$N_NODES \
    --node_rank $NODE_RANK \
    train.py \
        --force \
        --gpus $WORLD_SIZE \
        --student_type distilbert \
        --student_config training_configs/distilbert-base-uncased.json \
        --teacher_type bert \
        --cache_dir $CACHE_DIR \
        --teacher_name bert-base-uncased \
        --alpha_ce 0.33 --alpha_mlm 0.33 --alpha_cos 0.33 --alpha_clm 0.0 --mlm \
        --freeze_pos_embs \
        --dump_path /logs/distilbert/$FILENAME \
        --data_file $BASE_DIR/binarized_$FILENAME.bert-base-uncased.pickle \
        --token_counts $BASE_DIR/token_counts.bert-base-uncased.pickle