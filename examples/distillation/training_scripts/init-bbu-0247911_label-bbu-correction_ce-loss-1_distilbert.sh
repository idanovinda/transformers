export CACHE_DIR=/pre-trained-transformers
export BASE_DIR=$1
export FILENAME=$2
export CUDA_VISIBLE_DEVICES=$3
export NODE_RANK=0
export N_NODES=1
export N_GPU_NODE=$4
export WORLD_SIZE=$4
export EPOCH=$5
export BINARIZED=$6

if [ $BINARIZED -eq 1 ]; then
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
fi

pkill -f 'python -u train.py'

export TRAINABLE=$7
if [ $TRAINABLE -eq 1 ]; then
python -m torch.distributed.launch \
    --nproc_per_node=$N_GPU_NODE \
    --nnodes=$N_NODES \
    --node_rank $NODE_RANK \
    train.py \
	--gradient_accumulation_steps 100 \
	--n_epoch $EPOCH \
        --force \
        --gpus $WORLD_SIZE \
        --student_type distilbert \
        --student_config training_configs/distilbert-base-uncased.json \
        --teacher_type bert \
        --cache_dir $CACHE_DIR \
        --teacher_name bert-base-uncased \
        --alpha_ce 1.0 --alpha_mlm 0.0 --alpha_cos 0.0 --alpha_clm 0.0 --mlm \
        --freeze_pos_embs \
        --dump_path /logs/distilbert/$FILENAME.init-bbu-0247911_label-bbu-correction_ce-loss-1_distilbert \
        --data_file $BASE_DIR/binarized_$FILENAME.bert-base-uncased.pickle \
        --token_counts $BASE_DIR/token_counts.bert-base-uncased.pickle \
        --checkpoint_epoch_interval 1 \
        --student_pretrained_weights /logs/weight_initialization/bert-base-uncased_0247911.pth \
        --teacher_distribution correction
fi