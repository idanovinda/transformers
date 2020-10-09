bash distributed_run.sh /datasets/wikipedia_full wikipedia_full 0,1,2,3,4,5,6,7 8 bert-base-uncased 3 \
	/logs/distilbert/wikipedia_init-bbu-0247911_label-bbu-uniform_bs-4000_no-mlm_no-cos uniform

bash distributed_run.sh /datasets/wikipedia_full wikipedia_full 0,1,2,3,4,5,6,7 8 bert-base-uncased 3 \
	/logs/distilbert/wikipedia_init-bbu-0247911_label-bbu-shuffle_bs-4000_no-mlm_no-cos shuffle
