# coding=utf-8
# Copyright 2019-present, the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocessing script before distillation.
"""
import argparse
import logging
import pickle
import random
import time
import ray
import numpy as np

from transformers import BertTokenizer, GPT2Tokenizer, RobertaTokenizer


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
)
logger = logging.getLogger(__name__)

ray.init(dashboard_host="127.0.0.1")

@ray.remote
def tokenize_data(tokenizer, bos, sep, data):
    rslt = []
    for text in data:
        text = f"{bos} {text.strip()} {sep}"
        token_ids = tokenizer.encode(text, add_special_tokens=False, truncation='only_second', max_length=tokenizer.model_max_length)
        rslt.append(token_ids)
    return rslt

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess the data to avoid re-doing it several times by (tokenization + token_to_ids)."
    )
    parser.add_argument("--file_path", type=str, default="data/dump.txt", help="The path to the data.")
    parser.add_argument("--tokenizer_type", type=str, default="bert", choices=["bert", "roberta", "gpt2"])
    parser.add_argument("--tokenizer_name", type=str, default="bert-base-uncased", help="The tokenizer to use.")
    parser.add_argument("--dump_file", type=str, default="data/dump", help="The dump file prefix.")
    parser.add_argument("--cache_dir", type=str, default=None, help="The directory where the pretrained transformers saved")
    parser.add_argument("--n_par", type=int, default=1, help="Number of parallel process")
    args = parser.parse_args()

    logger.info(f"Loading Tokenizer ({args.tokenizer_name})")
    if args.tokenizer_type == "bert":
        tokenizer = BertTokenizer.from_pretrained(args.tokenizer_name, cache_dir=args.cache_dir)
        bos = tokenizer.special_tokens_map["cls_token"]  # `[CLS]`
        sep = tokenizer.special_tokens_map["sep_token"]  # `[SEP]`
    elif args.tokenizer_type == "roberta":
        tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name, cache_dir=args.cache_dir)
        bos = tokenizer.special_tokens_map["cls_token"]  # `<s>`
        sep = tokenizer.special_tokens_map["sep_token"]  # `</s>`
    elif args.tokenizer_type == "gpt2":
        tokenizer = GPT2Tokenizer.from_pretrained(args.tokenizer_name, cache_dir=args.cache_dir)
        bos = tokenizer.special_tokens_map["bos_token"]  # `<|endoftext|>`
        sep = tokenizer.special_tokens_map["eos_token"]  # `<|endoftext|>`

    logger.info(f"Loading text from {args.file_path}")
    with open(args.file_path, "r", encoding="utf8") as fp:
        data = fp.readlines()

    logger.info("Start encoding")
    logger.info(f"{len(data)} examples to process.")

    rslt = []
    iter = 0
    interval = 10000
    start = time.time()
    for i in range(args.n_par):
        rslt_i = tokenize_data.remote(tokenizer, bos, sep, data[int(i/args.n_par*len(data)):int((i+1)/args.n_par*len(data))])
        rslt.extend(ray.get(rslt_i))
    logger.info("Finished binarization")
    logger.info(f"{len(data)} examples processed.")

    dp_file = f"{args.dump_file}.{args.tokenizer_name}.pickle"
    vocab_size = tokenizer.vocab_size
    if vocab_size < (1 << 16):
        rslt_ = [np.uint16(d) for d in rslt]
    else:
        rslt_ = [np.int32(d) for d in rslt]
    random.shuffle(rslt_)
    logger.info(f"Dump to {dp_file}")
    with open(dp_file, "wb") as handle:
        pickle.dump(rslt_, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
