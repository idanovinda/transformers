from datasets import load_dataset

wiki = load_dataset("wikipedia", "20200501.en", cache_dir="/datasets/huggingface", split="train")
wiki.remove_columns_("title")  # only keep the text

dest_path = "/datasets/huggingface/wikipedia_100/"
filename = "wikipedia_100.txt"

with open(dest_path+filename, "a+") as f:
    for i in range(100):
        f.write(wiki[i]['text'])