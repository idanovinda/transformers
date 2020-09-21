from datasets import load_dataset

wiki = load_dataset("wikipedia", "20200501.en", cache_dir="/datasets/huggingface", split="train")
wiki.remove_columns_("title")  # only keep the text

dest_path = "/datasets/huggingface/wikipedia/"
filename = "wikipedia.txt"

with open(dest_path+filename, "a+") as f:
    for i in range(len(wiki)):
        split_text = wiki[i]['text'].strip("\n").split("\n")
        for text in split_text:
            if text != "":
                f.write(text+"\n")