import pickle

dump_filename = "/datasets/huggingface/wikipedia_all/binarized_wikipedia_all.bert-base-uncased.pickle"
base_dir = "/datasets/huggingface/wikipedia_split/"
objects = []

for i in range(10):
    with open(base_dir+"binarized_wikipedia_"+str(i)+".bert-base-uncased.pickle", "rb") as handle:
        objects.extend(pickle.load(handle))
        print(len(objects))

with open(dump_filename, "wb") as result_file:
    pickle.dump(objects, result_file, protocol=pickle.HIGHEST_PROTOCOL)
