import pickle
with open("data/dictionary.txt", "rb") as fp:
    dictionary = pickle.load(fp)

print(dictionary)