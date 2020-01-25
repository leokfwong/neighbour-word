import pandas as pd

csv_file = pd.read_csv("../tp2/data/train_posts.csv", header=None, sep=",")

# Fetch column with sentences and only keep words
corpus = csv_file[0].str.replace('[^a-zA-Z \n]','')
# Set all words to lower
corpus = corpus.str.lower()

# Write to txt
#corpus.to_csv(r'train_posts.txt', index=None, sep=' ', header=None)
with open('data/train_posts.txt', 'w') as filehandle:
    filehandle.writelines("%s\n" % line for line in corpus)