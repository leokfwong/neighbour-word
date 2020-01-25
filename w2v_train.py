import gensim
from gensim.test.utils import datapath
from gensim import utils
import time

# Load pre-trained models
# wv = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)

# Train model from corpus
class Corpus(object):
    def __init__(self, filename):
        self.filename = filename
 
    def __iter__(self):
    	for line in open(self.filename, encoding="utf8"):
    		yield line.split()

# Train model
sentences = Corpus('data/espn_2019.txt')
start = time.time()
# min_count = 5
# size = 100
model = gensim.models.Word2Vec(sentences, min_count=5, compute_loss=True, seed=42)
end = time.time()
print(end - start)

# getting the training loss value
print("Model traning loss value: " + str(model.get_latest_training_loss()))

# Save model to convert to spacy
model.wv.save_word2vec_format("models/espn-2019-model-min-count-5.txt")