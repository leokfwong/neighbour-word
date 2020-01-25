import getopt, sys
import spacy
import time

# Need to run this line to convert the gensim model into spacy format
# python -m spacy init-model en ./models/spacy-min-count-100 --vectors-loc models/gensim-model-min-count-100.txt.gz

def computeNeighboringWords(model, top=10, input="", write=False, threshold=0.5, n_prune=0, write_file=False, verbose=False, file_name_suffix=""):
	# Load model with spacy
	nlp = spacy.load("models/" + model)

	# Prune words from vocab
	if n_prune > 0:
		n_vectors = len(list(nlp.vocab.strings)) - n_prune
		print ("Pruning " + str(n_prune) + " vectors")
		removed_words = nlp.vocab.prune_vectors(n_vectors)

	# Fetch list of words from vocab
	vocab = list(nlp.vocab.strings)

	# Initialize variables
	count = 0
	subset_tokens = vocab
	start_time = time.time()

	# Determine if input is all words from vocab, a list of words or a single word
	if (len(input) == 0):
		print("Finding " + str(top) + " closest neighbors for all words")
		row_count = len(vocab)
	elif (isinstance(input, list)):
		print("Finding " + str(top) + " closest neighbors for list of words")
		subset_tokens = input
		row_count = len(subset_tokens)
	else:
		print("Finding " + str(top) + " closest neighbors for the word " + str(input))
		subset_tokens = []
		subset_tokens.append(input)
		row_count = len(subset_tokens)

	# Iterate through input
	for token1 in subset_tokens:
		if count < row_count:
			# Check if token1 is a valid word
			if (nlp(token1).vector_norm > 0):
				# Initialize list
				words = []
				score = []
				# Iterate through every other word
				for token2 in vocab:
					# Not equal to word being compared to
					if (token2 != token1):
						# Check if token2 is a valid word
						if (nlp(token2).vector_norm > 0):
							# Calculate similarity score between token1 and token2
							sim_score = nlp(token1).similarity(nlp(token2))
							# Append results to list
							words.append(token2)
							score.append(sim_score)
				# If word was valid
				if len(words) > 0:
					# Sort in descending order by similarity scores, keep if above threshold
					list1, list2 = (list(t) for t in zip(*sorted(zip(score, words), reverse=True)))
					list1 = [x for x in list1 if x > float(threshold)]
					list2 = list2[:len(list1)]

					# Fetch top neighboring words
					if len(list1) > int(top):
						neigh = list2[:int(top)]
					else:
						neigh = list2

					# Print output
					out = '{:<12}  {:<12}  {:<12}'.format(token1, str(len(list1)), " ".join(neigh))
					if (verbose):
						print(out)

					# Write to file
					if (write_file):
						print("Writing to output file...")
						if (count == 0):
							out_mode = "w"
						else:
							out_mode = "a"
						with open("output/out-" + model + file_name_suffix + ".txt", out_mode) as outfile:
							outfile.write(out + "\n")

					# Increment count by one if not counting all
					count += 1
			else:
				print("Word " + token1 + " was never seen before by the model.")
				
	# Calculate execution time
	end_time = time.time()
	print("Total execution time: {}".format(end_time - start_time))

if __name__ == "__main__":
	# If no parameters are passed through the terminal, define values here
	model="spacy-min-count-100"
	top=10
	input=["king"]
	threshold=0.5
	n_prune=0
	write_file=False
	verbose=False
	file_name_suffix=""

	# Fetch commandline arguments
	args_list = sys.argv[1:]
	if (len(args_list) > 0):
		options = "m:t:i:h:n:wvf:"
		long_options = ["model=", "top=", "input=", "threshold=", "n_prune=", "write_file", "verbose", "file_name_suffix="]
		try:
			opts, args = getopt.getopt(args_list, options, long_options)
		except getopt.error as err:
			print (str(err))
			sys.exit(2)

		# Evaluate options
		for opt, arg in opts:
			if opt in ("-m", "--model"):
				model=arg
			elif opt in ("-t", "--top"):
				top=arg
			elif opt in ("-i", "--input"):
				input=arg
			elif opt in ("-h", "--threshold"):
				threshold=arg
			elif opt in ("-n", "--n_prune"):
				n_prune=int(arg)
			elif opt in ("-w", "--write_file"):
				write_file=True
			elif opt in ("-v", "--verbose"):
				verbose=True
			elif opt in ("-f", "--file_name_suffix"):
				file_name_suffix=arg
			else:
				print("Invalid argument.")

	# Call function
	computeNeighboringWords(
		model=model,
		top=top, 
		input=input, 
		threshold=threshold,
		n_prune=n_prune, 
		write_file=write_file, 
		verbose=verbose, 
		file_name_suffix=file_name_suffix)