#!/usr/bin/env python3

# From scratch Latent Dirichlet Allocation

import json
import random
import re
import string

from collections import Counter
from nltk.stem import PorterStemmer

print("Warning: This script is incredible inefficient. Use the gensim-based approach instead!")

ps = PorterStemmer()

STOPWORDS = {word.strip() for word in open('../Lexicon/stopwords.txt').readlines()}.union({'case'})
MULTI_WORD_TOKENS = open('../Lexicon/multi_word_tokens.txt').readlines()

input_file = '../Documents/documents_1996-2019.txt' #'Documents/example.txt' #'test_input.txt'


def sample_by_weights(weights):
	"""Sample indexes according to weights.
	For example, weights [1, 2, 3] would pick index 0 1/6, 1 1/3, & 2 1/2 times."""
	total_0n = sum(weights) * random.random()
	for index, weight in enumerate(weights):
		total_0n -= weight
		if total_0n <= 0:
			return(index)


def standardise_line(line):
	# Remove awkward characters, work out what these are and whether they can be cleaned more elegantly
	line = re.compile(r'(\x87|\x96|\u202c|\u202a|\xa0)').sub('', line)

	# Remove punctuation
	line = line.translate(str.maketrans('', '', string.punctuation+'—–‘’“”')).lower()

	# Convert probable image files to '<IMAGE>'
	line = re.compile(r'\S*(jpg|jpeg|png|tiff|pdf|xls|xlsx|doc|docx)(?!\w)').sub('<FILE>', line)

	# Convert urls to '<URL>'
	line = re.compile(r'(?<!\w)http\S*').sub('<URL>', line)

	# Replace currency with '<CURRENCY>'
	line = re.compile(r'(?<!\w)[$£¥€￠＄﹩￡￥¢₤][\d.]*').sub('<CURRENCY>', line)
	
	# Convert years in 1900..2099 to '<YEAR>'
	line = re.compile(r'(?<!\w)(19|20)\d{2}(?!\w)').sub('<YEAR>', line)

	# Convert counts to '<COUNT>'
	line = re.compile(r'(?<!\w)n\d+').sub('<COUNT>', line)

	# Convert multi-word tokens to '_' token
	for multi_word_token in MULTI_WORD_TOKENS:
		multi_word_token = multi_word_token.strip()
		line = line.replace(multi_word_token, multi_word_token.replace(' ','_'))

	# Remove newline, split by white space		
	line = line.strip().split()

	return line


def valid_token(word):
	if word in STOPWORDS:
		return False
	elif word.isnumeric():
		return False
	else:
		return True


def stem_word(word):
	if word.startswith('<') and word.endswith('>'):
		return word
	else:
		return ps.stem(word)


# Read in file, assuming that each line is a document.
print(f"Parsing input document: {input_file}")
documents = [[stem_word(word) for word in standardise_line(line) if valid_token(word)] for line in open(input_file).readlines()]
found_once = {token for token, count in Counter([word for document in documents for word in document]).items() if count == 1}
documents = [[word for word in document if word not in found_once] for document in documents if document] # Remove low frequency words and empty documents


# Build LDA model
num_topics = 8
print(f"Modelling for {num_topics} topics.")
## each document, d, is a list of words

## First assign each word a topic randomly
## Iterate each document a word at a time
## Considering the distribution of topics in the documents, and words in that topics, update topics for each word.
## Continue until a 'joint sample' is found (???)

# Determine counts
document_topic_counts = [Counter() for _ in documents] # Times topic assigned to document
topic_word_counts = [Counter() for _ in range(num_topics)] # Times words assigned to topic
topic_counts = [0 for _ in range(num_topics)] # Number of words in each topic
document_lengths = [x for x in map(len, documents)] # Number of words in each documents
distinct_words = len(set(word for document in documents for word in document)) # Number of distinct words
document_total = len(documents) # Number of documents

## TODO: Vectorise this, it's slow af atm.
# Conditional probability functions
def p_topic_given_document(topic, doc, alpha=0.1):
	"""Fraction words in document assigned to topic, smoothed."""
	return((document_topic_counts[doc][topic] + alpha) / (document_lengths[d] + num_topics * alpha))

def p_word_given_topic(word, topic, beta=0.1):
	"""Fraction of words assign to topic for word, smoothed."""
	return((topic_word_counts[topic][word] + beta) / (topic_counts[topic] + distinct_words * beta))

# Updating weights
def topic_weights(doc, word, topic_idx):
	return(p_word_given_topic(word, topic_idx) * p_topic_given_document(topic_idx, doc))

def select_new_topic(doc, word):
	return(sample_by_weights([topic_weights(doc, word, topic_idx) for topic_idx in range(num_topics)]))


print('Initialising weights')
# random.seed(0)
document_topics = [[random.randrange(num_topics) for word in document] for document in documents]

print('Calculating initial statistics')
for d in range(document_total):
	for word, topic in zip(documents[d], document_topics[d]):
		document_topic_counts[d][topic] += 1
		topic_word_counts[topic][word] += 1
		topic_counts[topic] += 1

print('Building LDA model')
eon = 0
limit = 1000
for _ in range(limit):
	eon += 1
	docnum = 0
	for d in range(document_total):
		docnum += 1
		print(f'\rStatus: eon {eon} of {limit} (document {docnum} of {document_total})', end='')
		for i, (word, topic) in enumerate(zip(documents[d], document_topics[d])):
			document_topic_counts[d][topic] -= 1
			topic_word_counts[topic][word] -= 1
			topic_counts[topic] -= 1
			document_lengths[d] -= 1

			# Assign new topic
			new_topic = select_new_topic(d, word)
			document_topics[d][i] = new_topic

			# Add back to counts
			document_topic_counts[d][new_topic] += 1
			topic_word_counts[new_topic][word] += 1
			topic_counts[new_topic] += 1
			document_lengths[d] += 1

# Export models to JSON
print('Exporting models to JSON')
with open('./Output/topic_word_counts.json', 'w') as f:
    json.dump(topic_word_counts, f)

with open('./Output/document_topic_counts.json', 'w') as f:
    json.dump(document_topic_counts, f)

# Write various results files
print('Summarising topics')
with open('./Output/topic_word_counts.txt','a') as f:
	f.write('Topic,Word,Count\n')
	for k, word_counts in enumerate(topic_word_counts):
		for word, count in word_counts.most_common():
			if count > 0:
				f.write(f'{k},"{word}",{count}\n')


print('Summarising topics')
with open('./Output/topic_word_topten.txt','a') as f:
	f.write("Topic,Word,Count\n")
	for k, word_counts in enumerate(topic_word_counts):		
		for word, count in word_counts.most_common(10):
			f.write(f"{k},{word},{count}\n")


# Document topics
print('Identifying documents by topic')
with open('./Output/document_topics.txt','a') as f:
	f.write('ID,Topic,Count,Document\n')
	doc_id = 0
	for document, topic_counts in zip(documents, document_topic_counts):
		doc_id += 1
		for topic, count in topic_counts.most_common():

			if count > 0:
				f.write(f"{doc_id},{topic},{count},{' '.join(document)}\n")


# Data exploration
import pandas as pd
# Document topic proportions
doc_topics = pd.DataFrame(document_topic_counts)[range(num_topics)]
doc_topics = doc_topics.div(doc_topics.sum(axis=1),axis=0)*100	# Convert counts to proportions
doc_topics['Doc'] = [' '.join(doc) for doc in documents]

# Post where topic == 100%
perfect_examples = doc_topics.apply(lambda x: sum(x==100), axis=0).to_dict()


doc_topics[doc_topics[0] == 100].index.to_list()


documents_full = [line for line in open(input_file).readlines()]
doc_topics['Full'] = [line for line in open(input_file).readlines()]
doc_topics['SD'] = doc_topics[range(8)].std(axis=1)

doc_topics.to_csv('./Output/document_stats.csv')

breakpoint()

# gensim
from gensim import corpora, models
dictionary = gensim.corpora.Dictionary(processed_docs)
bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=10, id2word=dictionary, passes=2, workers=2)
