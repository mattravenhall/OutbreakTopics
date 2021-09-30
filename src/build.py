#!/usr/bin/env python3

# Modeller and Optimiser classes should live here

import gensim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import re
import string

from collections import Counter
from collections import defaultdict
from gensim.models import LdaModel
from nltk.stem import PorterStemmer

from utils import _setup_logger


CUSTOM_STOPWORDS = {'case', 'cases', 'health', 'report', 'reported', 'reports', 'reporting'}
STOPWORDS = {word.strip() for word in open('../Lexicon/stopwords.txt').readlines()}.union(CUSTOM_STOPWORDS)
MULTI_WORD_TOKENS = open('../Lexicon/multi_word_tokens.txt').readlines()

ps = PorterStemmer()


class Modeller(object):
    """Perform topic modelling for a given set of documents."""
    def __init__(self, input_file, num_topics, logger=None):
        self.logger = logger if logger is not None else _setup_logger('Modeller')

        self.input_file = input_file
        self.num_topics = num_topics

        self.corpus = None
        self.dictionary = None
        self.model = None
        self.topic_weights = None

        self.raw_documents = []
        self.documents = []
        self.num_docs = None

    def build_model(self, seed=None, load_not_build=False):
        self.documents = self.parse_documents(self.input_file, update_raw=True)
        self.num_docs = len(self.raw_documents)
        self.build_gensim_model(seed=seed, load_not_build=load_not_build)
        self.get_topic_weights()
        self.get_token_topics()

    def sample_by_weights(self, weights):
        """Sample indexes according to weights.
        For example, weights [1, 2, 3] would pick index 0 1/6, 1 1/3, & 2 1/2 times."""
        total_0n = sum(weights) * random.random()
        for index, weight in enumerate(weights):
            total_0n -= weight
            if total_0n <= 0:
                return(index)

    @staticmethod
    def standardise_line(line):
        # Remove awkward characters
        # TODO: Determine what these are and whether they can be cleaned more elegantly. Encoding issue?
        line = re.compile(r'(\x87|\x96|\u202c|\u202a|\xa0)').sub('', line)

        # Left-to-right embedding (U+202A)
        # Right-to-left embedding (U+202B)

        # Remove punctuation
        line = line.translate(str.maketrans('', '', string.punctuation+'—–‘’“”')).lower()

        # Mask files as '<FILE>'
        line = re.compile(r'\S*(jpg|jpeg|png|tiff|pdf|xls|xlsx|doc|docx)(?!\w)').sub('<FILE>', line)

        # Mask urls as '<URL>'
        line = re.compile(r'(?<!\w)http\S*').sub('<URL>', line)

        # Mask currency as '<CURRENCY>'
        line = re.compile(r'(?<!\w)[$£¥€￠＄﹩￡￥¢₤][\d.]*').sub('<CURRENCY>', line)
        
        # Mask years as 1900..2099 to '<YEAR>'
        # line = re.compile(r'(?<!\w)(19|20)\d{2}(?!\w)').sub('<YEAR>', line)  # Standardise years
        line = re.compile(r'(?<!\w)(19|20)\d{2}(?!\w)').sub('', line)  # Remove years
        # TODO: Consider removing, rather than masking, years 

        # Mask counts as '<COUNT>'
        line = re.compile(r'(?<!\w)n\d+').sub('<COUNT>', line)

        # Mask ages as '<AGE>'
        # TODO: Possible expand this
        line = re.compile(r'\d+yearold').sub('<AGE>', line)

        # Convert multi-word tokens to '_' separated tokens
        for multi_word_token in MULTI_WORD_TOKENS:
            multi_word_token = multi_word_token.strip()
            line = line.replace(multi_word_token, multi_word_token.replace(' ','_'))

        # Remove newline, split by white space
        line = line.strip().split()

        return line

    @staticmethod
    def valid_token(word):
        if word in STOPWORDS or word.isnumeric():
            return False
        else:
            return True

    @staticmethod
    def stem_word(word):
        if word.startswith('<') and word.endswith('>'):
            return word
        else:
            return ps.stem(word)

    def parse_documents(self, input_file, update_raw=False):
        """Read in file, assuming that each line is a document."""
        self.logger.info(f"Parsing input document: {input_file}")
        raw_documents = [doc for doc in open(input_file).readlines()]
        if update_raw:
            self.raw_documents = raw_documents
        documents = [[self.stem_word(word) for word in self.standardise_line(line) if self.valid_token(word.lower())] for line in raw_documents]
        found_once = {token for token, count in Counter([word for document in documents for word in document]).items() if count == 1}
        documents = [[word for word in document if word not in found_once] for document in documents if document] # Remove low frequency words and empty documents
        return documents

    def build_gensim_model(self, model_name='topic.model', seed=None, save=True, load_not_build=False):
        """Build LDA model with gensim."""
        self.logger.info(f"Building gensim model")

        # Build model
        self.dictionary = gensim.corpora.Dictionary(self.documents)
        self.corpus = [self.dictionary.doc2bow(doc) for doc in self.documents]

        if load_not_build:
            self.logger.info(f"Loading prebuilt model '{model_name}'")
            self.model = LdaModel.load(model_name)
        else:
            # self.model = gensim.models.LdaModel(
            self.model = gensim.models.LdaMulticore(
                self.corpus,
                num_topics=self.num_topics,
                id2word=self.dictionary,
                passes=10,
                workers=2,
                random_state=seed,
                # alpha='auto', eta='auto',   # LdaMulticore doesn't allow these autos, LdaModel does.
            )

            ## Save Model
            if save:
                self.model.save(model_name)
            # model = LdaModel.load('topic.model')

        # Calculate model coherence
        coherence = gensim.models.CoherenceModel(
            model=self.model,
            texts=self.documents,
            dictionary=self.dictionary,
            coherence='c_v'
        ).get_coherence()
        self.logger.info(f"Model Coherence: {coherence}")

    def export_model(self):
        """Package up model details for passing to visualiser."""

        packaged_model = {
            'corpus': self.corpus,
            'model': self.model,
            'token_topics': self.token_topics,
            'topic_num': self.topic_num,
            'topic_weights': self.topic_weights,
            'dictionary': self.dictionary,
        }

        return packaged_model

    def get_topic_weights(self):
        """Get topic weights of built model"""
        self.logger.info("Isolating document topic weights")

        topic_weights = defaultdict(dict)
        for doc, topic_prob in enumerate(self.model[self.corpus]):
            for topic, probability in topic_prob:
                topic_weights[doc][topic] = probability
        self.topic_weights = pd.DataFrame(topic_weights).fillna(0).transpose()
        self.topic_num = np.argmax(self.topic_weights.values, axis=1)

    def get_token_topics(self):
        """Get topic weights of built model"""
        # TODO: Determine if this can be useful, currently almost all tokens are uninformative.

        self.logger.info("Isolating token topic weights")
        token_topics = defaultdict(dict)
        for token_id in range(self.model.num_terms):
            token_str = self.model.id2word[token_id]
            topic_probs = self.model.get_term_topics(token_id, minimum_probability=0)
            if not topic_probs:
                token_topics[token_str][0] = 0
            else:
                for topic, probability in topic_probs:
                    token_topics[token_str][topic] = probability
        self.token_topics = pd.DataFrame(token_topics).fillna(0).transpose()

    def find_representative_documents(self, filename='rep_doc'):
        """Identify and write out documents most representative of each topic."""

        self.logger.info(f"Finding representative documents")

        rep_docs = pd.DataFrame(
            [self.topic_weights.max(), self.topic_weights.idxmax()],
            index=['prob', 'docid']
        )

        # rep_docs = dict(self.topic_weights.idxmax())
        for topic in range(self.num_topics):
            probability = f"{rep_docs[topic]['prob']:.2f}".replace('.', 'p')
            doc_id = rep_docs[topic]['docid'].astype(int)
            n_p_gt_0_95 = sum(self.topic_weights[topic] >= 0.95)
            self.logger.debug(f"Topic {topic} has {n_p_gt_0_95} document{'s' if n_p_gt_0_95 > 1 else ''}"
                              f" with topic probability >= 0.95")
            # self.logger.info(f"Topic {topic}: {sum(modeller.topic_weights[topic] > 0.5)}")
            with open(f"{filename}_topic_{topic}_{probability}.txt", 'w') as rep_doc:
                try:
                    rep_doc.write(self.raw_documents[doc_id])
                except:
                    self.logger.critical('>>>>rep doc failure<<<<')
                    breakpoint()

    def predict_topic(self, new_doc_file):
        """Given a document, predict its topic."""
        self.logger.info(f"Predicting topic for given document")

        new_doc = self.parse_documents(new_doc_file)[0]
        new_doc = self.dictionary.doc2bow(new_doc)
        self.logger.info(f"Predicted Topics: {self.model.get_document_topics(new_doc)}")

        # # Update the model by incrementally training on the new corpus.
        # lda.update(other_corpus)  # update the LDA model with additional documents

    def find_optimal_topic_number(self, topic_range=np.arange(6,20,2), alpha_range=np.arange(0,1,0.2), beta_range=np.arange(0.0,1,0.2)): #limit=20, start=2, step=2):
        # Then pick the optimum topic number given the coherence score for each model
        # Best coherence prior prior to plateau is best.
        # ie. the coherence is best and the number of topics isn't excessive.
        # This should probably be run as a prep-step with human evaluation, rather than automated.

        self.logger.info(f"Looking for optimal topic number")

        # TODO: Make this setup stage generic, plug into self.build_model()
        documents = self.parse_documents(self.input_file)
        dictionary = gensim.corpora.Dictionary(documents)
        corpus = [dictionary.doc2bow(doc) for doc in documents]

        def coherence_values_computation(dictionary, corpus, texts, topic_range, alpha_range, beta_range):
            coherence_values = defaultdict(list)
            # model_list = []

            for alpha in alpha_range:
                for beta in beta_range:
                    for num_topics in topic_range:
                        self.logger.info(f"Identifying coherence with topics_{num_topics}, alpha={alpha}, beta={beta}")

                        model = gensim.models.LdaModel(
                        # model = gensim.models.LdaMulticore(
                            corpus,
                            num_topics=num_topics,
                            id2word=dictionary,
                            passes=10,
                            # workers=2,
                            # alpha='auto',
                            # eta='auto',
                        )
                        # # model_list.append(model)

                        self.logger.info(f"Calculating coherence with {num_topics} topics")
                        coherencemodel = gensim.models.CoherenceModel(
                            model=model, texts=texts, dictionary=dictionary, coherence='c_v'
                        )
                        # coherence_values.append(coherencemodel.get_coherence())

                        coherence_values['Topics'].append(num_topics)
                        coherence_values['Alpha'].append(alpha)
                        coherence_values['Beta'].append(beta)
                        coherence_values['Coherence'].append(coherencemodel.get_coherence())

                        # del model
                    # return model_list, coherence_values
            return coherence_values

        # model_list,
        coherence_values = coherence_values_computation(
           dictionary=dictionary, corpus=corpus, texts=documents,
           # start=start, step=step, limit=limit
           topic_range=topic_range,
           alpha_range=alpha_range,
           beta_range=beta_range
        )
           # dictionary=id2word, corpus=corpus, texts=data_lemmatized,

        # TODO: Optimise alpha and eta also

        self.logger.info(f"Plotting coherence by topic number")
        coherence_fig, coherence_ax = plt.subplots()
        coherence_ax.plot(range(start, limit, step), coherence_values)
        coherence_ax.set_title("Optimising Number of Topics by Coherence")
        coherence_ax.set_xlabel("Number of Topics")
        coherence_ax.set_ylabel("Coherence Score")
        # plt.legend(("coherence_values"), loc='best')
        coherence_fig.savefig("coherence_by_num_topics.png")
        plt.close(coherence_fig)
