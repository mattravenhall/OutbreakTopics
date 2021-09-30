#!/usr/bin/env python3

# gensim-based Latent Dirichlet Allocation

import gensim
import json
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyLDAvis.gensim
import random
import re
import string
import sys
import logging

from collections import Counter
from collections import defaultdict
from gensim.models import LdaModel
from nltk.stem import PorterStemmer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

ps = PorterStemmer()

CUSTOM_STOPWORDS = {'case', 'cases', 'health', 'report', 'reported', 'reports', 'reporting'}
STOPWORDS = {word.strip() for word in open('../../Lexicon/stopwords.txt').readlines()}.union(CUSTOM_STOPWORDS)
MULTI_WORD_TOKENS = open('../../Lexicon/multi_word_tokens.txt').readlines()

# TODO: Utilise coherence and perplexity to optimise the model and number of topics built
# TODO: Include docstrings in Google format        

def _setup_logger(logger_name, log_to_file=False, level=logging.DEBUG): # TODO: Set level to logging.INFO
    # logging.basicConfig(level=logging.DEBUG, format='%(message)s')
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    # Console logger
    c_handler = logging.StreamHandler()
    c_handler.setLevel(logging.DEBUG)
    c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(c_format)
    logger.addHandler(c_handler)

    # File logger
    if log_to_file:
        f_handler = logging.FileHandler('file.log')
        f_handler.setLevel(logging.ERROR)
        f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        f_handler.setFormatter(f_format)
        logger.addHandler(f_handler)

    logger.debug("Logger set up successfully")

    return logger


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
            [modeller.topic_weights.max(), modeller.topic_weights.idxmax()],
            index=['prob', 'docid']
        )

        # rep_docs = dict(self.topic_weights.idxmax())
        for topic in range(self.num_topics):
            probability = f"{rep_docs[topic]['prob']:.2f}".replace('.','p')
            doc_id = rep_docs[topic]['docid'].astype(int)
            n_p_gt_0_95 = sum(modeller.topic_weights[topic] >= 0.95)
            self.logger.debug(f"Topic {topic} has {n_p_gt_0_95} document{'s' if n_p_gt_0_95 > 1 else ''} with topic probability >= 0.95")
            # self.logger.info(f"Topic {topic}: {sum(modeller.topic_weights[topic] > 0.5)}")
            with open(f"{filename}_topic_{topic}_{probability}.txt", 'w') as rep_doc:
                try:
                    rep_doc.write(self.raw_documents[doc_id])
                except:
                    self.logger.critical('>>>>rep doc failure<<<<')
                    breakpoint()

    def find_representative_tokens(self):
        # TODO: Find (list of n?) tokens that are most specific to each topic

        self.logger.info(f"Finding representative tokens")

        raise NotYetImplemented

    def predict_topic(self, new_doc_file):
        """Given a document, predict its topic."""
        self.logger.info(f"Predicting topic for given document")

        new_doc = self.parse_documents(new_doc_file)[0]
        new_doc = self.dictionary.doc2bow(new_doc)
        self.logger.info(f"Predicted Topics: {self.model.get_document_topics(new_doc)}")

        # # Update the model by incrementally training on the new corpus.
        # lda.update(other_corpus)  # update the LDA model with additional documents


class Optimiser(object):
    """Identify the optimum parameters for a given corpus."""
    def __init__(self, input_documents, logger=None):
        self.logger = logger if logger is not None else _setup_logger('Optimiser')
        
        self.input_documents = input_documents

        
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


class Visualiser(object):
    """Generate visualisations for a given topic model."""
    def __init__(self, model, logger=None):
        self.logger = logger if logger is not None else _setup_logger('Visualiser')
        self.model = self._verify_model(model)
        self.topic_colors = np.array([
            mcolors.to_hex(cm.Paired(n % 12)) for n in range(self.model['model'].num_topics)
        ])
        # self.plot_colors = np.array([color for name, color in mcolors.TABLEAU_COLORS.items()])


    def _verify_model(self, model):
        required_attributes = {
            "corpus": "Model has not been build, please run <Modeller>.build_model() and re-export.",
            "model": "Model has not been build, please run <Modeller>.build_model() and re-export.",
            "topic_weights": "topic_weights not set, please run <Modeller>.get_topic_weights() and re-export.",
            "topic_num": "topic_num not set, please run <Modeller>.get_topic_weights() and re-export.",
            "token_topics": "token_topics not set, please run <Modeller>.get_token_topics() and re-export.",
        }

        failures = 0
        missing = set()
        for attribute in required_attributes.keys():
            attribute_value = model.get(attribute)
            if attribute_value is None:
                self.logger.critical(required_attributes[attribute])
                missing.add(attribute)
                failures += 1

        if failures:
            raise ValueError(f"{failures} missing attribute{'s' if failures > 1 else ''}: {', '.join(sorted(missing))}")
        else:
            return model

    def plot_tsne(self, filename='tsne'):
        """Build and plot tSNE of model"""

        self.logger.info("Building tSNE")
        tsne_model = TSNE(n_components=2, verbose=1, init='pca')
        tsne_lda = tsne_model.fit_transform(self.model['topic_weights'])

        # Plot tSNE
        self.logger.info("Plotting tSNE")
        tsne_fig, tsne_ax = plt.subplots()
        tsne_ax.set_title(f"t-SNE Clustering of {self.model['model'].num_topics} LDA Topics")
        for topic in range(self.model['model'].num_topics):
            scatter = tsne_ax.scatter(
                x=tsne_lda[self.model['topic_num'] == topic,0],
                y=tsne_lda[self.model['topic_num'] == topic,1],
                color=self.topic_colors[topic]
            )
        tsne_ax.legend(
            labels=np.array(range(self.model['model'].num_topics)),
            labelcolor=self.topic_colors,
            loc='upper right',
            ncol=2,
            # facecolor='#1b2d5c',
            fontsize='x-small'
        )
        tsne_fig.savefig(filename)
        plt.close(tsne_fig)

        self.logger.info(f"t-SNE plot saved to file: {filename}.png")

    def plot_pca(self, filename='pca'):
        """Build and plot PCA of model"""

        self.logger.info("Building PCA")
        pca_model = PCA(n_components=2)
        pca_lda = pca_model.fit_transform(self.model['topic_weights'])

        # Plot PCA
        self.logger.info("Plotting PCA")
        pca_fig, pca_ax = plt.subplots()
        pca_ax.set_title(f"PCA Clustering of {self.model['model'].num_topics} LDA Topics")
        for topic in range(self.model['model'].num_topics):
            pca_ax.scatter(
                x=pca_lda[self.model['topic_num'] == topic,0],
                y=pca_lda[self.model['topic_num'] == topic,1],
                color=self.topic_colors[topic],
            )
        # pca_ax.scatter(x=pca_lda[:,0], y=pca_lda[:,1], color=self.topic_colors[self.topic_num])
        pca_ax.set_xlabel(f"PC1 ({100*pca_model.explained_variance_ratio_[0]:.1f}%)")
        pca_ax.set_ylabel(f"PC2 ({100*pca_model.explained_variance_ratio_[1]:.1f}%)")
        pca_ax.legend(
            labels=np.array(range(self.model['model'].num_topics)),
            labelcolor=self.topic_colors,
            loc='upper right',
            ncol=2,
            # facecolor='#1b2d5c',
            fontsize='x-small'
        )
        pca_fig.savefig(filename)
        plt.close(pca_fig)

        self.logger.info(f"PCA plot saved to file: {filename}.png")

    def plot_tsne_pairs(self):
        """Plot a tSNE pairs plot of learned model"""

        import seaborn as sns

        self.logger.info(f"Plotting t-SNE pairs")

        # Create the default pairplot
        # sns.pairplot(df.transpose(), diag_kind = 'kde', plot_kws = {'alpha': 0.6, 's': 80, 'edgecolor': 'k'}, height = 4)
        sns.pairplot(self.model['model'].num_topics.transpose(), diag_kind = 'kde', plot_kws = {'alpha': 0.6, 's': 80, 'edgecolor': 'k'}, height = 4)
        plt.savefig('pairs')

    def plot_pyldavis(self):
        """Plot visualisation of LDA model with pyLDAvis"""

        self.logger.info(f"Plotting pyLDAvis")

        ## Visualisation
        ## INFO: https://github.com/bmabey/pyLDAvis
        ## INFO: https://shichaoji.com/tag/topic-modeling-python-lda-visualization-gensim-pyldavis-nltk/

        vis_data = pyLDAvis.gensim.prepare(self.model['model'], self.model['corpus'], self.dictionary)
        pyLDAvis.save_html(vis_data, open('pyLDAvis_latest.html', 'w'))

        # TODO: pyLDAvis seems a little janky:
        # - The generated css and js links are currently broken.
        # - Token labels are missing.

    def plot_doc_topics(self):

        self.logger.info(f"Plotting document topics")

        topic_list = [x for x in range(self.model['model'].num_topics)]

        doc_topics_fig, doc_topics_ax = plt.subplots()        
        pcm = doc_topics_ax.imshow(
            self.model['topic_weights'].sort_values(topic_list).transpose(),
            cmap='Greys',
            aspect='auto',
            interpolation='nearest'
        )
        # doc_topics_fig.colorbar(pcm)
        for num_topic in range(self.model['model'].num_topics):
            doc_topics_ax.scatter(
                0, #range(self.num_docs),
                num_topic,
                color=self.topic_colors[num_topic],
                marker='>',
                linewidths=5
                # linewidths=200/self.num_topics
            )
        doc_topics_ax.set_title('Topic Probabilities per Document')
        doc_topics_ax.set_xlabel('Document')
        doc_topics_ax.set_ylabel('Topic')
        doc_topics_fig.savefig('doc_topics.png')
        plt.close(doc_topics_fig)

    def plot_token_topics(self, filename="word_topics"):

        self.logger.info(f"Plotting token topics")

        topic_list = [x for x in range(self.model['model'].num_topics)]

        tokens_fig, tokens_ax = plt.subplots()
        tokens_ax.imshow(
            self.token_topics.sort_values(topic_list).transpose(),
            cmap='Blues',
            aspect='auto',
            interpolation='nearest'
        )
        tokens_ax.set_xlabel('Token')
        tokens_ax.set_ylabel('Topic')
        # plt.show()
        tokens_fig.savefig(f"{filename}.png")
        plt.close(tokens_fig)
   

    def plot_topic_clouds(self, n_words=1000, filename='wordcloud'):
        """Plot word clouds for each topic"""
        from wordcloud import WordCloud


        # bg_colors = [col_hex for col_hex in mcolors.to_hex(cm.Pastel1(topic))]
        # bg_colors = [mcolors.to_hex(cm.Pastel1(n)) for n in range(self.num_topics)]
        # bg_colors = ['#fbb4ae', '#b3cde3', '#ccebc5', '#decbe4', '#fed9a6', '#ffffcc', '#e5d8bd', '#fddaec', '#f2f2f2']
        # bg_colors = ['#b3e2cd', '#fdcdac', '#cbd5e8', '#f4cae4', '#e6f5c9', '#fff2ae', '#f1e2cc', '#cccccc']

        self.logger.info(f"Plotting topic word clouds")
        for topic in range(self.model['model'].num_topics):
            bg_color = self.topic_colors[topic%len(self.topic_colors)]
            word_dict = dict(self.model['model'].show_topic(topic, n_words))

            cloud_fig, cloud_ax = plt.subplots()
            cloud_ax.imshow(
                WordCloud(
                    width = 3000, height = 2000,
                    random_state=1, collocations=False,
                    background_color=bg_color, colormap='Greys'
                ).fit_words(word_dict)
            )
            cloud_ax.set_title(f"Topic {topic}")
            cloud_ax.axis('off')
            cloud_fig.savefig(f"{filename}_topic_{topic}.png", bbox_inches='tight')  # TODO: Resolve false tkinker error from using bbox_inches='tight' here
            plt.close(cloud_fig)

    def plot_word_bars(self, n_words=20, filename='word_bars'):

        self.logger.info(f"Plotting topic bars")
        for topic in range(self.model['model'].num_topics):
            word_probs = dict(self.model['model'].show_topic(topic, n_words))
            bars_fig, bars_ax = plt.subplots()
            bars_ax.bar(
                range(len(word_probs)),
                list(word_probs.values()),
                align='center',
                color=self.topic_colors[topic]
            )

            bars_ax.get_xaxis().set_visible(False)
            for i, word in enumerate(list(word_probs.keys())):
                bars_ax.annotate(
                    word, (i, 0), color='black', rotation=90,
                    ha='center', xytext=(0,5), textcoords="offset points"
                )
            bars_ax.set_ylabel('Probability')

            bars_ax.set_title(f"Topic {topic}")
            bars_fig.savefig(f"{filename}_topic_{topic}.png")
            plt.close(bars_fig)


if __name__ == '__main__':
    # TODO: Add argparse (and logging?)
    corpus = '../Documents/documents_1996-2019.txt' #'Documents/example.txt' #'test_input.txt'

    run_types = {'TRAIN', 'PLOT'}

    if 'OPTIMISE' in run_types:
        optimiser = Optimiser(corpus)
        optimiser.find_optimal_topic_number()
        # breakpoint()
    
    if 'TRAIN' in run_types:
        modeller = Modeller(corpus, num_topics=12)
        # if saved model exists: load model; else: build new
        modeller.build_model(seed=15729335, load_not_build=True)
        # modeller.predict_topic('../Documents/pneumonia_of_unknown_cause.txt')

        # modeller.find_representative_documents()
        # modeller.find_representative_tokens()

        # modeller.plot_pca()
        # modeller.plot_tsne()
        # modeller.plot_pyldavis()
        # modeller.plot_doc_topics()
        # modeller.plot_topic_clouds()
        # modeller.plot_word_bars()
        # modeller.plot_token_topics()

    if 'PLOT' in run_types:
        trained_model = modeller.export_model()
        visualiser = Visualiser(trained_model)

        visualiser.plot_tsne()
        visualiser.plot_pca()
        # visualiser.plot_pyldavis()
        # visualiser.plot_doc_topics()
        # visualiser.plot_topic_clouds()
        # visualiser.plot_word_bars()
        # visualiser.find_representative_documents()
        # visualiser.plot_token_topics()
        # visualiser.find_representative_tokens()
        # breakpoint()

    breakpoint()
