#!/usr/bin/env python3

# The Visualisation class should live here

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pyLDAvis.gensim

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from wordcloud import WordCloud

from utils import _setup_logger


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

    def plot_pyldavis(self):
        """Plot visualisation of LDA model with pyLDAvis"""

        self.logger.info(f"Plotting pyLDAvis")

        vis_data = pyLDAvis.gensim.prepare(self.model['model'], self.model['corpus'], self.model['dictionary'])
        pyLDAvis.save_html(vis_data, open('pyLDAvis_latest.html', 'w'))

    def plot_doc_topics(self):
        """Plot document topics, sorted by index"""
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

        self.logger.info(f"Plotting topic word clouds")
        for topic in range(self.model['model'].num_topics):
            self.logger.info(f"Plotting topic {topic}")
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
            self.logger.info(f"Plotting topic {topic}")
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

