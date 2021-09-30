# Primary run script

from build import Modeller
from utils import _setup_logger, _setup_argparse
from visualise import Visualiser


MAIN_LOGGER = _setup_logger('TopicModeller')
CMD_ARGS = _setup_argparse()


if __name__ == '__main__':

    if 'OPTIMISE' in CMD_ARGS.run_types:
        optimiser = Modeller(CMD_ARGS.corpus, num_topics=CMD_ARGS.num_topics)
        optimiser.find_optimal_topic_number()
    else:
        modeller = Modeller(CMD_ARGS.corpus, num_topics=CMD_ARGS.num_topics)
        modeller.build_model(seed=CMD_ARGS.seed, load_not_build='TRAIN' not in CMD_ARGS.run_types)
        if CMD_ARGS.predict:
            modeller.predict_topic(CMD_ARGS.predict)
        if 'REP_DOCS' in CMD_ARGS.explorations:
            modeller.find_representative_documents()

        if CMD_ARGS.visualisations is not None:
            trained_model = modeller.export_model()
            visualiser = Visualiser(trained_model)

            if 'PCA' in CMD_ARGS.explorations:
                visualiser.plot_tsne()
            if 'TSNE' in CMD_ARGS.explorations:
                visualiser.plot_pca()
            if 'PYLDAVIS' in CMD_ARGS.explorations:
                visualiser.plot_pyldavis()
            if 'DOC_TOPICS' in CMD_ARGS.explorations:
                visualiser.plot_doc_topics()
            if 'WORDCLOUDS' in CMD_ARGS.explorations:
                visualiser.plot_topic_clouds()
            if 'WORD_BARS' in CMD_ARGS.explorations:
                visualiser.plot_word_bars()
            if 'TOKEN_TOPICS' in CMD_ARGS.explorations:
                visualiser.plot_token_topics()
