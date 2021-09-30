#!/usr/bin/env python3

# Support functions (e.g. setup_logger) and constants should live here

import argparse
import logging
import sys
import os


def _file_exists(file_path):
    if not os.path.isfile(file_path):
        print(f"Provided file '{file_path}' does not exists.")
        sys.exit(-1)
    return file_path


def _setup_logger(logger_name, log_to_file=False, level=logging.INFO):
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    # Console logger
    c_handler = logging.StreamHandler()
    c_handler.setLevel(logging.DEBUG)
    c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(c_format)
    logger.addHandler(c_handler)

    # File logger
    if log_to_file:
        f_handler = logging.FileHandler('topic_modeller.log')
        f_handler.setLevel(logging.ERROR)
        f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        f_handler.setFormatter(f_format)
        logger.addHandler(f_handler)

    logger.debug("Logger set up successfully")

    return logger


def _setup_argparse():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument('--corpus', type=_file_exists, help="Path to corpus file.")
    parser.add_argument('--num_topics', type=int, default=12, help="Number of topics to learn.")
    parser.add_argument('--run_types', type=str, default={''},
                        choices={
                            "OPTIMISE", "TRAIN"
                        }, help="Functions to perform.")
    parser.add_argument('--explorations', type=str, nargs='+',
                        choices={
                            "PCA", "TSNE", "PYLDAVIS", "DOC_TOPICS",
                            "WORDCLOUDS", "WORD_BARS", "TOKEN_TOPICS",
                            "REP_DOCS",
                        }, help="Visualisations & explorations to produce.")
    parser.add_argument('--load_prebuilt', action='store_true', help="Include to load an existing model.")
    parser.add_argument('--seed', type=int, default=None, help=f"Model seed for reproducible builds.")
    parser.add_argument('--predict', type=_file_exists, default=None, help="Path to file to predict topics for.")

    args = parser.parse_args()
    return args
