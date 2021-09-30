# LDA topic modelling of outbreak reports

## Quick Build
```bash
python3 __main__.py --corpus '../Documents/documents_1996-2019.txt'
```
> nb. Input document assumes one document per line.

## Optimisation
```bash
python3 __main__.py --corpus '../Documents/documents_1996-2019.txt' --run_types OPTIMISE
```

|     |
| --- |
| [<img src="./Examples/coherence_by_num_topics.png" width="250px"/>](./Examples/coherence_by_num_topics.png) |

## Exploration
### Word Clouds
```bash
python3 __main__.py --corpus 'training_documents.txt' --explorations WORDCLOUDS
```
|     |     |
| --- | --- |
| [<img src="./Examples/wordcloud_topic_0.png" width="250px"/>](./Examples/wordcloud_topic_0.png) | [<img src="./Examples/wordcloud_topic_1.png" width="250px"/>](./Examples/wordcloud_topic_1.png) |
| [<img src="./Examples/wordcloud_topic_4.png" width="250px"/>](./Examples/wordcloud_topic_4.png) | [<img src="./Examples/wordcloud_topic_5.png" width="250px"/>](./Examples/wordcloud_topic_5.png) |

### Word Bars
```bash
python3 __main__.py --corpus 'training_documents.txt' --explorations WORD_BARS
```
|     |     |
| --- | --- |
| [<img src="./Examples/word_bars_topic_0.png" width="250px"/>](./Examples/word_bars_topic_0.png) | [<img src="./Examples/word_bars_topic_1.png" width="250px"/>](./Examples/word_bars_topic_1.png) |
| [<img src="./Examples/word_bars_topic_4.png" width="250px"/>](./Examples/word_bars_topic_4.png) | [<img src="./Examples/word_bars_topic_5.png" width="250px"/>](./Examples/word_bars_topic_5.png) |

### Clustering
```bash
python3 __main__.py --corpus 'training_documents.txt' --explorations PCA TSNE
```
|     |     |
| --- | --- |
| [<img src="./Examples/pca.png" width="250px"/>](./Examples/pca.png) | [<img src="./Examples/tsne.png" width="250px"/>](./Examples/tsne.png) |

### Topics by Document
```bash
python3 __main__.py --corpus 'training_documents.txt' --explorations DOC_TOPICS
```

|     |
| --- |
| [<img src="./Examples/doc_topics.png" width="250px"/>](./Examples/doc_topics.png) |

### pyLDAvis
```bash
python3 __main__.py --corpus 'training_documents.txt' --explorations PYLDAVIS
```

### Representative Documents
```bash
python3 __main__.py --corpus 'training_documents.txt' --explorations REP_DOCS
```

### Topic Prediction
```bash
python3 __main__.py --corpus 'training_documents.txt' --predict 'document_for_prediction.txt'
```
> nb. Input document assumes one document per line.

More info can be found in an accompanying [blog post](https://mattravenhall.github.io/2021/10/01/Topic-Modelling-Disease-Outbreak-News.html).
