"""topic modeling? just run on LD
"""
import os
import pandas as pd
import config as c

import gensim

TXT_COLUMN = "post_lemmas"
N_WORDS = 10 # to save out per topic (doesn't impact model, just output)

# model training parameters
N_TOPICS = 5
CHUNKSIZE = 2000 # num of documents used each training chunk
PASSES = 20 # num of passes through corpus during training
ITERATIONS = 400 # max num of iterations through corpus when inferring topic distribution
EVAL_EVERY = None
UPDATE_EVERY = 1
RANDOM_STATE = 0
ALPHA = "auto"
ETA = "auto"
PER_WORD_TOPICS = True


import_fname = os.path.join(c.DATA_DIR, "derivatives", "posts-clean.tsv")
export_fname = os.path.join(c.DATA_DIR, "results", "validate-topicmodel.tsv")

df = pd.read_csv(import_fname, sep="\t", encoding="utf-8",
    usecols=["post_id", "user_id", "lucidity", TXT_COLUMN],
    index_col="post_id")



print("Running LDA model...")



# grab relevant rows and build corpus
# (each document needs to be tokenized, so list of lists)
doc_list = df.query("lucidity=='lucid'"
    )[TXT_COLUMN].str.split().tolist()


# build gensim dictionary (id2word) and corpus (freq counts)
dictionary = gensim.corpora.Dictionary(doc_list)
corpus = [ dictionary.doc2bow(doc) for doc in doc_list ]

# build model
lda = gensim.models.ldamodel.LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=N_TOPICS,
    iterations=ITERATIONS,
    random_state=RANDOM_STATE,
    update_every=UPDATE_EVERY,
    eval_every=EVAL_EVERY,
    passes=PASSES,
    alpha=ALPHA,
    eta=ETA,
    per_word_topics=PER_WORD_TOPICS,
)

# # get array of topic vectors
# topic_vectors = lda.get_topics()

# get list of topic words and weights
topics = lda.show_topics(num_topics=N_TOPICS, num_words=N_WORDS, formatted=False)

topic_df = pd.concat(
    [ pd.DataFrame(words, columns=["word", "weight"],
        index=pd.Index([n+1]*N_WORDS, name="topic_number"))
    for n, words in topics ]
)


# export
topic_df.to_csv(export_fname, sep="\t", encoding="utf-8",
    na_rep="NA", index=True, float_format="%.6f")
