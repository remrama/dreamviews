"""
Export a 2-column tsv of post_id and lemmatized post text
for subsequent analysis/validation steps (classifier and wordshift).
"""

import random

import spacy
from tqdm import tqdm

import config as c

random.seed(91)

EXPORT_STEM = "validate-lemmas"
export_path = c.derivatives_dir / f"{EXPORT_STEM}.tsv"

df = c.load_dreamviews_posts()

posts = df.set_index("post_id")["post_text"]

# Load spaCy model (used for named entity recognition)
# # Speed up spaCy by disabling some unncessary stuff
# SPACY_PIPE_DISABLES = ["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"]
SPACY_PIPE_DISABLES = []
nlp = spacy.load(c.SPACY_MODEL, disable=SPACY_PIPE_DISABLES)
nlp.add_pipe("merge_entities")  # So "John Paul" gets treated as a single entity


def lemmatize(x, shuffle=False, pos_remove_list=None):
    """Convert a spaCy doc to space-separate string of shuffled lemmas."""
    doc = nlp(x)
    if pos_remove_list is None:
        pos_remove_list = ["PROPN", "SMY"]
    token_list = []
    for token in doc:
        if (
            (token.is_alpha)
            and (len(token) >= 3)
            and (not token.like_email)
            and (not token.like_url)
            and (not token.like_num)
            and (not token.is_stop)
            and (not token.is_oov)
            and (token.pos_ not in pos_remove_list)
        ):
            token_list.append(token.lemma_.lower())
    if shuffle:
        token_list = random.sample(token_list, len(token_list))
    if token_list:
        return " ".join(token_list)
    return


tqdm.pandas(desc="Lemmatizing posts")
lemmas = posts.progress_apply(lemmatize).dropna().to_frame(name="post_lemmas")

lemmas.to_csv(export_path, sep="\t", encoding="ascii")
