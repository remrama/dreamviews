# dreamviews_ds

Turning the public [DreamViews Dream Journal](https://www.dreamviews.com/blogs/) into a usable dataset. This repo is strictly for collecting, cleaning, describing, and validating the dataset. It's used in other projects and can be found [here](https://media.giphy.com/media/St0Nd0Qt4WNMLy29vi/giphy.gif).

---

See `config.py` for directory info and other configuration stuff. The `DATA_DIR` in `config.py` needs to be set first to match the desired data output location.

The code filenames start with one of `init`, `collect`, `convert`, `clean`, `extract`, `describe`, `validate`, and this precursor explains which of the stages the file is used for.

Raw data from the web scraping is in `DATA_DIR/source`, "middle-ground" output, like the cleaned posts and aggregated users or raw LIWC output are in `DATA_DIR/derivatives`, and any plots or statistics are output to `DATA_DIR/results`.


### A few preliminary scripts.

```shell
# create the relevant subfolders of the data directory (specified in config.py)
python init-generate_data_dirs.py   # ==> DATA_DIR/source/
                                    # ==> DATA_DIR/derivatives/
                                    # ==> DATA_DIR/results/
python init-generate_liwc_dict.py   # ==> DATA_DIR/dictionaries/myliwc.dic
```


### COLLECT, CONVERT, and CLEAN data

The first thing we do is grab all the data and tidy it up a bit (without being very invasive or restrictive).

1. Collect/scrape the DreamViews journal and relevant public profiles.
2. Convert the html files to tsv, super minimal cleaning just to prevent errors.
3. Clean the raw tsv data a bit for final/usable post and user files.
4. Extract a principled/controlled subset of the data for manual annotations.

```shell
# collect raw dream journal posts as html files
# (note we also get raw user html files, but we need to convert the
#  raw posts first because we grab only the users who contribute posts)
python collect-posts.py             # ==> DATA_DIR/source/dreamviews-posts.zip
```

Need to jump out and insert the timestamp of day data was collected in the `config.py` file. This is mildly annoying, but the most recent blog/journal posts are stamped as coming from "today" or "yesterday", so those need a reference. Should be a `YYYY-MM-DD` string. This is also used in both the conversion scripts, including users, so make sure they are collected on the same day.

```shell
# convert raw html dream journal posts to a tsv (nothing cleaned yet)
# and get the names of user profiles to pull
python convert-posts.py             # ==> DATA_DIR/derivatives/posts-raw.tsv
                                    # ==> DATA_DIR/derivatives/users-anon_key.json

# now, with the usernames generated, we can collect relevant user profiles
python collect-users.py             # ==> DATA_DIR/source/dreamviews-users.zip

# clean the dream journal posts, exporting a tsv and also a folder of text files
python clean-posts.py               # ==> DATA_DIR/derivatives/posts-clean.tsv
python tsv2txt-posts.py             # ==> DATA_DIR/derivatives/posts/<post_id>.txt

# convert and clean the users
python convert-users.py             # ==> DATA_DIR/derivatives/users-raw.csv
python clean-users.py               # ==> DATA_DIR/derivatives/users-clean.csv

# grab a tightly controlled subset of the data for manual stuff
python extract-subset.py            # ==> DATA_DIR/derivatives/posts-subset.csv
```


### DESCRIBE data

Export plots and summary statistics describing the dataset.

1. Count how much data there is.
2. Describe the breakdown of lucid/non-lucid labels.
3. Explore the word counts before and after lemmatization.
4. Describe the demographics.

```shell
##### Visualize the amount of data there is.

# frequency of posts over time
python describe-timecourse.py       # ==> DATA_DIR/results/describe-timecourse.png

# frequency of posts per user
python describe-usercount.py         # ==> DATA_DIR/results/describe-usercount.png

# post length (word counts)
python describe-wordcount.py        # ==> DATA_DIR/results/describe-wordcount.png


##### Visualize the user demographics.

# reported gender and age
python describe-demographics.py     # ==> DATA_DIR/results/describe-demographics.png

# reported location/country
python describe-location.py         # ==> DATA_DIR/results/describe-location.tsv
                                    # ==> DATA_DIR/results/describe-location.png


##### Each post can have "category" or "tag" labels.
##### Visualize the amount of posts with from each relevant category.

# generate a tsv for top categories and labels
python describe-toplabels.py        # ==> DATA_DIR/results/describe-topcategories.tsv
                                    # ==> DATA_DIR/results/describe-toptags.tsv

# lucid/non-lucid/nightmare overlap
python describe-categorycounts.py   # ==> DATA_DIR/results/describe-categorycounts.png

# number of participants with both lucid and non-lucid posts
python describe-categorypairs.py    # ==> DATA_DIR/results/describe-categorypairs.tsv
                                    # ==> DATA_DIR/results/describe-categorypairs.png
```


### VALIDATE data

Show that overall the posts "look like" dreams, and then the lucid and non-lucid posts are differentiable in predictable ways based on previous literature.

1. Show that LDs and non-LDs can be distinguished with language with a BoW classifier.
2. Visualize a general word difference between LD and non-LD.
3. Show that LDs have more insight and agency in LIWC (lucidity and control, respectively).
4. Explore LD language with topic modeling.

```shell
# classifier
python validate-classifier_run.py   # ==> DATA_DIR/derivatives/validate-classifier.npz
python validate-classifier.py       # ==> DATA_DIR/results/validate-classifier.png

# words that differentiate lucid and non-lucid
python validate-wordshift_run.py    # ==> DATA_DIR/derivatives/validate-wordshift.tsv
python validate-wordshift.py        # ==> DATA_DIR/results/validate-wordshift.tsv
                                    # ==> DATA_DIR/results/validate-wordshift.png

# run LIWC to get effects at the category and word levels
python validate-liwc_run.py --words # ==> DATA_DIR/derivatives/posts-liwc.tsv
                                    # ==> DATA_DIR/derivatives/posts-liwc_words-data.npz
                                    # ==> DATA_DIR/derivatives/posts-liwc_words-attr.npz

# plot total insight and agency effects LD vs non-LD
python validate-liwc.py             # ==> DATA_DIR/derivatives/validate-liwc.tsv
                                    # ==> DATA_DIR/results/validate-liwc.tsv
                                    # ==> DATA_DIR/results/validate-liwc.png

# plot individual word contributions for insight and agency effects LD vs non-LD
python validate-liwcwords_perms.py  # ==> DATA_DIR/results/validate-liwcwords.tsv
python validate-liwcwords.py        # ==> DATA_DIR/results/validate-liwcwords.png

# explore latent topic structure of lucid dreams
python validate-topicmodel_run.py   # ==> DATA_DIR/results/validate-topicmodel.tsv
python validate-topicmodel.py       # ==> DATA_DIR/results/validate-topicmodel.png
```
