# dreamviews_ds

Turning the public [DreamViews Dream Journal](https://www.dreamviews.com/blogs/) into a usable dataset. There's lots of descriptive output and such, but the final dataset, to be used for other projects and such, can be found [here](https://media.giphy.com/media/St0Nd0Qt4WNMLy29vi/giphy.gif).

### Part 1 - Data collection and cleaning
The first thing we do is grab all the data and tidy it up a bit (without being very invasive or restrictive).
1. **collect**/scrape the DreamViews journal and relevant public profiles
2. **convert** the html files to tsv, super minimal cleaning just to prevent errors
3. **clean** the raw tsv data a bit for final/usable post and user files
4. export plots and summary statistics describing the dataset
5. minor validation steps (e.g., lucid vs non-lucid word clouds)

### Part 2 - Manual annotations and validation
We also manually clean and annotate a subset of the dataset. This subset can be used for further validation, finer-grained analyses, and development of algorithms for automated detection of lucidity. See our custom [Dream Report Annotation Manual](https://d-re-a-m.readthedocs.io/) for annotation details.
1. extract a principled subset of the data
2. upload the data to tagtog for annotating
3. download and convert the tagtog data
4. visualize and analyze interrater reliability for annotations
5. minor results from annotations (e.g., temporal moment of lucidity)
6. validate user-defined lucidity against experimenter-defined lucidity
7. develop a classifier that can be used to determine lucid from non-lucid dreams and apply to the rest of the dataset (with probabilities saved out on final dataset)


## Linear code layout

See `config.py` for directory info and other configuration stuff.

Run `init_data_directory_structure.py` before anything else, after manually setting the `DATA_DIR` in `config.py` to match your desired data output location.


### Collect, convert, and clean data

```bash
# collect raw dream journal posts as html files
# (note we also get raw user html files, but we need to convert the
#  raw posts first because we grab only the users who contribute posts)
python collect-posts.py             # ==> DATA_DIR/source/dreamviews-posts.zip
```

Need to jump out and insert the timestamp of day data was collected in the `config.py` file. This is mildly annoying, but the most recent blog/journal posts are stamped as coming from "today" or "yesterday", so those need a reference. Should be a `YYYY-MM-DD` string. This is also used in both the conversion scripts, including users, so make sure they are collected on the same day.

```bash
# convert raw dream journal posts to minimally-cleaned text files
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
```


### Describe data

```bash
# frequency of posts over time
python describe-timecourse.py       # ==> DATA_DIR/results/describe-timecourse.png/eps

# lucid/non-lucid/nightmare overlap
python describe-category_venn.py    # ==> DATA_DIR/results/describe-category_venn.png/eps
                                    # ==> DATA_DIR/results/describe-category_venn.tsv

# number of participants with both lucid and non-lucid posts
python describe-category_pair.py    # ==> DATA_DIR/results/describe-category_pair.png/eps
                                    # ==> DATA_DIR/results/describe-category_pair.tsv

# reported gender and age
python describe-demographics.py     # ==> DATA_DIR/results/describe-demographics.png/eps

# reported location/country
python describe-locations.py        # ==> DATA_DIR/results/describe-locations.png
                                    # ==> DATA_DIR/results/describe-locations.tsv

# number of posts per user
python describe-userfreq.py         # ==> DATA_DIR/results/describe-userfreq.png/eps

# post length (word counts)
python describe-wordcount.py        # ==> DATA_DIR/results/describe-wordcount.png/eps

# generate a tsv for top categories and labels
python describe-labels.py           # ==> DATA_DIR/results/describe-labels_categories.tsv
                                    # ==> DATA_DIR/results/describe-labels_tags.tsv
```

### Validate/explore the lucid dreams with some non-lucid comparisons

```bash
# words that differentiate lucid and non-lucid
python analysis-wordshift_perms.py  # ==> DATA_DIR/results/validate-wordshift_perms.tsv
python analysis-wordshift_plot.py   # ==> DATA_DIR/results/validate-wordshift_plot.png/eps
                                    # ==> DATA_DIR/results/validate-wordshift_stats.tsv

# draw wordclouds, never to be looked at
python analysis-wordcloud.py        # ==> DATA_DIR/results/validate-wordcloud.png/eps

# generate a custom LIWC dictionary and run LIWC analysis
python liwc-generate_mydic.py       # ==> DATA_DIR/dictionaries/myliwc.dic
python analysis-liwc_scores.py -t   # ==> DATA_DIR/derivatives/posts-liwc.tsv
                                    # ==> DATA_DIR/derivatives/posts-liwc_tokens.npz
                                    # ==> DATA_DIR/derivatives/posts-liwc_tokens.npy
python analysis-liwc_stats.py       # ==> DATA_DIR/results/analysis-liwc.tsv
python analysis-liwc_stats_tokens.py # ==> DATA_DIR/results/analysis-liwc_tokens.tsv
# plot summary of all categories (not interesting)
python analysis-liwc_plot.py        # ==> DATA_DIR/results/analysis-liwc.png/eps
# focus on subset of relevant categories and their word contributions (interesting)
python analysis-liwc_plot_tokens.py # ==> DATA_DIR/results/analysis-liwc_tokens.png/eps

# LIWC subset -- consider showing word contribution plots
# topic modeling
# word count comparison
```


### post-tagtog annotation analysis section

A total WiP, but a few things for now.

Note that I'm using an old file from mannheim_dv results.
It's accurate, but I still need to get a final conversion
script and it will provide output slightly different.

But I can use this for results now and make changes later.
See previous conversion scripts in content_ld and mannheim_dv.

```bash
######## need something that exports
# ==> DATA_DIR/derivatives/posts-annotations.tsv

# draw a distribution showing when the first moment of lucidity typically occurs
# while also outputing a new file that has text
# before/after that moment (2 rows per post)
python annotations-lucidmoment.py   # ==> DATA_DIR/derivatives/posts-annotations_lucidprepost.tsv
                                    # ==> DATA_DIR/results/annotations-lucidmoment.tsv
                                    # ==> DATA_DIR/results/annotations-lucidmoment.png/eps
```