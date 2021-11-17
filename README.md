# dreamviews_ds

Turning the public [DreamViews Dream Journal](https://www.dreamviews.com/blogs/) into a usable dataset. There's lots of descriptive output and such, but the final dataset, to be used for other projects and such, can be found [here](https://media.giphy.com/media/St0Nd0Qt4WNMLy29vi/giphy.gif).

### Part 1 - Data collection and cleaning
The first thing we do is grab all the data and tidy it up a bit (without being very invasive or restrictive).
1. collect/scrape the DreamViews journal and relevant public profiles
2. clean the raw data and apply minimal exclusion criteria
3. export plots and summary statistics describing the dataset
4. minor validation steps (e.g., lucid vs non-lucid word clouds)

### Part 2 - Manual annotations and validation
We also manually clean and annotate a subset of the dataset. This subset can be used for further validation, finer-grained analyses, and development of algorithms for automated detection of lucidity. See our custom [Dream Report Annotation Manual](https://d-re-a-m.readthedocs.io/) for annotation details.
1. extract a principled subset of the data
2. upload the data to tagtog for annotating
3. download and convert the tagtog data
4. visualize and analyze interrater reliability for annotations
5. minor results from annotations (e.g., temporal moment of lucidity)
6. validate user-defined lucidity against experimenter-defined lucidity
7. develop a classifier that can be used to determine lucid from non-lucid dreams and apply to the rest of the dataset (with probabilities saved out on final dataset)


