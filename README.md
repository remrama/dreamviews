# dreamviews

Turning the public [DreamViews dream journal](https://www.dreamviews.com/blogs) into a usable dataset.

This repository holds the code for curating the corpus, describing it, compiling the manuscript. The [derived corpus](https://zenodo.org/records/19161756) is publicly available and archived on Zenodo.

- `scripts/` - Data collection and analysis
- `manuscript/` - Manuscript compiling
- `output/` - Files exported from data analysis and used in manuscript compilation

To go from nothing to manuscript:

```bash
cd scripts
conda env create --file environment.yml
python runall.py --compile
```