# dreamviews

Turning the public [DreamViews dream journal](https://www.dreamviews.com/blogs) into a usable dataset.

This repository holds the code for scraping and analyzing the data (`scripts/`) and compiling the manuscript (`manuscript/`).

- `scripts/` - Data collection and analysis
- `manuscript/` - Manuscript compiling
- `output/` - Files exported from data analysis and used in manuscript compilation

To go from nothing to manuscript:

```bash
cd scripts
conda env create --file environment.yml
python runall.py
cd ../manuscript
make
```