"""
Generate a custom LIWC dictionary file.
Not creating new dictionaries or categories,
just taking a subset of the categories
and adding a few others (agency) to one file
so I can just run this once. Also I need to
convert the .xlsx file to a .dic file anyways
so might as well add the agency stuff here.
"""
import os
import csv
import pandas as pd
import liwc
import config as c


KEEP_CATEGORIES = [ # stuff from LIWC2015 to keep (Agency will be in there by default)
    "I",
    "We",
    "You",
    "They",
    "Affect",
    "Posemo",
    "Negemo",
    "Anx",
    "Anger",
    "Sad",
    "Social",
    "Family",
    "Friend",
    "CogProc",
    "Insight",
    "Percept",
    "See",
    "Hear",
    "Feel",
    "Drives",
    "Affiliation",
    "Achieve",
    "Reward",
    "Risk",
    "FocusPast",
    "FocusPresent",
    "FocusFuture",
    "Work",
    "Relig",
    "Death",
]


import_fname1 = os.path.join(c.DATA_DIR, "dictionaries", "LIWC2015 dictionary poster.xlsx")
import_fname2 = os.path.join(c.DATA_DIR, "dictionaries", "a_AgencyCommunion.dic")
export_fname = os.path.join(c.DATA_DIR, "dictionaries", "myliwc.dic")

# load in the full 2015 dictionary via excel file
df = pd.read_excel(import_fname1, header=3)
df = df.iloc[1:].reset_index(drop=True)
# categories = [ c.split("\n")[1] for c in df.columns if "\n" in c ]

# generate a dictionary dictionary! :///
# each key, item pair is a category_name, list_of_words
dictionary = {}
for col in df:
    if "\n" in col: # reset
        category = col.split("\n")[1]
        words = df[col].dropna().tolist()
        dictionary[category] = words
    else:
        more_words = df[col].dropna().tolist()
        dictionary[category].extend(more_words)


# remove unwanted categories
dictionary = { c: wlist for c, wlist in dictionary.items() if c in KEEP_CATEGORIES }


# load in the Agency stuff. already dic file so easiest to use the nice liwc package
lexicon, category_names = liwc.read_dic(import_fname2)
# the lexicon loads in a { word: category_list }
# format as opposed to { category: word_list }
# so get to my way which is worse on memory but it is what it is
agency_word_list = [ w for w, c in lexicon.items() if "agency" in c ]

# add it to the other categories
dictionary["agency"] = agency_word_list

# lowercase all the category names
dictionary = { c.lower(): wlist for c, wlist in dictionary.items() }

# each category needs a number, that's how the LIWC dic files work
category_ids = { d: i+1 for i, d in enumerate(dictionary.keys()) }

# get a list of all the words of all categories, the whole vocabulary
vocabulary = sorted(set([ w for wlist in dictionary.values() for w in wlist ]))
# vocabulary = sorted(set(filter(lambda v: v==v, df.values.flat)))


# write to file
with open(export_fname, "wt", encoding="utf-8", newline="") as f:
    writer = csv.writer(f, delimiter="\t")

    # write the header with all the category IDs
    writer.writerow("%")
    for cat, cat_id in category_ids.items():
        writer.writerow([cat_id, cat])
    writer.writerow("%")

    # write all the vocabulary words and their corresponding category IDs
    for word in vocabulary:
        row_data = [word]
        # find the categories this word is in and add them to row
        for cat, word_list in dictionary.items():
            if word in word_list:
                cat_id = category_ids[cat]
                row_data.append(cat_id)
        writer.writerow(row_data)
