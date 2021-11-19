"""
User count per country
save figure and table
"""
import os
import numpy as np
import pandas as pd
import config as c

import plotly.graph_objects as go

import_fname = os.path.join(c.DATA_DIR, "derivatives", "users-clean.tsv")
export_fname_plot  = os.path.join(c.DATA_DIR, "results", "describe-locations.png")
export_fname_table = os.path.join(c.DATA_DIR, "results", "describe-locations.tsv")


df = pd.read_csv(import_fname, sep="\t", encoding="utf-8")

# get a count per country
df_counts = df["country"].fillna("unstated"
    ).value_counts(
    ).rename("n_users"
    ).rename_axis("country"
    ).reset_index(drop=False)

# add a log count
df_counts["n_users-log10"] = df_counts["n_users"].apply(np.log10)

# generate string to show how many users are not included
unstated_n = df.country.isna().sum()
unstated_pct = df.country.isna().mean() * 100
unstated_txt = f"{unstated_n} ({unstated_pct:.0f}%) did not report location"



# draw
fig = go.Figure(
    data=go.Choropleth(
        locations=df_counts["country"],
        z=df_counts["n_users-log10"],
        text=df_counts["country"].tolist(),
        hovertemplate="%{text}",
        colorscale="Viridis",
        marker_line_color="darkgray",
        marker_line_width=.5,
        colorbar_title="# of users",
        colorbar_tickvals=[0,1,2,3],
        colorbar_ticktext=["1","10","100","1000"],
        colorbar_thickness=10,
        colorbar_thicknessmode="pixels",
        colorbar_len=200,
        colorbar_lenmode="pixels",
        colorbar_yanchor="top",
        colorbar_y=1,
        colorbar_tickprefix="",
        colorbar_ticks="outside",
    ),
    layout=go.Layout(
        height=300,
        margin=dict(r=0, t=0, l=0, b=0),
        geo=dict(
            projection_type="natural earth",
            scope="world",
            # resolution=50,
            countrycolor="darkgray",
            countrywidth=.5,
            coastlinecolor="darkgray",
            coastlinewidth=.5,
            framecolor="black",
            framewidth=.5,
            bgcolor="white",
            showcoastlines=True,
            showcountries=True,
            showsubunits=False,
            showlakes=False,
            showframe=True,
        ),
        font=dict(
            family="Sans-Serif",
            size=10,
            color="black",
        ),
        annotations=[dict(
            x=1.1,
            y=.02,
            xref="paper",
            yref="paper",
            text=unstated_txt,
            showarrow=False
        )]
    )
)



### export

fig.write_image(export_fname_plot, scale=5)

df_counts.to_csv(export_fname_table, sep="\t", encoding="utf-8",
    index=False, float_format="%.2f")
