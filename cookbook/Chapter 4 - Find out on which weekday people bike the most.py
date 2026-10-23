
# %%
# Chapter 4 - Which Weekday Do People Bike the Most?
# D100 Exercise 3: Refactoring Pandas to Polars

from pathlib import Path

import pandas as pd
import polars as pl
import matplotlib.pyplot as plt


# %%
# Plotting settings

plt.style.use("ggplot")
plt.rcParams["figure.figsize"] = (15, 5)
plt.rcParams["font.family"] = "sans-serif"

pd.set_option("display.width", 5000)
pd.set_option("display.max_columns", 60)


# %%
# Set the dataset path relative to this script

DATA_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "bikes.csv"
)

WEEKDAY_NAMES = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]


# %%
# 1. Load the bicycle data
#
# The dataset contains daily counts of cyclists
# using bicycle paths in Montreal during 2012.


# Original Pandas implementation
bikes = pd.read_csv(
    DATA_PATH,
    sep=";",
    encoding="latin1",
    parse_dates=["Date"],
    dayfirst=True,
    index_col="Date",
)

print("\nPandas - First five rows:")
print(bikes.head())


# Polars implementation
pl_bikes = pl.read_csv(
    DATA_PATH,
    separator=";",
    encoding="latin1"
)

# Parse the Date column explicitly
pl_bikes = pl_bikes.with_columns(
    pl.col("Date").str.strptime(
        pl.Date,
        format="%d/%m/%Y"
    )
)

print("\nPolars - First five rows:")
print(pl_bikes.head())


# %%
# 2. Plot daily cyclists on Berri 1


# Pandas
fig, ax = plt.subplots(figsize=(12, 5))

bikes["Berri 1"].plot(
    ax=ax,
    color="steelblue"
)

ax.set_title("Daily Berri 1 Cyclists - Pandas")
ax.set_xlabel("Date")
ax.set_ylabel("Number of Cyclists")
ax.grid(True, alpha=0.3)

fig.tight_layout()


# Polars
fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    pl_bikes["Date"].to_list(),
    pl_bikes["Berri 1"].to_list(),
    color="seagreen"
)

ax.set_title("Daily Berri 1 Cyclists - Polars")
ax.set_xlabel("Date")
ax.set_ylabel("Number of Cyclists")
ax.grid(True, alpha=0.3)

fig.tight_layout()


# %%
# 3. Select the Berri bicycle path
#
# Pandas uses a Date index.
# Polars keeps Date as a regular column.


# Pandas
berri_bikes = bikes[["Berri 1"]].copy()

print("\nPandas - Berri 1:")
print(berri_bikes.head(5))


# Polars
pl_berri_bikes = pl_bikes.select(
    ["Date", "Berri 1"]
)

print("\nPolars - Berri 1:")
print(pl_berri_bikes.head(5))


# %%
# 4. Add a weekday column
#
# Pandas:
# Monday = 0, Sunday = 6
#
# Polars:
# Monday = 1, Sunday = 7


# Pandas
# Use day_of_week instead of the deprecated weekday

berri_bikes["weekday"] = (
    berri_bikes.index.day_of_week
)

print("\nPandas - Weekday column:")
print(berri_bikes.head(5))


# Polars
# Subtract 1 to match Pandas' numbering

pl_berri_bikes = pl_berri_bikes.with_columns(
    (
        pl.col("Date").dt.weekday() - 1
    ).alias("weekday")
)

print("\nPolars - Weekday column:")
print(pl_berri_bikes.head(5))


# %%
# 5. Group by weekday and sum cyclist counts


# Pandas
weekday_counts = (
    berri_bikes
    .groupby("weekday")["Berri 1"]
    .sum()
    .sort_index()
)

print("\nPandas - Cyclist counts by weekday:")
print(weekday_counts)


# Polars
pl_weekday_counts = (
    pl_berri_bikes
    .group_by("weekday")
    .agg(
        pl.col("Berri 1")
        .sum()
        .alias("total_cyclists")
    )
    .sort("weekday")
)

print("\nPolars - Cyclist counts by weekday:")
print(pl_weekday_counts)


# %%
# 6. Replace weekday numbers with names
#
# Polars does not use a Pandas-style index.
# We create a new string column instead.


# Pandas
weekday_counts.index = [
    WEEKDAY_NAMES[int(day)]
    for day in weekday_counts.index
]

print("\nPandas - Weekday names:")
print(weekday_counts)


# Polars
# FIX: Cast weekday integers to strings first.
# This prevents the integer-to-string conversion error.

pl_weekday_counts = (
    pl_weekday_counts
    .with_columns(
        pl.col("weekday")
        .cast(pl.String)
        .replace(
            {
                "0": "Monday",
                "1": "Tuesday",
                "2": "Wednesday",
                "3": "Thursday",
                "4": "Friday",
                "5": "Saturday",
                "6": "Sunday",
            }
        )
        .alias("weekday_name")
    )
    .select(
        [
            "weekday",
            "weekday_name",
            "total_cyclists",
        ]
    )
)

print("\nPolars - Weekday names:")
print(pl_weekday_counts)


# %%
# 7. Plot weekday totals using Pandas


fig, ax = plt.subplots(figsize=(12, 6))

weekday_counts.plot(
    kind="bar",
    ax=ax,
    color="steelblue"
)

ax.set_title(
    "Total Berri 1 Cyclists by Weekday - Pandas"
)

ax.set_xlabel("Weekday")
ax.set_ylabel("Total Number of Cyclists")

plt.setp(
    ax.get_xticklabels(),
    rotation=45,
    ha="right"
)

ax.grid(
    axis="y",
    alpha=0.3
)

fig.tight_layout()


# %%
# 8. Plot weekday totals using Polars
#
# Polars calculates the data.
# Matplotlib creates the chart.


weekdays = (
    pl_weekday_counts["weekday_name"].to_list()
)

cyclist_counts = (
    pl_weekday_counts["total_cyclists"].to_list()
)

fig, ax = plt.subplots(figsize=(12, 6))

ax.bar(
    weekdays,
    cyclist_counts,
    color="seagreen"
)

ax.set_title(
    "Total Berri 1 Cyclists by Weekday - Polars"
)

ax.set_xlabel("Weekday")
ax.set_ylabel("Total Number of Cyclists")

plt.setp(
    ax.get_xticklabels(),
    rotation=45,
    ha="right"
)

ax.grid(
    axis="y",
    alpha=0.3
)

fig.tight_layout()


# %%
# 9. Display all charts

plt.show()


# %%
# 10. Final verification
#
# Check that Pandas and Polars produce
# equivalent results.


print("\n--- Chapter 4 Verification ---")

print("\nPandas DataFrame shape:")
print(bikes.shape)

print("\nPolars DataFrame shape:")
print(pl_bikes.shape)

# Polars has one extra column because it
# retains Date instead of using an index.

assert bikes.shape[0] == pl_bikes.height, (
    "Pandas and Polars row counts differ!"
)

# Compare cyclist totals for each weekday

pandas_totals = weekday_counts.to_dict()

polars_totals = dict(
    zip(
        pl_weekday_counts["weekday_name"].to_list(),
        pl_weekday_counts["total_cyclists"].to_list()
    )
)

for day in WEEKDAY_NAMES:

    assert pandas_totals[day] == polars_totals[day], (
        f"Different cyclist counts for {day}"
    )

print("\nAll verification checks passed!")


# %%
# 11. Identify the busiest weekday


busiest_day = (
    pl_weekday_counts
    .sort(
        "total_cyclists",
        descending=True
    )
    .head(1)
)

print("\nBusiest weekday for cycling:")

print(
    busiest_day.select(
        ["weekday_name", "total_cyclists"]
    )
)

print(
    "\nChapter 4 refactoring completed successfully!"
)
