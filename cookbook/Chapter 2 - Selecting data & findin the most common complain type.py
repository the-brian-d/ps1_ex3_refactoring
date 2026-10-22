
# %%
# Chapter 2 - Selecting Data and Finding the Most Common Complaint Type
# D100 Exercise 3: Refactoring Pandas to Polars

from pathlib import Path

import pandas as pd
import polars as pl
import matplotlib.pyplot as plt


# %%
# Set the path relative to this Python script

DATA_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "311-service-requests.csv"
)


# %%
# 1. Reading the dataset
#
# We use a subset of New York City's 311 service
# requests dataset.
#
# The dataset contains mixed data types.
# Some ZIP codes contain missing values or hyphens,
# so we initially read all columns as strings.


# Original Pandas implementation
complaints = pd.read_csv(
    DATA_PATH,
    dtype="unicode"
)

print("\nPandas - First five rows:")
print(complaints.head())


# Polars implementation
#
# infer_schema=False tells Polars to read all
# columns as strings instead of inferring types.

pl_complaints = pl.read_csv(
    DATA_PATH,
    infer_schema=False
)

print("\nPolars - First five rows:")
print(pl_complaints.head())


# %%
# 2. Selecting a single column
#
# Select the "Complaint Type" column.


# Pandas
print("\nPandas - Complaint Type:")
print(complaints["Complaint Type"].head())


# Polars
print("\nPolars - Complaint Type:")
print(
    pl_complaints.select("Complaint Type").head()
)


# %%
# 3. Selecting the first five rows


# Pandas
print("\nPandas - First five rows:")
print(complaints[:5])


# Polars
print("\nPolars - First five rows:")
print(pl_complaints.head(5))


# %%
# 4. Selecting the first five rows of one column
#
# Combine column selection and row selection.


# Pandas
print("\nPandas - First five complaint types:")
print(complaints["Complaint Type"][:5])


# Polars
print("\nPolars - First five complaint types:")
print(
    pl_complaints
    .select("Complaint Type")
    .head(5)
)


# %%
# 5. Selecting multiple columns
#
# Select Complaint Type and Borough.


# Pandas
print("\nPandas - Multiple columns:")
print(
    complaints[
        ["Complaint Type", "Borough"]
    ].head()
)


# Polars
print("\nPolars - Multiple columns:")
print(
    pl_complaints.select(
        ["Complaint Type", "Borough"]
    ).head()
)


# %%
# 6. Finding the most common complaint types
#
# Count how often each complaint type appears.


# Original Pandas implementation
complaint_counts = (
    complaints["Complaint Type"]
    .value_counts()
)

print("\nPandas - Top 10 complaint types:")
print(complaint_counts.head(10))


# Polars implementation
#
# 1. Remove missing complaint types
# 2. Group rows by complaint type
# 3. Count rows in each group
# 4. Sort by count in descending order

pl_complaint_counts = (
    pl_complaints
    .filter(
        pl.col("Complaint Type").is_not_null()
    )
    .group_by("Complaint Type")
    .agg(
        pl.len().alias("count")
    )
    .sort(
        "count",
        descending=True
    )
)

print("\nPolars - Top 10 complaint types:")
print(pl_complaint_counts.head(10))


# %%
# 7. Plotting the top 10 complaint types
#
# First, create the original Pandas visualisation.


# Pandas
fig, ax = plt.subplots(figsize=(12, 6))

complaint_counts.head(10).plot(
    kind="bar",
    ax=ax,
    color="steelblue"
)

ax.set_title(
    "Top 10 Complaint Types - Pandas"
)

ax.set_xlabel("Complaint Type")
ax.set_ylabel("Number of Complaints")

plt.setp(
    ax.get_xticklabels(),
    rotation=45,
    ha="right"
)

fig.tight_layout()


# %%
# 8. Plotting the top 10 complaint types with Polars
#
# Polars handles the data manipulation.
# Matplotlib creates the visualisation.


# Select the top 10 most common complaint types
pl_top_10 = pl_complaint_counts.head(10)

# Extract the labels and counts
complaint_types = (
    pl_top_10["Complaint Type"].to_list()
)

counts = (
    pl_top_10["count"].to_list()
)

# Create the chart
fig, ax = plt.subplots(figsize=(12, 6))

ax.bar(
    complaint_types,
    counts,
    color="seagreen"
)

ax.set_title(
    "Top 10 Complaint Types - Polars"
)

ax.set_xlabel("Complaint Type")
ax.set_ylabel("Number of Complaints")

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
# 9. Display both charts

plt.show()


# %%
# 10. Final verification
#
# Check that Pandas and Polars loaded the same
# number of observations and columns.


print("\n--- Chapter 2 Verification ---")

print("\nPandas DataFrame shape:")
print(complaints.shape)

print("\nPolars DataFrame shape:")
print(pl_complaints.shape)

print("\nNumber of distinct complaint types:")

print(
    pl_complaint_counts.height
)

print("\nMost common complaint type:")

print(
    pl_complaint_counts.head(1)
)

# Verify matching shapes
assert complaints.shape == pl_complaints.shape, (
    "Pandas and Polars DataFrame shapes differ!"
)

# Verify matching complaint counts
assert (
    complaint_counts.sum()
    == pl_complaint_counts["count"].sum()
), "Pandas and Polars complaint totals differ!"

print(
    "\nChapter 2 refactoring completed successfully!"
)
