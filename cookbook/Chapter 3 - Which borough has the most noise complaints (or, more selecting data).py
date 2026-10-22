
# %%
# Chapter 3 - Which Borough Has the Most Noise Complaints?
# D100 Exercise 3: Refactoring Pandas to Polars

from pathlib import Path

import pandas as pd
import polars as pl
import matplotlib.pyplot as plt


# %%
# Plotting settings

plt.style.use("ggplot")
plt.rcParams["figure.figsize"] = (15, 5)

pd.set_option("display.width", 5000)
pd.set_option("display.max_columns", 60)


# %%
# Set the dataset path relative to this script

DATA_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "311-service-requests.csv"
)


# %%
# 1. Loading NYC 311 service requests
#
# The dataset contains mixed data types.
# Read all columns as strings to avoid type
# inference problems.


# Original Pandas implementation
complaints = pd.read_csv(
    DATA_PATH,
    dtype="unicode"
)

print("\nPandas - Dataset loaded:")
print(complaints.shape)


# Polars implementation
pl_complaints = pl.read_csv(
    DATA_PATH,
    infer_schema=False
)

print("\nPolars - Dataset loaded:")
print(pl_complaints.shape)


# %%
# 2. Selecting the first five rows


# Pandas
print("\nPandas - First five rows:")
print(complaints.head(5))


# Polars
print("\nPolars - First five rows:")
print(pl_complaints.head(5))


# %%
# 3. Selecting only noise complaints
#
# We want complaints where the Complaint Type
# equals "Noise - Street/Sidewalk".


# Original Pandas implementation
noise_complaints = complaints[
    complaints["Complaint Type"]
    == "Noise - Street/Sidewalk"
]

print("\nPandas - First three noise complaints:")
print(noise_complaints.head(3))


# Polars implementation
pl_noise_complaints = pl_complaints.filter(
    pl.col("Complaint Type")
    == "Noise - Street/Sidewalk"
)

print("\nPolars - First three noise complaints:")
print(pl_noise_complaints.head(3))


# %%
# 4. Combining multiple filtering conditions
#
# Select noise complaints located in Brooklyn.


# Pandas
is_noise = (
    complaints["Complaint Type"]
    == "Noise - Street/Sidewalk"
)

in_brooklyn = (
    complaints["Borough"] == "BROOKLYN"
)

brooklyn_noise = complaints[
    is_noise & in_brooklyn
]

print("\nPandas - Brooklyn noise complaints:")
print(brooklyn_noise.head(5))


# Polars
#
# Polars uses expressions through pl.col().
# The & operator combines the two conditions.

pl_brooklyn_noise = pl_complaints.filter(
    (
        pl.col("Complaint Type")
        == "Noise - Street/Sidewalk"
    )
    &
    (
        pl.col("Borough")
        == "BROOKLYN"
    )
)

print("\nPolars - Brooklyn noise complaints:")
print(pl_brooklyn_noise.head(5))


# %%
# 5. Selecting specific columns after filtering
#
# We only want:
# - Complaint Type
# - Borough
# - Created Date
# - Descriptor


# Pandas
selected_columns = [
    "Complaint Type",
    "Borough",
    "Created Date",
    "Descriptor"
]

print("\nPandas - Selected Brooklyn columns:")
print(
    brooklyn_noise[selected_columns].head(10)
)


# Polars
pl_selected_brooklyn = (
    pl_complaints
    .filter(
        (
            pl.col("Complaint Type")
            == "Noise - Street/Sidewalk"
        )
        &
        (
            pl.col("Borough")
            == "BROOKLYN"
        )
    )
    .select(selected_columns)
    .head(10)
)

print("\nPolars - Selected Brooklyn columns:")
print(pl_selected_brooklyn)


# %%
# 6. Which borough has the most noise complaints?
#
# Count noise complaints by borough.


# Pandas
noise_complaint_counts = (
    noise_complaints["Borough"]
    .value_counts()
)

print("\nPandas - Noise complaints by borough:")
print(noise_complaint_counts)


# Polars
pl_noise_counts = (
    pl_noise_complaints
    .filter(
        pl.col("Borough").is_not_null()
    )
    .group_by("Borough")
    .agg(
        pl.len().alias("noise_count")
    )
    .sort(
        "noise_count",
        descending=True
    )
)

print("\nPolars - Noise complaints by borough:")
print(pl_noise_counts)


# %%
# 7. Count all complaints by borough
#
# This is needed to calculate the proportion
# of complaints relating to street noise.


# Pandas
complaint_counts = (
    complaints["Borough"]
    .value_counts()
)

print("\nPandas - Total complaints by borough:")
print(complaint_counts)


# Polars
pl_total_counts = (
    pl_complaints
    .filter(
        pl.col("Borough").is_not_null()
    )
    .group_by("Borough")
    .agg(
        pl.len().alias("total_count")
    )
    .sort(
        "total_count",
        descending=True
    )
)

print("\nPolars - Total complaints by borough:")
print(pl_total_counts)


# %%
# 8. Calculate normalized noise complaint rates
#
# Divide the number of street noise complaints
# by the total complaints in each borough.
#
# This allows a comparison that accounts for
# differences in total complaint volumes.


# Pandas
noise_ratios = (
    noise_complaint_counts
    / complaint_counts.astype(float)
)

print("\nPandas - Normalized noise complaints:")
print(noise_ratios)


# Polars
#
# Join the two count tables using Borough.
# Then calculate the ratio using an expression.

pl_noise_ratios = (
    pl_noise_counts
    .join(
        pl_total_counts,
        on="Borough",
        how="inner"
    )
    .with_columns(
        (
            pl.col("noise_count")
            / pl.col("total_count")
        ).alias("noise_ratio")
    )
    .sort(
        "noise_ratio",
        descending=True
    )
)

print("\nPolars - Normalized noise complaints:")
print(pl_noise_ratios)


# %%
# 9. Plot normalized noise complaints using Pandas


fig, ax = plt.subplots(figsize=(12, 6))

noise_ratios.plot(
    kind="bar",
    ax=ax,
    color="steelblue"
)

ax.set_title(
    "Noise Complaints by Borough (Normalized) - Pandas"
)

ax.set_xlabel("Borough")

ax.set_ylabel(
    "Ratio of Noise Complaints to Total Complaints"
)

plt.setp(
    ax.get_xticklabels(),
    rotation=45,
    ha="right"
)

fig.tight_layout()


# %%
# 10. Plot normalized noise complaints using Polars
#
# Polars performs the calculations.
# Matplotlib handles the visualization.


boroughs = (
    pl_noise_ratios["Borough"].to_list()
)

ratios = (
    pl_noise_ratios["noise_ratio"].to_list()
)

fig, ax = plt.subplots(figsize=(12, 6))

ax.bar(
    boroughs,
    ratios,
    color="seagreen"
)

ax.set_title(
    "Noise Complaints by Borough (Normalized) - Polars"
)

ax.set_xlabel("Borough")

ax.set_ylabel(
    "Ratio of Noise Complaints to Total Complaints"
)

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
# 11. Display both charts

plt.show()


# %%
# 12. Final verification
#
# Verify that Pandas and Polars produce
# equivalent results.


print("\n--- Chapter 3 Verification ---")

print("\nPandas DataFrame shape:")
print(complaints.shape)

print("\nPolars DataFrame shape:")
print(pl_complaints.shape)

# Check that both libraries loaded the same data
assert complaints.shape == pl_complaints.shape, (
    "Pandas and Polars DataFrame shapes differ!"
)

# Check the number of noise complaints
assert (
    len(noise_complaints)
    == pl_noise_complaints.height
), "Noise complaint counts do not match!"

# Check the number of Brooklyn noise complaints
assert (
    len(brooklyn_noise)
    == pl_brooklyn_noise.height
), "Brooklyn noise complaint counts do not match!"

# Compare normalized ratios for each borough
pl_ratios_dict = dict(
    zip(
        pl_noise_ratios["Borough"].to_list(),
        pl_noise_ratios["noise_ratio"].to_list()
    )
)

for borough, pandas_ratio in noise_ratios.dropna().items():

    polars_ratio = pl_ratios_dict[borough]

    assert abs(pandas_ratio - polars_ratio) < 1e-10, (
        f"Different ratio for {borough}"
    )

print("\nAll verification checks passed!")

print("\nBorough with the most noise complaints:")

print(
    pl_noise_counts.head(1)
)

print("\nBorough with the highest normalized noise ratio:")

print(
    pl_noise_ratios.select(
        ["Borough", "noise_ratio"]
    ).head(1)
)

print(
    "\nChapter 3 refactoring completed successfully!"
)

