
# %%
# Chapter 7 - Cleaning Up Messy Data
# Refactoring Pandas code to Polars

from pathlib import Path

import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import numpy as np

# Plot settings
plt.style.use("ggplot")
plt.rcParams["figure.figsize"] = (15, 5)
plt.rcParams["font.family"] = "sans-serif"

# Locate the data file
DATA_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "311-service-requests.csv"
)


# %%
# 1. LOAD THE NYC 311 SERVICE REQUESTS DATA

# Pandas
requests = pd.read_csv(
    DATA_PATH,
    dtype="unicode"
)

print("\nPandas - First five rows:")
print(requests.head())


# Polars
pl_requests = pl.read_csv(
    DATA_PATH,
    infer_schema_length=0
)

print("\nPolars - First five rows:")
print(pl_requests.head())


# %%
# 2. EXAMINE UNIQUE ZIP CODES

# Pandas
print("\nPandas - Unique ZIP codes:")
print(requests["Incident Zip"].unique())


# Polars
print("\nPolars - Unique ZIP codes:")
print(
    pl_requests.select(
        pl.col("Incident Zip").unique()
    )
)


# %%
# 3. HANDLE MISSING VALUES
#
# Treat "NO CLUE", "N/A", and "0" as missing.
# Also treat empty strings as missing.
#
# Using keep_default_na=False gives us explicit control
# over the missing-value rules in Pandas.

na_values = ["NO CLUE", "N/A", "0", ""]

# Pandas
requests = pd.read_csv(
    DATA_PATH,
    na_values=na_values,
    keep_default_na=False,
    dtype={"Incident Zip": str}
)

print("\nPandas - ZIP codes after handling missing values:")
print(requests["Incident Zip"].unique())


# Polars
pl_requests = pl.read_csv(
    DATA_PATH,
    infer_schema_length=0,
    null_values=na_values
)

print("\nPolars - ZIP codes after handling missing values:")
print(
    pl_requests.select(
        pl.col("Incident Zip").unique()
    )
)


# %%
# 4. IDENTIFY ZIP CODES CONTAINING DASHES
#
# ZIP+4 codes contain a dash, for example 29616-0759.

# Pandas
rows_with_dashes = (
    requests["Incident Zip"]
    .str.contains("-", regex=False, na=False)
)

print("\nPandas - Number of ZIP codes containing dashes:")
print(rows_with_dashes.sum())

print("\nPandas - Example rows:")
print(requests[rows_with_dashes].head(10))


# Polars
pl_rows_with_dashes = pl_requests.filter(
    pl.col("Incident Zip")
    .str.contains("-", literal=True)
    .fill_null(False)
)

print("\nPolars - Number of ZIP codes containing dashes:")
print(pl_rows_with_dashes.height)

print("\nPolars - Example rows:")
print(
    pl_rows_with_dashes.select(
        ["Incident Zip", "Descriptor", "City"]
    ).head(10)
)


# %%
# 5. IDENTIFY LONG ZIP CODES

# Pandas
long_zip_codes = (
    requests["Incident Zip"].str.len() > 5
)

print("\nPandas - Long ZIP codes:")
print(
    requests.loc[
        long_zip_codes.fillna(False),
        "Incident Zip"
    ].unique()
)


# Polars
pl_long_zip_codes = pl_requests.filter(
    pl.col("Incident Zip").str.len_chars() > 5
)

print("\nPolars - Long ZIP codes:")
print(
    pl_long_zip_codes.select(
        pl.col("Incident Zip").unique()
    )
)


# %%
# 6. TRUNCATE ZIP CODES TO FIVE CHARACTERS

# Pandas
requests["Incident Zip"] = (
    requests["Incident Zip"].str.slice(0, 5)
)

print("\nPandas - ZIP codes after truncation:")
print(requests["Incident Zip"].head(10))


# Polars
pl_requests = pl_requests.with_columns(
    pl.col("Incident Zip")
    .str.slice(0, 5)
    .alias("Incident Zip")
)

print("\nPolars - ZIP codes after truncation:")
print(
    pl_requests.select("Incident Zip").head(10)
)


# %%
# 7. REMOVE INVALID "00000" ZIP CODES

# Pandas
print("\nPandas - Rows containing 00000:")
print(
    requests[
        requests["Incident Zip"] == "00000"
    ].head(10)
)

zero_zips = requests["Incident Zip"] == "00000"

requests.loc[zero_zips, "Incident Zip"] = np.nan


# Polars
print("\nPolars - Rows containing 00000:")
print(
    pl_requests.filter(
        pl.col("Incident Zip") == "00000"
    ).head(10)
)

pl_requests = pl_requests.with_columns(
    pl.when(pl.col("Incident Zip") == "00000")
    .then(pl.lit(None, dtype=pl.String))
    .otherwise(pl.col("Incident Zip"))
    .alias("Incident Zip")
)

print("\nPolars - ZIP codes after removing 00000:")
print(
    pl_requests.select("Incident Zip").head(10)
)


# %%
# 8. EXAMINE CLEANED UNIQUE ZIP CODES

# Pandas
unique_zips = (
    requests["Incident Zip"]
    .fillna("NaN")
    .astype(str)
    .unique()
)

unique_zips.sort()

print("\nPandas - Cleaned unique ZIP codes:")
print(unique_zips)


# Polars
pl_unique_zips = (
    pl_requests
    .select(
        pl.col("Incident Zip")
        .fill_null("NaN")
        .unique()
        .sort()
    )
)

print("\nPolars - Cleaned unique ZIP codes:")
print(pl_unique_zips)


# %%
# 9. FIND ZIP CODES OUTSIDE THE EXPECTED AREA
#
# For this exercise, we assume ZIP codes beginning
# with 0 or 1 are relatively close to New York.
#
# This is a simplified rule, not a reliable
# geographical classification.

# Pandas
zips = requests["Incident Zip"]

is_close = (
    zips.str.startswith("0")
    | zips.str.startswith("1")
)

is_far = (
    ~is_close.fillna(False)
    & zips.notna()
)

print("\nPandas - ZIP codes outside the expected area:")
print(zips[is_far].head(20))


# Polars
pl_is_far = (
    ~(
        pl.col("Incident Zip").str.starts_with("0")
        | pl.col("Incident Zip").str.starts_with("1")
    ).fill_null(False)
    & pl.col("Incident Zip").is_not_null()
)

pl_far_requests = pl_requests.filter(pl_is_far)

print("\nPolars - ZIP codes outside the expected area:")
print(
    pl_far_requests.select(
        "Incident Zip"
    ).head(20)
)


# %%
# 10. INVESTIGATE UNUSUAL ZIP CODES

# Pandas
pandas_far_requests = (
    requests.loc[
        is_far,
        ["Incident Zip", "Descriptor", "City"]
    ]
    .sort_values("Incident Zip")
)

print("\nPandas - Unusual ZIP-code details:")
print(pandas_far_requests.head(20))


# Polars
pl_far_requests_details = (
    pl_requests
    .filter(pl_is_far)
    .select(
        ["Incident Zip", "Descriptor", "City"]
    )
    .sort("Incident Zip")
)

print("\nPolars - Unusual ZIP-code details:")
print(pl_far_requests_details.head(20))


# %%
# 11. COUNT REQUESTS BY CITY

# Pandas
pandas_city_counts = (
    requests["City"]
    .str.upper()
    .value_counts()
)

print("\nPandas - Most common cities:")
print(pandas_city_counts.head(20))


# Polars
pl_city_counts = (
    pl_requests
    .with_columns(
        pl.col("City")
        .str.to_uppercase()
        .alias("City")
    )
    .group_by("City")
    .agg(
        pl.len().alias("count")
    )
    .sort("count", descending=True)
)

print("\nPolars - Most common cities:")
print(pl_city_counts.head(20))


# %%
# 12. CREATE REUSABLE ZIP-CODE CLEANING FUNCTIONS

# Pandas
def fix_zip_codes(zips):
    """
    Clean ZIP codes in a Pandas Series.

    Truncate ZIP codes to five characters and
    replace 00000 with a missing value.
    """
    zips = zips.str.slice(0, 5).copy()
    zips = zips.mask(zips == "00000")

    return zips


# Polars
def pl_fix_zip_codes(data):
    """
    Clean ZIP codes in a Polars DataFrame.

    Truncate ZIP codes to five characters and
    replace 00000 with null.
    """
    return (
        data
        .with_columns(
            pl.col("Incident Zip")
            .str.slice(0, 5)
            .alias("Incident Zip")
        )
        .with_columns(
            pl.when(
                pl.col("Incident Zip") == "00000"
            )
            .then(pl.lit(None, dtype=pl.String))
            .otherwise(pl.col("Incident Zip"))
            .alias("Incident Zip")
        )
    )


# %%
# 13. APPLY THE REUSABLE CLEANING FUNCTIONS
#
# Reload the data so both libraries begin with
# their original uncleaned ZIP codes.

# Pandas
requests = pd.read_csv(
    DATA_PATH,
    na_values=na_values,
    keep_default_na=False,
    dtype={"Incident Zip": str}
)

requests["Incident Zip"] = fix_zip_codes(
    requests["Incident Zip"]
)

print("\nPandas - Final cleaned ZIP codes:")
print(
    sorted(
        requests["Incident Zip"]
        .dropna()
        .unique()
    )[:20]
)


# Polars
pl_requests = pl.read_csv(
    DATA_PATH,
    infer_schema_length=0,
    null_values=na_values
)

pl_requests = pl_fix_zip_codes(pl_requests)

print("\nPolars - Final cleaned ZIP codes:")
print(
    pl_requests
    .select(
        pl.col("Incident Zip")
        .drop_nulls()
        .unique()
        .sort()
    )
    .head(20)
)


# %%
# 14. VERIFY THE CLEANING RESULTS

# Confirm row counts match
assert len(requests) == pl_requests.height

# Confirm no ZIP codes exceed five characters
assert (
    pl_requests
    .filter(
        pl.col("Incident Zip").str.len_chars() > 5
    )
    .height
    == 0
)

# Confirm 00000 ZIP codes have been removed
assert (
    pl_requests
    .filter(
        pl.col("Incident Zip") == "00000"
    )
    .height
    == 0
)

print("\nBasic validation checks passed.")


# %%
# 15. COMPARE PANDAS AND POLARS ZIP CODES
#
# Use the same placeholder for missing values
# to make comparisons easier.

pandas_zips = (
    requests["Incident Zip"]
    .fillna("<MISSING>")
    .tolist()
)

polars_zips = (
    pl_requests["Incident Zip"]
    .fill_null("<MISSING>")
    .to_list()
)

# Identify differences between the two libraries
differences = [
    (i, pd_zip, pl_zip)
    for i, (pd_zip, pl_zip) in enumerate(
        zip(pandas_zips, polars_zips)
    )
    if pd_zip != pl_zip
]

print("\nZIP-CODE COMPARISON")
print("----------------------------")
print("Pandas rows:", len(pandas_zips))
print("Polars rows:", len(polars_zips))
print("Mismatched ZIP codes:", len(differences))

# Display up to 20 differences
if differences:
    print("\nFirst 20 differences:")

    for row_index, pd_zip, pl_zip in differences[:20]:
        print(
            f"Row {row_index}: "
            f"Pandas = {pd_zip!r}, "
            f"Polars = {pl_zip!r}"
        )

    print(
        "\nThe differences need investigation "
        "before we consider the results equivalent."
    )

else:
    print("\nVerification successful!")
    print("Pandas and Polars produce matching cleaned ZIP codes.")


# %%
# 16. FINAL SUMMARY

print("\nCHAPTER 7 SUMMARY")
print("----------------------------")
print("Total requests:", pl_requests.height)

print(
    "Missing ZIP codes:",
    pl_requests["Incident Zip"].null_count()
)

print(
    "Unique non-missing ZIP codes:",
    pl_requests["Incident Zip"].drop_nulls().n_unique()
)

print("\nChapter 7 analysis complete!")
