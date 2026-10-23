
# %%
# Chapter 8 - How to Deal with Timestamps
# Refactoring Pandas code to Polars

from pathlib import Path
from datetime import datetime

import pandas as pd
import polars as pl


# %%
# FILE PATH AND COLUMN NAMES

DATA_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "popularity-contest"
)

COLUMN_NAMES = [
    "atime",
    "ctime",
    "package-name",
    "mru-program",
    "tag",
]


# %%
# 1. LOAD THE POPULARITY-CONTEST DATA
#
# The file contains information about installed packages.
#
# atime = last access time
# ctime = last status-change time
#
# Both timestamps represent seconds since 1970-01-01.


# Pandas
popcon = pd.read_csv(
    DATA_PATH,
    sep=" ",
)[:-1]

popcon.columns = COLUMN_NAMES

print("\nPandas - First five rows:")
print(popcon.head(5))


# Polars
#
# The popularity-contest file is space-separated.
# Some rows do not contain a value for "tag".
#
# Read the file line by line and handle missing tags.

with DATA_PATH.open(
    "r",
    encoding="utf-8",
) as file:
    lines = file.read().splitlines()

# Remove empty lines
lines = [
    line for line in lines
    if line.strip()
]

# Remove the header and footer
data_lines = lines[1:-1]

# Parse the records
rows = []

for line in data_lines:

    # Split into at most five fields
    fields = line.split(maxsplit=4)

    # Some records have only four fields
    # because the tag is missing.
    if len(fields) == 4:
        fields.append(None)

    # Check for unexpected formats
    if len(fields) != 5:
        raise ValueError(
            f"Unexpected number of fields: {line!r}"
        )

    rows.append(fields)

print(
    f"\nSuccessfully parsed {len(rows)} package records."
)

# Create the Polars DataFrame
pl_popcon = pl.DataFrame(
    rows,
    schema=COLUMN_NAMES,
    orient="row",
)

print("\nPolars - First five rows:")
print(pl_popcon.head(5))


# %%
# 2. CONVERT UNIX TIMESTAMPS TO INTEGERS
#
# Unix timestamps represent the number of seconds
# elapsed since 1970-01-01 00:00:00 UTC.


# Pandas
popcon["atime"] = popcon["atime"].astype(int)
popcon["ctime"] = popcon["ctime"].astype(int)

print("\nPandas - Integer timestamps:")
print(
    popcon[
        ["atime", "ctime", "package-name"]
    ].head(5)
)

print("\nPandas - Data types:")
print(
    popcon[
        ["atime", "ctime"]
    ].dtypes
)


# Polars
pl_popcon = pl_popcon.with_columns(
    pl.col("atime").cast(pl.Int64),
    pl.col("ctime").cast(pl.Int64),
)

print("\nPolars - Integer timestamps:")
print(
    pl_popcon.select(
        ["atime", "ctime", "package-name"]
    ).head(5)
)

print("\nPolars - Data types:")
print(pl_popcon.schema)


# %%
# 3. CONVERT UNIX TIMESTAMPS INTO DATETIMES
#
# Pandas uses pd.to_datetime().
# Polars uses pl.from_epoch().


# Pandas
popcon["atime"] = pd.to_datetime(
    popcon["atime"],
    unit="s",
)

popcon["ctime"] = pd.to_datetime(
    popcon["ctime"],
    unit="s",
)

print("\nPandas - Converted datetimes:")
print(
    popcon[
        ["atime", "ctime", "package-name"]
    ].head(5)
)


# Polars
pl_popcon = pl_popcon.with_columns(
    pl.from_epoch(
        pl.col("atime"),
        time_unit="s",
    ).alias("atime"),

    pl.from_epoch(
        pl.col("ctime"),
        time_unit="s",
    ).alias("ctime"),
)

print("\nPolars - Converted datetimes:")
print(
    pl_popcon.select(
        ["atime", "ctime", "package-name"]
    ).head(5)
)

print("\nPolars - Updated data types:")
print(pl_popcon.schema)


# %%
# 4. REMOVE RECORDS WITH TIMESTAMP ZERO
#
# Timestamp zero represents 1970-01-01.
#
# Keep packages whose access times are later
# than this date.


# Pandas
popcon = popcon[
    popcon["atime"] > "1970-01-01"
]

print("\nPandas - Packages with valid access times:")
print(popcon.head(5))

print(
    "\nRemaining Pandas rows:",
    len(popcon)
)


# Polars
pl_popcon = pl_popcon.filter(
    pl.col("atime") > datetime(1970, 1, 1)
)

print("\nPolars - Packages with valid access times:")
print(pl_popcon.head(5))

print(
    "\nRemaining Polars rows:",
    pl_popcon.height
)


# %%
# 5. EXCLUDE LIBRARY PACKAGES
#
# We want to find packages whose names
# do not contain the substring "lib".
#
# The ~ operator means NOT.


# Pandas
nonlibraries = popcon[
    ~popcon["package-name"]
    .str.contains(
        "lib",
        regex=False,
        na=False,
    )
]

print("\nPandas - Non-library packages:")
print(nonlibraries.head(10))


# Polars
pl_nonlibraries = pl_popcon.filter(
    ~pl.col("package-name")
    .str.contains(
        "lib",
        literal=True,
    )
    .fill_null(False)
)

print("\nPolars - Non-library packages:")
print(pl_nonlibraries.head(10))


# %%
# 6. SORT PACKAGES BY TIMESTAMP
#
# Sort ctime in descending order to put
# the most recent timestamps first.


# Pandas
pandas_sorted = (
    nonlibraries
    .sort_values(
        "ctime",
        ascending=False,
    )
    .head(10)
)

print("\nPandas - Top 10 non-library packages:")
print(pandas_sorted)


# Polars
polars_sorted = (
    pl_nonlibraries
    .sort(
        "ctime",
        descending=True,
    )
    .head(10)
)

print("\nPolars - Top 10 non-library packages:")
print(polars_sorted)


# %%
# 7. VERIFY PANDAS AND POLARS RESULTS
#
# Check that both libraries produce equivalent results.


print("\nVERIFICATION")
print("----------------------------")


# Check the number of rows after date filtering
assert len(popcon) == pl_popcon.height

print("Date filtering: PASSED")


# Check the number of non-library packages
assert len(nonlibraries) == pl_nonlibraries.height

print("Library filtering: PASSED")


# Compare the selected package names
pandas_packages = sorted(
    nonlibraries["package-name"].tolist()
)

polars_packages = sorted(
    pl_nonlibraries["package-name"].to_list()
)

assert pandas_packages == polars_packages

print("Package selection: PASSED")


# Confirm no library names remain
assert not pl_nonlibraries[
    "package-name"
].str.contains(
    "lib",
    literal=True,
).any()

print("Library exclusion: PASSED")


# Confirm the timestamp columns are datetime values
assert pl_popcon.schema["atime"] == pl.Datetime("us")
assert pl_popcon.schema["ctime"] == pl.Datetime("us")

print("Datetime conversion: PASSED")


# Check that the results are sorted correctly
ctime_values = polars_sorted["ctime"].to_list()

assert ctime_values == sorted(
    ctime_values,
    reverse=True,
)

print("Timestamp sorting: PASSED")


# %%
# 8. FINAL SUMMARY

print("\nCHAPTER 8 SUMMARY")
print("----------------------------")

print(
    "Total packages after date filtering:",
    pl_popcon.height,
)

print(
    "Non-library packages:",
    pl_nonlibraries.height,
)

print(
    "Library packages excluded:",
    pl_popcon.height - pl_nonlibraries.height,
)

print("\nVerification successful!")
print("Pandas and Polars produce matching results.")

print("\nChapter 8 analysis complete!")
