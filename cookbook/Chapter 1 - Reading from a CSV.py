
# %%
# Chapter 1 - Reading from a CSV
# D100 Exercise 3: Refactoring Pandas to Polars

from pathlib import Path

import pandas as pd
import polars as pl
import matplotlib.pyplot as plt


# %%
# Set the path relative to this script
# This works regardless of the terminal's current directory.

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "bikes.csv"


# %%
# 1. Reading data from a CSV file
#
# The dataset contains daily cyclist counts on bicycle
# paths in Montreal during 2012.
#
# Initially, we load the data without specifying the
# correct separator to demonstrate the problem.


# Original Pandas implementation
broken_df = pd.read_csv(
    DATA_PATH,
    encoding="ISO-8859-1"
)

# Polars implementation
pl_broken_df = pl.read_csv(
    DATA_PATH,
    encoding="iso-8859-1"
)


# %%
# 2. Display the first three rows

# Pandas
print("\nPandas - First three rows (incorrect loading):")
print(broken_df.head(3))

# Polars
print("\nPolars - First three rows (incorrect loading):")
print(pl_broken_df.head(3))


# %%
# 3. Load the CSV correctly
#
# The dataset uses:
# - Semicolons as separators
# - Latin-1 encoding
# - Dates in DD/MM/YYYY format
# - Empty fields representing missing values


# Original Pandas implementation
fixed_df = pd.read_csv(
    DATA_PATH,
    sep=";",
    encoding="latin1",
    parse_dates=["Date"],
    dayfirst=True,
    index_col="Date",
)

print("\nPandas - Correctly loaded data:")
print(fixed_df.head(3))


# Polars implementation
pl_fixed_df = pl.read_csv(
    DATA_PATH,
    separator=";",
    encoding="latin1"
)

# Convert date strings into proper Date values
pl_fixed_df = pl_fixed_df.with_columns(
    pl.col("Date").str.strptime(
        pl.Date,
        format="%d/%m/%Y"
    )
)

print("\nPolars - Correctly loaded data:")
print(pl_fixed_df.head(3))

# Verify the column types
print("\nPolars column types:")
print(pl_fixed_df.schema)


# %%
# 4. Selecting a column
#
# Select the number of cyclists counted on Berri 1.


# Pandas
print("\nPandas - Berri 1:")
print(fixed_df["Berri 1"].head())


# Polars
print("\nPolars - Berri 1:")
print(
    pl_fixed_df.select("Berri 1").head()
)


# %%
# 5. Plotting one column
#
# Compare daily cyclist counts on Berri 1.


# Original Pandas implementation
fixed_df["Berri 1"].plot(
    title="Berri 1 - Pandas",
    figsize=(12, 5)
)

plt.xlabel("Date")
plt.ylabel("Number of cyclists")
plt.grid(True, alpha=0.3)
plt.tight_layout()


# Polars implementation using Matplotlib
fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    pl_fixed_df["Date"].to_list(),
    pl_fixed_df["Berri 1"].to_list(),
    color="steelblue",
    linewidth=1.5
)

ax.set_title("Berri 1 - Polars")
ax.set_xlabel("Date")
ax.set_ylabel("Number of cyclists")
ax.grid(True, alpha=0.3)

fig.tight_layout()


# %%
# 6. Plotting all bicycle paths
#
# Compare daily cyclist counts across different
# bicycle paths in Montreal.


# Original Pandas implementation
fixed_df.plot(
    figsize=(15, 10),
    title="Montreal Bicycle Traffic - Pandas"
)

plt.xlabel("Date")
plt.ylabel("Number of cyclists")
plt.grid(True, alpha=0.3)
plt.tight_layout()


# Polars implementation using Matplotlib
fig, ax = plt.subplots(figsize=(15, 10))

# Get the dates from the Polars DataFrame
dates = pl_fixed_df["Date"].to_list()

# Plot every bicycle path
for column in pl_fixed_df.columns:

    # Date is our horizontal axis
    if column == "Date":
        continue

    # Skip columns containing only missing values
    if (
        pl_fixed_df[column].null_count()
        == pl_fixed_df.height
    ):
        continue

    ax.plot(
        dates,
        pl_fixed_df[column].to_list(),
        label=column,
        linewidth=1.5
    )

ax.set_title("Montreal Bicycle Traffic - Polars")
ax.set_xlabel("Date")
ax.set_ylabel("Number of cyclists")

ax.legend(
    title="Bicycle Path",
    loc="upper left"
)

ax.grid(True, alpha=0.3)

fig.tight_layout()


# %%
# Display all generated charts

plt.show()


# %%
# 7. Final verification

print("\n--- Chapter 1 Verification ---")

print("\nPandas DataFrame shape:")
print(fixed_df.shape)

print("\nPolars DataFrame shape:")
print(pl_fixed_df.shape)

print("\nPolars Date column type:")
print(pl_fixed_df.schema["Date"])

print("\nChapter 1 refactoring completed successfully!")
