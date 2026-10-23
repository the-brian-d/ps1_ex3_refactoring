
# %%
# Chapter 5 - Combining DataFrames and Canadian Weather Data
# D100 Exercise 3: Refactoring Pandas to Polars

from pathlib import Path
from io import BytesIO

import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import requests


# %%
# Plotting settings

plt.style.use("ggplot")
plt.rcParams["figure.figsize"] = (15, 5)
plt.rcParams["font.family"] = "sans-serif"


# %%
# Project paths

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

WEATHER_PATH = DATA_DIR / "weather_2012.csv"

# Use a different filename to preserve the original dataset
OUTPUT_PATH = DATA_DIR / "weather_2012_polars.csv"

# Set True to attempt downloading all 12 months.
# Leave False for the initial offline test.
RUN_LIVE_DOWNLOAD = False


# %%
# 1. Load the existing weather data
#
# The repository includes hourly weather data
# for Montreal during 2012.


# Original Pandas implementation
weather_2012_pd = pd.read_csv(
    WEATHER_PATH,
    parse_dates=["date_time"]
)

print("\nPandas - First five rows:")
print(weather_2012_pd.head())


# Polars implementation
weather_2012_pl = pl.read_csv(
    WEATHER_PATH,
    try_parse_dates=True
)

# Explicitly ensure the timestamp is a Datetime
if weather_2012_pl.schema["date_time"] == pl.String:
    weather_2012_pl = weather_2012_pl.with_columns(
        pl.col("date_time").str.to_datetime()
    )

print("\nPolars - First five rows:")
print(weather_2012_pl.head())

print("\nPolars column types:")
print(weather_2012_pl.schema)


# %%
# 2. Plot hourly temperature during 2012


# Pandas
fig, ax = plt.subplots(figsize=(15, 6))

ax.plot(
    weather_2012_pd["date_time"],
    weather_2012_pd["temperature_c"],
    color="steelblue"
)

ax.set_title("Hourly Temperature in 2012 - Pandas")
ax.set_xlabel("Date")
ax.set_ylabel("Temperature (C)")
ax.grid(True, alpha=0.3)

fig.tight_layout()


# Polars
fig, ax = plt.subplots(figsize=(15, 6))

ax.plot(
    weather_2012_pl["date_time"].to_list(),
    weather_2012_pl["temperature_c"].to_list(),
    color="seagreen"
)

ax.set_title("Hourly Temperature in 2012 - Polars")
ax.set_xlabel("Date")
ax.set_ylabel("Temperature (C)")
ax.grid(True, alpha=0.3)

fig.tight_layout()


# %%
# 3. Demonstrate dropping columns with missing values
#
# Pandas:
# dropna(axis=1, how="any")
#
# Polars:
# Select only columns where null_count() == 0.
#
# This is a demonstration. We do not overwrite
# our original DataFrame because weather measurements
# can legitimately contain missing observations.


# Pandas
pandas_complete = weather_2012_pd.dropna(
    axis=1,
    how="any"
)

print("\nPandas - Columns without missing values:")
print(pandas_complete.columns.tolist())


# Polars
complete_columns = [
    column
    for column in weather_2012_pl.columns
    if weather_2012_pl[column].null_count() == 0
]

polars_complete = weather_2012_pl.select(
    complete_columns
)

print("\nPolars - Columns without missing values:")
print(polars_complete.columns)


# %%
# 4. Drop unnecessary columns
#
# The original monthly data includes Year, Month,
# Day and Time (LST).
#
# These are redundant when a date_time column exists.
#
# The already-cleaned repository CSV does not contain
# these fields, so we ignore missing column names.


unnecessary_columns = [
    "Year",
    "Month",
    "Day",
    "Time (LST)",
]

pl_weather_reduced = weather_2012_pl.drop(
    unnecessary_columns,
    strict=False
)

print("\nPolars - After removing unnecessary columns:")
print(pl_weather_reduced.head())


# %%
# 5. Column name cleaning
#
# The original weather files may contain unwanted
# characters such as BOM marks, quotes or Â.
#
# We define a reusable helper function.


def normalize_column_name(name: str) -> str:
    """Remove unwanted characters from a column name."""

    return (
        name.replace("\ufeff", "")
        .replace("ï»¿", "")
        .replace("Â", "")
        .strip()
        .strip('"')
        .strip()
    )


pl_weather_renamed = pl_weather_reduced.rename(
    {
        col: normalize_column_name(col)
        for col in pl_weather_reduced.columns
    }
)

print("\nPolars - Cleaned column names:")
print(pl_weather_renamed.columns)


# %%
# 6. Manually rename weather columns
#
# This mapping handles the columns found in
# the original Environment Canada files.


COLUMN_MAPPING = {
    "Date/Time (LST)": "date_time",
    "Longitude (x)": "longitude",
    "Latitude (y)": "latitude",
    "Station Name": "station_name",
    "Climate ID": "climate_id",
    "Temp (°C)": "temperature_c",
    "Dew Point Temp (°C)": "dew_point_temp_c",
    "Rel Hum (%)": "relative_humidity",
    "Wind Spd (km/h)": "wind_speed_kmh",
    "Visibility (km)": "visibility_km",
    "Stn Press (kPa)": "station_pressure_kpa",
    "Weather": "weather",
}


# Only rename columns that are present
available_mapping = {
    old: new
    for old, new in COLUMN_MAPPING.items()
    if old in pl_weather_renamed.columns
}

pl_weather_renamed = pl_weather_renamed.rename(
    available_mapping
)

# Convert all column names to lowercase
pl_weather_renamed = pl_weather_renamed.rename(
    {
        col: col.lower()
        for col in pl_weather_renamed.columns
    }
)

print("\nPolars - Final column names:")
print(pl_weather_renamed.columns)


# %%
# 7. Plot the March 2012 temperature
#
# Filter the existing dataset rather than downloading
# the data again.


# Pandas
march_pd = weather_2012_pd[
    weather_2012_pd["date_time"].dt.month == 3
]

fig, ax = plt.subplots(figsize=(15, 5))

ax.plot(
    march_pd["date_time"],
    march_pd["temperature_c"],
    color="steelblue"
)

ax.set_title("March 2012 Temperature - Pandas")
ax.set_xlabel("Date")
ax.set_ylabel("Temperature (C)")
ax.grid(True, alpha=0.3)

fig.tight_layout()


# Polars
march_pl = weather_2012_pl.filter(
    pl.col("date_time").dt.month() == 3
)

fig, ax = plt.subplots(figsize=(15, 5))

ax.plot(
    march_pl["date_time"].to_list(),
    march_pl["temperature_c"].to_list(),
    color="seagreen"
)

ax.set_title("March 2012 Temperature - Polars")
ax.set_xlabel("Date")
ax.set_ylabel("Temperature (C)")
ax.grid(True, alpha=0.3)

fig.tight_layout()


# %%
# 8. Calculate median temperature by hour
#
# We want to find out whether it gets colder
# at night and which hour is warmest.


# Pandas
march_pd = march_pd.copy()

march_pd["Hour"] = (
    march_pd["date_time"].dt.hour
)

pandas_hourly_median = (
    march_pd
    .groupby("Hour")["temperature_c"]
    .median()
    .sort_index()
)

print("\nPandas - Median temperature by hour:")
print(pandas_hourly_median)


# Polars
polars_hourly_median = (
    march_pl
    .with_columns(
        pl.col("date_time")
        .dt.hour()
        .alias("Hour")
    )
    .group_by("Hour")
    .agg(
        pl.col("temperature_c")
        .median()
        .alias("median_temperature")
    )
    .sort("Hour")
)

print("\nPolars - Median temperature by hour:")
print(polars_hourly_median)


# %%
# 9. Plot median temperature by hour


# Pandas
fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    pandas_hourly_median.index,
    pandas_hourly_median.values,
    marker="o",
    color="steelblue"
)

ax.set_title(
    "Median March Temperature by Hour - Pandas"
)

ax.set_xlabel("Hour of Day")
ax.set_ylabel("Median Temperature (C)")
ax.set_xticks(range(24))
ax.grid(True, alpha=0.3)

fig.tight_layout()


# Polars
fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    polars_hourly_median["Hour"].to_list(),
    polars_hourly_median["median_temperature"].to_list(),
    marker="o",
    color="seagreen"
)

ax.set_title(
    "Median March Temperature by Hour - Polars"
)

ax.set_xlabel("Hour of Day")
ax.set_ylabel("Median Temperature (C)")
ax.set_xticks(range(24))
ax.grid(True, alpha=0.3)

fig.tight_layout()


# %%
# 10. Reusable monthly weather cleaning function
#
# This function accepts raw Environment Canada data
# and returns a standardised Polars DataFrame.
#
# We drop completely empty columns rather than
# columns with any missing values, to preserve
# useful weather measurements.


def clean_data(data: pl.DataFrame) -> pl.DataFrame:
    """Clean raw monthly weather data using Polars."""

    # Work with a new DataFrame
    df = data.clone()

    # Normalise column names
    df = df.rename(
        {
            col: normalize_column_name(col)
            for col in df.columns
        }
    )

    # Remove completely empty columns
    keep_columns = [
        col
        for col in df.columns
        if df[col].null_count() < df.height
    ]

    df = df.select(keep_columns)

    # Remove redundant columns
    df = df.drop(
        [
            "Year",
            "Month",
            "Day",
            "Time (LST)",
        ],
        strict=False
    )

    # Rename columns
    rename_mapping = {
        old: new
        for old, new in COLUMN_MAPPING.items()
        if old in df.columns
    }

    df = df.rename(rename_mapping)

    # Convert names to lowercase
    df = df.rename(
        {
            col: col.lower()
            for col in df.columns
        }
    )

    # Ensure date_time exists
    if "date_time" not in df.columns:
        raise ValueError(
            "The weather data has no date_time column."
        )

    # Convert timestamps
    if df.schema["date_time"] == pl.String:
        df = df.with_columns(
            pl.col("date_time")
            .str.to_datetime(strict=False)
        )

    # Convert temperature to numeric
    if "temperature_c" in df.columns:
        df = df.with_columns(
            pl.col("temperature_c")
            .cast(pl.Float64, strict=False)
        )

    return df


# %%
# 11. Reusable function to download one month
#
# Note:
# Historical download endpoints may change.
# A failed request will raise an error instead
# of silently returning incorrect data.


def download_weather_month(
    year: int,
    month: int
) -> pl.DataFrame:
    """Download and clean one month of hourly weather."""

    url = (
        "https://climate.weather.gc.ca/"
        "climate_data/bulk_data_e.html"
    )

    params = {
        "format": "csv",
        "stationID": 5415,
        "Year": year,
        "Month": month,
        "timeframe": 1,
        "submit": "Download Data",
    }

    response = requests.get(
        url,
        params=params,
        timeout=60
    )

    response.raise_for_status()

    # Check that we received CSV content,
    # rather than an HTML error page.
    beginning = response.content[:200].lstrip().lower()

    if (
        beginning.startswith(b"<!doctype html")
        or beginning.startswith(b"<html")
    ):
        raise ValueError(
            f"The server returned HTML for {year}-{month:02d}, "
            "not a weather CSV."
        )

    raw_data = pl.read_csv(
        BytesIO(response.content),
        encoding="latin1",
        infer_schema=False,
        truncate_ragged_lines=True
    )

    return clean_data(raw_data)


# %%
# 12. Download and combine all 12 months
#
# By default, this section is disabled.
#
# Set RUN_LIVE_DOWNLOAD = True at the top of
# the script to attempt downloading the data.


if RUN_LIVE_DOWNLOAD:

    print("\nDownloading monthly weather data...")

    data_by_month = []

    for month in range(1, 13):

        print(f"Downloading 2012-{month:02d}...")

        monthly_data = download_weather_month(
            2012,
            month
        )

        data_by_month.append(monthly_data)

    # Combine the monthly DataFrames.
    # diagonal_relaxed handles columns that differ
    # between months and compatible type differences.

    weather_2012_combined = pl.concat(
        data_by_month,
        how="diagonal_relaxed"
    )

    weather_2012_combined = (
        weather_2012_combined
        .sort("date_time")
    )

    print("\nCombined weather data:")
    print(weather_2012_combined.head())

    print("\nCombined shape:")
    print(weather_2012_combined.shape)

    # Save using a new filename.
    # Do not overwrite the original repository data.

    weather_2012_combined.write_csv(
        OUTPUT_PATH
    )

    print(f"\nSaved to: {OUTPUT_PATH}")


# %%
# 13. Demonstrate concatenation using local data
#
# This allows us to test pl.concat() without
# requiring an internet connection.


local_months = [
    weather_2012_pl.filter(
        pl.col("date_time").dt.month() == month
    )
    for month in range(1, 13)
]

weather_2012_recombined = pl.concat(
    local_months,
    how="vertical"
).sort("date_time")

print("\nLocally recombined weather data:")
print(weather_2012_recombined.head())


# %%
# 14. Final verification


print("\n--- Chapter 5 Verification ---")

print("\nPandas DataFrame shape:")
print(weather_2012_pd.shape)

print("\nPolars DataFrame shape:")
print(weather_2012_pl.shape)

assert len(weather_2012_pd) == weather_2012_pl.height, (
    "Pandas and Polars row counts differ!"
)

assert (
    weather_2012_recombined.height
    == weather_2012_pl.height
), "Concatenation changed the row count!"

assert (
    weather_2012_recombined["date_time"].to_list()
    == weather_2012_pl.sort("date_time")["date_time"].to_list()
), "Concatenation changed the timestamps!"

print("\nAll verification checks passed!")


# %%
# 15. Identify the warmest hour in March


warmest_hour = (
    polars_hourly_median
    .sort(
        "median_temperature",
        descending=True
    )
    .head(1)
)

print("\nHour with the highest median March temperature:")
print(warmest_hour)


# %%
# 16. Display all charts

plt.show()

print(
    "\nChapter 5 refactoring completed successfully!"
)
