
# %%
# Chapter 6 - String Operations: Which Month Was the Snowiest?
# Refactoring Pandas code to Polars

from pathlib import Path

import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import numpy as np

# Plot settings
plt.style.use("ggplot")
plt.rcParams["figure.figsize"] = (15, 4)
plt.rcParams["font.family"] = "sans-serif"

# Locate the data folder independently of the terminal's location
DATA_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "weather_2012.csv"
)


# %%
# 1. LOAD THE WEATHER DATA
# We use the weather data from Chapter 5.

# Pandas
weather_2012 = pd.read_csv(
    DATA_PATH,
    parse_dates=["date_time"],
    index_col="date_time",
).sort_index()

print("\nPandas - First five rows:")
print(weather_2012.head(5))


# Polars
pl_wather_2012 = pl.read_csv(
    DATA_PATH,
    try_parse_dates=True,
)

# Make sure date_time is a datetime column
if pl_wather_2012.schema["date_time"] == pl.String:
    pl_wather_2012 = pl_wather_2012.with_columns(
        pl.col("date_time").str.to_datetime()
    )

pl_wather_2012 = pl_wather_2012.sort("date_time")

print("\nPolars - First five rows:")
print(pl_wather_2012.head(5))


# %%
# 2. IDENTIFY HOURS WHEN IT WAS SNOWING
#
# We assume it was snowing whenever the weather description
# contains the word "Snow".
#
# Convert True to 1.0 and False to 0.0.

# Pandas
weather_description = weather_2012["weather"]

is_snowing = (
    weather_description
    .str.contains("Snow", regex=False, na=False)
    .astype(float)
)

print("\nPandas - Snow indicators:")
print(is_snowing.head(10))

# Plot Pandas results
plt.figure(figsize=(15, 3))
is_snowing.plot()
plt.title("Pandas - Snowfall During 2012")
plt.xlabel("Date")
plt.ylabel("Snowing (1 = Yes, 0 = No)")
plt.tight_layout()
plt.show()


# Polars
pl_wather_2012 = pl_wather_2012.with_columns(
    pl.col("weather")
    .str.contains("Snow", literal=True)
    .fill_null(False)
    .cast(pl.Float64)
    .alias("is_snowing")
)

print("\nPolars - Snow indicators:")
print(
    pl_wather_2012.select(
        ["date_time", "weather", "is_snowing"]
    ).head(10)
)

# Plot Polars results using Matplotlib
plt.figure(figsize=(15, 3))
plt.plot(
    pl_wather_2012["date_time"].to_list(),
    pl_wather_2012["is_snowing"].to_list(),
)
plt.title("Polars - Snowfall During 2012")
plt.xlabel("Date")
plt.ylabel("Snowing (1 = Yes, 0 = No)")
plt.tight_layout()
plt.show()


# %%
# 3. MEDIAN TEMPERATURE FOR EACH MONTH
#
# Pandas uses resample().
# Polars uses group_by_dynamic().
#
# "MS" in Pandas means month start.
# "1mo" in Polars means one calendar month.

# Pandas
pandas_monthly_temp = (
    weather_2012["temperature_c"]
    .resample("MS")
    .median()
)

print("\nPandas - Monthly median temperatures:")
print(pandas_monthly_temp)

plt.figure(figsize=(12, 5))
plt.bar(
    pandas_monthly_temp.index.strftime("%b"),
    pandas_monthly_temp.values,
)
plt.title("Pandas - Median Temperature by Month (2012)")
plt.xlabel("Month")
plt.ylabel("Median Temperature (°C)")
plt.tight_layout()
plt.show()


# Polars
polars_monthly_temp = (
    pl_wather_2012
    .group_by_dynamic("date_time", every="1mo")
    .agg(
        pl.col("temperature_c")
        .median()
        .alias("median_temperature_c")
    )
    .sort("date_time")
)

print("\nPolars - Monthly median temperatures:")
print(polars_monthly_temp)

# Extract month names and temperature values
temp_months = (
    polars_monthly_temp["date_time"]
    .dt.strftime("%b")
    .to_list()
)

temp_values = (
    polars_monthly_temp["median_temperature_c"]
    .to_list()
)

plt.figure(figsize=(12, 5))
plt.bar(temp_months, temp_values)
plt.title("Polars - Median Temperature by Month (2012)")
plt.xlabel("Month")
plt.ylabel("Median Temperature (°C)")
plt.tight_layout()
plt.show()


# %%
# 4. PERCENTAGE OF TIME IT SNOWED EACH MONTH
#
# True = 1.0
# False = 0.0
#
# The mean of these numbers gives the fraction of
# observed hours when it was snowing.
#
# Multiply by 100 to obtain a percentage.

# Pandas
pandas_monthly_snow = (
    is_snowing
    .resample("MS")
    .mean()
)

print("\nPandas - Monthly snow percentages:")
print(pandas_monthly_snow * 100)

plt.figure(figsize=(12, 5))
plt.bar(
    pandas_monthly_snow.index.strftime("%b"),
    pandas_monthly_snow.values * 100,
)
plt.title("Pandas - Percentage of Hours with Snow (2012)")
plt.xlabel("Month")
plt.ylabel("Hours with Snow (%)")
plt.tight_layout()
plt.show()


# Polars
polars_monthly_snow = (
    pl_wather_2012
    .group_by_dynamic("date_time", every="1mo")
    .agg(
        pl.col("is_snowing")
        .mean()
        .alias("snow_fraction")
    )
    .with_columns(
        (pl.col("snow_fraction") * 100)
        .alias("snow_percentage")
    )
    .sort("date_time")
)

print("\nPolars - Monthly snow percentages:")
print(polars_monthly_snow)

snow_months = (
    polars_monthly_snow["date_time"]
    .dt.strftime("%b")
    .to_list()
)

snow_percentages = (
    polars_monthly_snow["snow_percentage"]
    .to_list()
)

plt.figure(figsize=(12, 5))
plt.bar(snow_months, snow_percentages)
plt.title("Polars - Percentage of Hours with Snow (2012)")
plt.xlabel("Month")
plt.ylabel("Hours with Snow (%)")
plt.tight_layout()
plt.show()


# %%
# 5. FIND THE SNOWIEST MONTH USING POLARS

snowiest_month = (
    polars_monthly_snow
    .sort("snow_percentage", descending=True)
    .head(1)
)

month_name = (
    snowiest_month["date_time"]
    .dt.strftime("%B")
    .item()
)

snow_percentage = snowiest_month[
    "snow_percentage"
].item()

print("\nSNOWIEST MONTH")
print("----------------------------")
print(f"Month: {month_name}")
print(f"Hours with snow: {snow_percentage:.2f}%")


# %%
# 6. VERIFY THAT PANDAS AND POLARS AGREE

# Check the number of observations
assert len(weather_2012) == pl_wather_2012.height

# Check the number of hours with snow
np.testing.assert_allclose(
    is_snowing.to_numpy(),
    pl_wather_2012["is_snowing"].to_numpy(),
)

# Compare monthly median temperatures
np.testing.assert_allclose(
    pandas_monthly_temp.to_numpy(),
    polars_monthly_temp["median_temperature_c"].to_numpy(),
    equal_nan=True,
)

# Compare monthly snow fractions
np.testing.assert_allclose(
    pandas_monthly_snow.to_numpy(),
    polars_monthly_snow["snow_fraction"].to_numpy(),
    equal_nan=True,
)

print("\nVerification successful!")
print("Pandas and Polars produce matching results.")

print("\nChapter 6 analysis complete!")
