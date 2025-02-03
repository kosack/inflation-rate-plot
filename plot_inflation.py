import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import pandas as pd
from pandas._libs.tslibs.offsets import YearEnd
import requests
from scipy.interpolate import interp1d
import numpy as np
from datetime import datetime

# World Bank API base URL
BASE_URL = "https://api.worldbank.org/v2/country/{}/indicator/FP.CPI.TOTL.ZG?date=1970:2026&format=json"

# Countries and labels
COUNTRIES = {"USA": "United States", "DEU": "Germany", "FRA": "France"}

# U.S. Presidential Terms
PRESIDENTS: list[tuple[float, float, str, str]] = [
    (1970, 1974, "Nixon", "red"),
    (1974, 1977, "Ford", "darkred"),
    (1977, 1981, "Carter", "blue"),
    (1981, 1989, "Reagan", "red"),
    (1989, 1993, "Bush Sr.", "darkred"),
    (1993, 2001, "Clinton", "blue"),
    (2001, 2009, "Bush Jr.", "red"),
    (2009, 2017, "Obama", "blue"),
    (2017, 2021, "Trump", "red"),
    (2021, 2025, "Biden", "blue"),
    (2025, 2029, "Trump", "red"),
]


EVENTS: list[tuple[float, float, str]] = [
    (2019, 2022, "COVID Pandemic"),
    (2022, 2026, "Ukraine War"),
    (1965, 1973, "Vietnam War"),
    (2007, 2009, "Subprime Mortgage Crisis"),
    (1990, 1991, "Gulf War"),
    (2001, 2021, "War in Afghanistan"),
    (2003, 2011, "Iraq War"),
    (1973, 1977, "Oil Crisis (OPEC)"),
    (2008, 2009, "Bird Flu Outbreak"),
    (1981, 1982, "Oil Crisis (Iran)"),
    (2020, 2023, "Bird Flu Outbreak"),
    (2015, 2016, "Bird Flu Outbreak"),
]


def fetch_inflation_data(country_code) -> dict:
    """Fetch inflation data from the World Bank API."""
    response = requests.get(BASE_URL.format(country_code))
    data = response.json()
    if len(data) < 2:
        return None  # Handle API error

    records = data[1]
    inflation_data = {
        int(entry["date"]): entry["value"] if entry["value"] is not None else None
        for entry in records
    }
    return inflation_data


# Fetch inflation data for all three countries
data = {country: fetch_inflation_data(code) for code, country in COUNTRIES.items()}

# Convert to DataFrame
df = (
    pd.DataFrame(data)
    .assign(date=lambda d: pd.to_datetime(d.index, format="%Y"))
    .set_index("date", drop=True)
    .sort_index()
)


def overlay_presidents(ax=None):
    ax = plt.gca() if ax is None else ax
    transform_xdata_yaxis = transforms.blended_transform_factory(
        ax.transData, ax.transAxes
    )

    # Overlay presidential terms
    for start, end, name, color in PRESIDENTS:
        start = pd.to_datetime(start, format="%Y")
        end = pd.to_datetime(end, format="%Y")

        ax.axvspan(start, end, color=color, alpha=0.1)
        ax.text(
            start + (end - start) / 2,
            0.9,
            name,
            ha="center",
            fontsize=10,
            fontweight="bold",
            color="black",
            transform=transform_xdata_yaxis,
            clip_on=True,
        )


def overlay_events(ax=None, width=0.02):
    ax = plt.gca() if ax is None else ax
    transform_xdata_yaxis = transforms.blended_transform_factory(
        ax.transData, ax.transAxes
    )

    # Overlay events
    for i, (start, end, name) in enumerate(EVENTS):
        start = pd.to_datetime(start, format="%Y")
        end = pd.to_datetime(end, format="%Y")
        vloc = 0.2 + 0.05 * i

        ax.axvspan(
            xmin=start,
            xmax=end,
            ymin=vloc - width / 2,
            ymax=vloc + width / 2,
            color="black",
        )
        ax.text(
            start + (end - start) / 2,
            vloc + 0.01,
            name,
            transform=transform_xdata_yaxis,
            ha="center",
            va="bottom",
            fontsize=8,
            color="black",
            clip_on=True,
        )


fig, ax = plt.subplots(2, 1, figsize=(14, 12), layout="constrained")

for country in COUNTRIES.values():
    ax[0].plot(df.index, df[country], marker="o", label=country)

overlay_presidents(ax[0])
overlay_events(ax[0])

ax[0].axvline(datetime.today(), ls="dotted", lw=3, color="black")

ax[0].set_xlabel("Year")
ax[0].set_ylabel("Inflation Rate (%)")
ax[0].set_title("Inflation Rate Over Time (1970-2025)")
ax[0].legend(loc="lower left")


# Overlay the value of 20 dollars today, inflation adjusted for the past
presdf = (
    pd.DataFrame(PRESIDENTS, columns=["start", "end", "president", "color"])
    .assign(
        start=lambda d: pd.to_datetime(d.start, format="%Y"),
        end=lambda d: pd.to_datetime(d.end, format="%Y"),
        midyear=lambda d: d.start + (d.end - d.start) / 2,
    )
    .set_index("midyear")
)

base_year = 2023
base_price = 10
selected_dates = set(presdf[presdf.index.year < 2025].index)

country = COUNTRIES["USA"]


inflation_ratio = df[country] / 100 + 1

# integrate the inflation rate over all years (delta_t=1 year)
inflation_factor = inflation_ratio.cumprod().iloc[1:]  # dx = 1 year
cost_factor = inflation_factor / inflation_factor.iloc[-1]
deinflation_factor = interp1d(
    cost_factor.index.year, cost_factor, bounds_error="extrapolate"
)
deinflate = lambda year, value: deinflation_factor(year) * value

transform_xdata_yaxis = transforms.blended_transform_factory(
    ax[1].transData, ax[1].transAxes
)

for date in selected_dates:
    try:
        ax[0].text(
            x=date,
            y=0.01,
            s=f"\\${deinflate(date.year, base_price):.2f}",
            transform=transform_xdata_yaxis,
            ha="center",
            va="bottom",
            fontsize=8,
            color="green",
        )
    except ValueError as err:
        pass


# cmmpare price of eggs fetched from here: https://fred.stlouisfed.org/series/APU0000708111
# URL = https://fred.stlouisfed.org/release/tables?rid=454&eid=816251#snid=816885
eggdf: pd.DataFrame = (
    pd.read_csv("eggs_prices_st_louis.csv")
    .assign(date=lambda d: pd.to_datetime(d["observation_date"]))
    .set_index("date", drop=True)
    .drop(columns="observation_date")
    .rename(columns={"APU0000708111": "absolute"})
    .query("date < 2023")
    .assign(
        deinflated=lambda d: d["absolute"] / deinflation_factor(d.index.year),
        deinflated_2_dollars=lambda d: 2.0 * deinflation_factor(d.index.year),
    )
)


eggdf.plot(
    ax=ax[1],
    ylabel="Price [USD]",
    drawstyle="steps",
    title="Egg Prices, St. Louis",
)
overlay_presidents(ax[1])
overlay_events(ax[1])

# better limits than the auto ones
for axis in ax:
   axis.set_xlim(datetime(1975,1,1,),datetime.today())
ax[1].set_ylim(0,6)

plt.show()
