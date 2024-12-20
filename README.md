# 🍻 LeonardoDataVinci 🍻

## 🔴 Abstract

This project examines seasonal fluctuations in beer ratings using reviews from the “BeerAdvocate” forum, spanning 2002 to 2017. The analysis reveals that average ratings vary by season, suggesting influences beyond the intrinsic qualities of the beer itself.

This observation is significant because, ideally, beer ratings should reflect only the beer's inherent characteristics, unaffected by external factors. Our objective is to identify which subsets of beers contribute to the observed seasonality in average ratings and to explore potential external influences, such as meteorological factors.

Our analysis is divided into three steps. First, we explore the observed seasonal patterns, developing metrics to quantify seasonality and determine whether these patterns represent true, recurring annual phenomena. In the second part, we investigate the reasons behind these seasonal changes in ratings, by finding beer types that are seasonal in average ratings & seasonal in number of ratings. Additionally, we investigate if daily weather on the state level can be correlated to a change in ratings.

The results can be found on our [website](https://epfl-ada.github.io/ada-2024-project-leonardodatavinci/).

## ❓ Research Questions

- **Seasonality**: Is this pattern driven by consistent year-to-year effects, or are the fluctuations merely the result of outliers or a few exceptionally strong years distorting the average?
- **Beer Types**: Are specific types of beers driving these seasonal variations? Perhaps through seasonal spikes in average ratings or number of ratings?
- **Meteo:** Could factors like climate or weather influence these rating trends?

## 📂 Additional Datasets

To enrich our analysis and provide deeper insights, we have incorporated supplementary datasets beyond the primary beer ratings data:

- **World Bank Climate Data**: Monthly average temperature and precipitation data by state, sourced from the [Climate Change Knowledge Portal](https://climateknowledgeportal.worldbank.org/download-data). 
- **Global Historical Climatology Network daily Data (GHCNd)**: Daily average temperature data for all weather stations in each state, sourced from the [Global Historical Climatology Network](https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily)).
- **County Climate Zones**: Classification of U.S. counties into climate zones to analyze regional weather patterns and their influence on beer ratings. Data sourced from the [US Department of Energy](https://www.energy.gov/sites/prod/files/2015/10/f27/ba_climate_region_guide_7.3.pdf).
- **US Census Population Data**: County-level population data used to weight climate zone analysis for accurate state-level metrics. Data obtained from the [U.S. Census Bureau](https://www.census.gov/data/tables/time-series/demo/popest/2020s-counties-total.html).
- **Formatted Climate Zones Table**: A processed version of county climate zone data, sourced from [this GitHub Gist](https://gist.github.com/philngo/d3e251040569dba67942#file-climate_zones-csv).

## ⚙️ Methods
We focus our work on the reviews on the forum “BeerAdvocate” from 2002 until 2017, in the US.

- We specify only the US because there is enough data and only the northern hemisphere! And state-specific, so nice granularity!
- BeerAdvocate, because more data and focus on US

### 1. Year-to-Year Seasonal Patterns

**Question:**  
Does the observed seasonal pattern in beer ratings reflect a true year-to-year recurrence, or is it an artifact of averaging ratings across many years?

**Method:**  
To identify seasonality, we apply Fourier Analysis and examine the frequency spectrum for significant patterns. 
First, we preprocess the data using STL decomposition. STL (Seasonal and Trend decomposition using Loess) separates the seasonal component from the underlying trend and residual (noise) in the monthly rating averages.
After extracting the seasonal component with STL, we apply the Fourier transform to identify dominant frequencies. If the dominant period is 12 months (frequency of 0.083 cycles per month), this confirms that the average rating pattern is yearly seasonal.

### 2. Identify Seasonal and Unseasonal Beers - Number of Ratings vs. Average Rating*

**Question:**  Are specific types of beers driving these seasonal variations? Perhaps through seasonal spikes in average ratings or number of ratings?

**Method:**

We aim to identify beers with significant seasonality by examining two key scenarios: the number of ratings received each month and the average rating across seasons.
Metrics for identification include Fourier transform peaks to capture periodic patterns and Seasonal-Trend decomposition using Loess (STL) to separate seasonal components from trends.
To ensure data reliability, only beers with at least 500 reviews between 2002 and 2017 are considered. This threshold minimizes noise from beers with insufficient data to reflect seasonal trends.

First, we determine the beers with the highest seasonal impact based on the number of monthly ratings. Seasonal beers are identified and removed to observe whether overall seasonality decreases.
Next, we apply similar logic to beers with seasonal fluctuations in their average ratings.

The final analysis combines the seasonality in number of ratings and average rating to pinpoint the beers with the highest overall seasonality.

### 3.  Weather Influence By State

**Question:**  
Does state weather impact beer ratings and its seasonality?

**Method:**  
Correlate ratings with average state temperatures & precipitation. Use weighted mean of weather station data based on US county populations.

**Method:**
1. Data Acquisition
    - Monthly Weather Data: Download monthly average temperature and precipitation data for each U.S. state from the Climate Change Knowledge Portal.
    - Beer Ratings: Compile beer ratings with corresponding timestamps and state information.
    - County Population: Download from US Census
    - Climate Zones: Extract from US Department of Energy
2. Data Processing:
    - Temporal Alignment: Align beer ratings with corresponding monthly weather data based on the review dates.
    - State Aggregation: Aggregate beer ratings and weather data at the state level to facilitate comparative analysis.
3. Correlation Analysis:
    - Temperature and Ratings: Calculate the correlation between average monthly temperatures and beer ratings for each state.
    - Precipitation and Ratings: Assess the relationship between monthly precipitation levels and beer ratings.
    - Regional Comparisons: Compare states to identify regional patterns in weather-related beer rating fluctuations.


## 🎯 Timeline

| Date       | Task                                              |
|------------|---------------------------------------------------|
| 15.11.2023 | Data Handling and Preprocessing & Initial Exploratory Data Analysis |
| 29.11.2024 | Homework 2                                        |
| 10.12.2023 | Analysis                                          |
| 17.12.2023 | Create Data Story & Visualization                |
| 20.12.2024 | Milestone 3 Deadline                             |

## 🤝 Team Organization

| Team Member | Responsibilities                                   |
|-------------|----------------------------------------------------------------|
| Casimir     | Website, Coordination & Assistance     			               |
| Jakob       | Seasonality Analysis & Metric definition                       |
| Jeanne      | Seasonal vs. Unseasonal Beers                                  |
| Nicolas     | Meteo Analysis                                                 |
| Tim         | Seasonal vs. Unseasonal Beers, Team Leader, Repo Organizer     |

The team creates the data story and visualizations in a collaborative manner.



### Dataset

We are working with a beer review data set [[1](https://drive.google.com/drive/folders/1Wz6D2FM25ydFw_-41I9uTwG9uNsN4TCF)].


- `data`
    Here the raw and the processed data are stored
- `docs`
    Contains story telling text
- `notebooks`
    Data analysis
- `results`
    Plots and final data is stored here

##  🛠️ Installation

### Pull repository

1. Create personal access token (PAT) on Gitlab.
2. Clone repository with HTTPS using git. You will have to enter your PAT for authentification. 
    ```bash
    git clone https://github.com/epfl-ada/ada-2024-project-leonardodatavinci.git
    ```
 
### Environment setup

1. Ensure miniconda is installed on your system. 
2. Ensure mamba is installed in (base), the default conda environment. 
    ```bash
    conda install mamba -n base -c conda-forge
    ```
3. Create environment using mamba (conda would take very long to resolve dependencies) from `environment.yaml`. 
    ```bash
    mamba env create -f environment.yaml
    ```

## 🚀 Usage


### Running the code

1. Activate conda environemnt.
    ```bash
    conda activate ada-project
    ```
2. Call run.py from the project root directory.
    ```bash
    python run.py
    ```

## 📚 Libraries Used
- Data Manipulation:
    - [Pandas](https://pandas.pydata.org/)
    - [Numpy](https://numpy.org/)
- Data Visualization:
    - [Matplotlib](https://matplotlib.org/)
    - [Plotly](https://plotly.com/)
    - [Seaborn](https://seaborn.pydata.org/)
    - [Folium](https://python-visualization.github.io/folium/)
- Statistical Models:
    - [Scikit-learn](https://scikit-learn.org/stable/)
    - [Statsmodels](https://www.statsmodels.org/stable/index.html)
- Productivity:
    - [IPython](https://ipython.org/)
    - [Jupyter](https://jupyter.org/)
    - [Tqdm](https://tqdm.github.io/)
- Utilities:
    - [Pycountry](https://pypi.org/project/pycountry/)

## 👤 Authors and acknowledgment

- [Tim Kluser](https://github.com/klusertim)
- [Jakob Sebastian Behler](https://github.com/jakobbehler)
- [Jeanne Noëline Anémone Oeuvray](https://github.com/oeuvray)
- [Casimir Maximilian Nüsperling](https://github.com/cmaximilian)
- [Nicolas Filimonov](https://github.com/Rayjine)
