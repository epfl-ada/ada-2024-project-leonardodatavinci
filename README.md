# 🍻 LeonardoDataVinci 🍻

## 🔴 Abstract

This project investigates seasonal fluctuations in beer ratings using the reviews on the forum “BeerAdvocate” from 2002 until 2017. We can observe that, averaged over all years, ratings vary by season.

This is interesting, because ideally beer ratings would solely reflect the beer's intrinsic qualities and remain unaffected by external factors. 

Our goal is to identify some of those external factors. For example, seasonal trends may exist where certain beer styles are more popular in summer than in winter, or temperature variations could affect how a beer is perceived and rated.

Our analysis is divided into two steps. First, we explore the observed seasonal patterns, developing metrics to quantify seasonality and determine whether these patterns represent true, recurring annual phenomena or are simply artifacts of averaging ratings over multiple years. In the second part, we investigate the reasons behind these seasonal changes in ratings. Specifically, we examine if seasonal variations are driven by changes in the types of beer consumed or shifts in how people rate. Additionally, we investigate if daily weather on the state level can be correlated to a change in ratings.

## ❓ Research Questions

- Does the observed seasonal pattern in beer ratings reflect a true year-to-year recurrence, or is it an artifact of averaging ratings across many years?
- How do seasonal and year-round beers influence global beer rating trends, and to what extent do seasonal beers account for the observed seasonal differences in beer ratings compared to year-round beers?
- Can we establish correlations between daily weather and ratings?

## 📂 Additional Datasets

To enrich our analysis and provide deeper insights, we have incorporated supplementary datasets beyond the primary beer ratings data:

- **World Bank Climate Data**: Monthly average temperature and precipitation data by state, sourced from the [Climate Change Knowledge Portal](https://climateknowledgeportal.worldbank.org/download-data).
- **County Climate Zones**: Classification of U.S. counties into climate zones to analyze regional weather patterns and their influence on beer ratings. Data sourced from the [US Department of Energy](https://www.energy.gov/sites/prod/files/2015/10/f27/ba_climate_region_guide_7.3.pdf).
- **US Census Population Data**: County-level population data used to weight climate zone analysis for accurate state-level metrics. Data obtained from the [U.S. Census Bureau](https://www.census.gov/data/tables/time-series/demo/popest/2020s-counties-total.html).
- **Formatted Climate Zones Table**: A processed version of county climate zone data, sourced from [this GitHub Gist](https://gist.github.com/philngo/d3e251040569dba67942#file-climate_zones-csv).

## ⚙️ Methods
We focus our work on the reviews on the forum “BeerAdvocate” from 2002 until 2017, in the US.

- We specify only the US because there is enough data and only the northern hemisphere! And state-specific, so nice granularity!
- BeerAdvocate, because more data and focus on US

### 0. Intro: Show That There Is Something Going On!

**Question:**  
Why is it worth investigating  seasonal fluctuations in beer ratings?

**Method:**  

As we have a timestamp for each review and rating, we can sort them by seasons and years and show that there are significant seasonal fluctuations in the data by applying a statistical test.  

To measure seasonality in beer ratings, we introduced the Seasonality Score, calculated as the difference between the mean summer (June-August) and mean winter (December-February) ratings. This score helps quantify seasonal fluctuations in beer ratings.

We will refine this metric by calculating the seasonality score for each U.S. state, combining it with weather data to explore how climate influences beer ratings regionally.

We also plan to experiment with different time spans, as averaging over three months might smooth out important seasonal variations that could be more noticeable with other groupings.

To validate seasonality, we’ll use statistical tests (e.g., Kruskal-Wallis and t-test) on the main seasonality plot, comparing seasonal fluctuations across months and years for different beer types.
	
### 1. Year-to-Year Seasonal Patterns

**Question:**  
Does the observed seasonal pattern in beer ratings reflect a true year-to-year recurrence, or is it an artifact of averaging ratings across many years?

**Method:**  
To identify seasonality, we apply Fourier Analysis and examine the frequency spectrum for significant patterns. First, we calculate the monthly average rating across all beers, creating a time series of average monthly ratings over the entire period the app has been in operation. We then apply the Fast Fourier Transform (FFT) to this time series to analyze its frequency spectrum. To confirm seasonality, we look for a significant peak at a frequency of 0.083 cycles per month, corresponding to a 12-month period. 
In subsequent analyses, we perform the same procedure on the first derivative of the signal, allowing us to focus purely on the rates of change. Additionally, we analyze individual states and beer types
To get even clearer knowledge, STL (Seasonal and Trend decomposition) can be used to decompose the time series into trend, seasonal, and residual components, allowing us to isolate and verify the 12-month seasonal pattern in the ratings data.



### 2. Seasonal vs. Year-Round Beers

**Question:**   How do seasonal and year-round beers influence global beer rating trends, and to what extent do seasonal beers account for the observed seasonal differences in beer ratings compared to year-round beers?  

**Method:**
- **Identify Seasonal and Unseasonal Popular Beers**  
Identify seasonal popular beers (frequently rated only at certain periods) and unseasonal popular beers (always maintain the same proportion of popularity). Assess the differences across months using statistical tests.

- **Check for Fluctuations in Ratings**  
For both seasonal and unseasonal beer styles, check how their ratings fluctuate across months. Compute statistical tests to see if the fluctuation of these beer styles is significantly different across seasons.

- **Identify and Analyze Impactful Beers**  
Define a subset of beers as "impactful," i.e., beers with high popularity during certain periods and high deviation from the average rating. Remove these impactful beers from the data and plot the average ratings to see if there is significantly less seasonal fluctuation after their removal.

### 3.  Weather Influence By State

**Question:**  
Does state weather impact beer ratings and its seasonality?

**Method:**  
Correlate ratings with average state temperatures & precipitation. Use weighted mean of weather station data based on US county populations.


**Methodology:**  
1. Data Acquisition:
    - Weather Data: Download monthly average temperature and precipitation data for each U.S. state from the Climate Change Knowledge Portal.
    - Beer Ratings: Compile beer ratings with corresponding timestamps and state information.
2. Data Processing:
    - Temporal Alignment: Align beer ratings with corresponding monthly weather data based on the review dates.
    - State Aggregation: Aggregate beer ratings and weather data at the state level to facilitate comparative analysis.
3. Correlation Analysis:
    - Temperature and Ratings: Calculate the correlation between average monthly temperatures and beer ratings for each state.
    - Precipitation and Ratings: Assess the relationship between monthly precipitation levels and beer ratings.
    - Seasonal Pattern Analysis:
    - Trend Identification: Analyze how seasonal weather variations influence beer ratings across different states.
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
|-------------|----------------------------------------------------|
| Casimir     | Part 3                                             |
| Jakob       | Part 1                                             |
| Jeanne      | Part 2                                             |
| Nicolas     | Part 3                                             |
| Tim         | Part 0, Team Leader, Repo Organizer                |

The team will create the data story and visualizations in a collaborative manner.



### Dataset

We are working with a beer review data set [[1](https://drive.google.com/drive/folders/1Wz6D2FM25ydFw_-41I9uTwG9uNsN4TCF)].

## 🏗️ Project Structure
```
├── assets/
│   ├── css/
│   │   ├── styles.css
│   ├── img/
│   │   ├── favicon.png
│   ├── js/
│   │   ├── scale.fix.js
├── data/
│   ├── BeerAdvocate/
│   │   ├── beers.csv
│   │   ├── breweries.csv
│   │   ├── ratings.txt
│   │   ├── reviews.txt
│   │   └── users.csv
│   ├── climatezones/
│   │   ├── Map_KG-Global/
│   │   │   ├── KG_1986-2010.grd
│   │   │   ├── KG_1986-2010.txt
│   │   ├── ne_10m_admin_0_countries/
│   │   │   ├── ne_10m_admin_0_countries.shp
│   │   ├── climate_zones.csv
│   │   ├── countyPopulation.xlsx
│   │   ├── Koeppen-Geiger-ASCII.txt
│   │   ├── stateAbbreviations.csv
│   │   └── weather_stations_world.txt
│   ├── matched_beer_data/
│   │   ├── beers.csv
│   │   ├── breweries.csv
│   │   ├── ratings.csv
│   │   ├── ratings_ba.txt
│   │   └── ratings_rb.txt
│   │   ├── ratings_with_text_ba.txt
│   │   ├── ratings_with_text_rb.txt
│   │   ├── users.csv
│   │   └── users_approx.csv
│   └── RateBeer/
│   │   ├── beers.csv
│   │   ├── breweries.csv
│   │   ├── ratings.txt
│   │   ├── reviews.txt
│   │   └── users.csv
│   └── weather-data/
│       ├── cru-x0.5_timeseries_tas_timeseries_monthly_1901-2022_mean_historical_cru_ts4.07_mean.xlsx
│       ├── cru-x0.5_timeseries_pr_timeseries_monthly_1901-2022_mean_historical_cru_ts4.07_mean.xlsx
│       ├── us-precipitations.csv
│       └── us-temperatures.csv
├── illustrations/
│   └── Nicolas/
│       ├── avg_temp_state.html
│       ├── climate_map.html
│       ├── heatmap.html
│       ├── number_of_rows_per_month_year.html
│       ├── temperature.html
│       ├── timelapse_temperature.html
│       ├── timeline.html
│       └── weather_stations.html
├── papers/
│   ├── icdm2012.pdf
|   └── Lederrey-West_WWW-18.pdf
├── src/
│   └── utils/
│       ├── locationHelper.py
│       ├── preProcessingHelper.py
│       ├── USClimateAndWeatherHelper.py
│       ├── fourierAnalysis.py
│       └── fourierHelper.py
├── tests/
│   └── PreProcessing.ipynb
├── .gitignore 
├── README.md
├── _config.yml
├── LICENSE
├── index.html
├── Gemfile
└── environment.yaml
```

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
- Machine Learning:
    - [Scikit-learn](https://scikit-learn.org/stable/)
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
