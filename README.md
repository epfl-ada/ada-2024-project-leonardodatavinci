# 🧬 LeonardoDataVinci 🧬 

## 🔴 Description

P2 deliverable (done as a team): GitHub repository with the following:

- Readme.md file containing the detailed project proposal (up to 1000 words). Your README.md should contain:
        Title
        Abstract: A 150 word description of the project idea and goals. What’s the motivation behind your project? What story would you like to tell, and why?
        Research Questions: A list of research questions you would like to address during the project.
        Proposed additional datasets (if any): List the additional dataset(s) you want to use (if any), and some ideas on how you expect to get, manage, process, and enrich it/them. Show us that you’ve read the docs and some examples, and that you have a clear idea on what to expect. Discuss data size and format if relevant. It is your responsibility to check that what you propose is feasible.
        Methods
        Proposed timeline
        Organization within the team: A list of internal milestones up until project Milestone P3.
        Questions for TAs (optional): Add here any questions you have for us related to the proposed project.
- GitHub repository should be well structured and contain all the code for the initial analyses and data handling pipelines. For structure, please use this repository as a template
- Notebook presenting the initial results to us. We will grade the correctness, quality of code, and quality of textual descriptions. There should be a single Jupyter notebook containing the main results. The implementation of the main logic should be contained in external scripts/modules that will be called from the notebook.


### Dataset

We are working with a beer review data set [[1](https://drive.google.com/drive/folders/1Wz6D2FM25ydFw_-41I9uTwG9uNsN4TCF)].

## 📁 Project structure
```
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
├── docs/
├── notebooks/
├── results/
├── .gitignore 
├── README.md
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
    git clone XXXXXXXXXXXXXXXXXXXXXXX
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

### Pipeline

The pipeline is defined in `run.py` and follows roughly the steps outlined below. In order for certain parts of the pipeline to run you can comment out steps.

1. Blabla
2. Blabla
    1. Blabla
    2. Blabla
3. Run blabla
4. Analyse blabla results

### Running the code

1. Activate conda environemnt.
    ```bash
    conda activate spatialCNVenv
    ```
2. Call run.py from the project root directory.
    ```bash
    python run.py
    ```
    
## 🎯 Roadmap

1. [x] Blabla

    Blabla.

2. [ ] Blabla

    Blabla

## Libraries Used
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
