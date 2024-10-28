# 🧬 LeonardoDataVinci 🧬 

## 🔴 Description

Bla bla bla bla.

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
│       ├── beers.csv
│       ├── breweries.csv
│       ├── ratings.txt
│       ├── reviews.txt
│       └── users.csv
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

## 👤 Authors and acknowledgment
