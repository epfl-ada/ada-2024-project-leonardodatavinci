import pandas as pd
from src.utils.preProcessingHelper import merge_with_states
import plotly.express as px
from scipy.stats import pearsonr, stats

import numpy as np
np.random.seed(42)

from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, mutual_info_score, normalized_mutual_info_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots




def add_population_to_climate_data(climate_df_path, population_df_path):
    """
    Load climate and population data, and add population column to climate data based on string matching. Match based on county name.
    
    Parameters:
    climate_df_path (str): Path to the climate data CSV file.
    population_df_path (str): Path to the population data Excel file.
    
    Returns:
    pd.DataFrame: A DataFrame with population data added to the climate data per county.
    """
    
    # Load the data
    countiesClimate_df = pd.read_csv(climate_df_path)
    countiesPopulation_df = pd.read_excel(population_df_path)

    # Rename column in climate data
    countiesClimate_df.rename(columns={"County Name": "County"}, inplace=True)
    
    # Define the function to find population based on partial matching
    def find_population(row):
        # Find the matching row in countiesPopulation_df where countyClimate name is in countyPopulation name
        match = countiesPopulation_df[countiesPopulation_df['County'].str.contains(row['County'], case=False, na=False)]
        # Return the population if there's a match, otherwise return None
        return match['Population'].values[0] if not match.empty else None
    
    # Apply the function to add the population to countiesClimate_df
    countiesClimate_df['Population'] = countiesClimate_df.apply(find_population, axis=1)
    
    return countiesClimate_df


def get_most_populated_climate_zone(climate_df):
    """
    Group climate data by State and Climate Zone, summing populations, 
    and finding the most populated climate zone for each state.
    
    Parameters:
    climate_df (pd.DataFrame): A DataFrame with 'State', 'IECC Climate Zone', and 'Population' columns.
    
    Returns:
    pd.DataFrame: A DataFrame with the most populated climate zone for each state.
    """
    
    # Group by 'State' and 'IECC Climate Zone', summing populations within each group
    state_climate_pop = climate_df.groupby(['State', 'IECC Climate Zone'], as_index=False)['Population'].sum()

    # For each state, find the climate zone with the highest population
    most_populated_climate_zone = state_climate_pop.loc[state_climate_pop.groupby('State')['Population'].idxmax()]
    
    return most_populated_climate_zone


def extract_data_from_excel(file_path: str) -> pd.DataFrame:
    """
    Load data from an Excel file and reshape it to a long format.
    
    This function loads data from an Excel file and reshapes it to a long format, 
    where each row represents a specific value for a specific year and month.
    
    The function assumes the first sheet in the Excel file contains the data.
    
    Args:
    file_path (str): The path to the Excel file to load.

    Returns:
    pd.DataFrame: A DataFrame with the data from the Excel file.
    """
    # Load the Excel file, assuming the first sheet is relevant
    df = pd.read_excel(file_path)

    # Melt the dataframe to reshape it from wide to long format
    # The `melt` function will create columns for 'Year-Month' and data values
    df_melted = df.melt(id_vars=['code', 'name'], var_name='Year-Month', value_name="value")

    # Split 'Year-Month' column into separate 'Year' and 'Month' columns
    df_melted[['year', 'month']] = df_melted['Year-Month'].str.split('-', expand=True)
    df_melted['year'] = df_melted['year'].astype(int)       # Convert year to integer
    df_melted['month'] = df_melted['month'].astype(int)      # Convert month to integer

    # Rename columns for clarity
    df_melted = df_melted.rename(columns={'name': 'state'})

    # Drop the 'Year-Month' column as it's no longer needed
    df_melted = df_melted.drop(columns=['Year-Month'])

    return df_melted

def value_from_date_and_state(df: pd.DataFrame, year: int, month: int, state: str = "United States of America") -> float:
    """
    Return the value from the specified date and state.
    
    Parameters:
    df (pd.DataFrame): DataFrame with columns 'state', 'year', 'month', and 'value'
    state (str): State name
    year (int): Year of interest
    month (int): Month of interest
    
    Returns:
    float: Value from the specified date and state, or NaN if no match is found
    """
    # Filter the DataFrame to get the matching row
    match = df[(df['state'] == state) & (df['year'] == year) & (df['month'] == month)]
    
    # Check if a match was found and return the value, else return NaN
    if not match.empty:
        return match["value"].values[0]
    else:
        return float('nan')  # Return NaN if no match found


def apply_value_from_date_and_state(df_beer: pd.DataFrame, df_weather: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Add weather data to the beer data by merging DataFrames.

    Parameters:
    df_beer (pd.DataFrame): DataFrame with columns 'State', 'Year', 'Month'
    df_weather (pd.DataFrame): DataFrame with columns 'state', 'year', 'month', and 'value'
    
    Returns:
    pd.DataFrame: A DataFrame with the weather data added to the beer data
    """
    # Rename columns in df_beer to match df_weather for merging
    df_beer_renamed = df_beer.rename(columns={'State': 'state', 'Year': 'year', 'Month': 'month'})
    
    # Perform a left merge to add the 'value' column from df_weather to df_beer
    df_merged = df_beer_renamed.merge(df_weather[['state', 'year', 'month', 'value']], 
                                      on=['state', 'year', 'month'], 
                                      how='left')
    
    # Rename the 'value' column in the merged DataFrame to the specified column_name
    df_merged = df_merged.rename(columns={'value': column_name})
    
    return df_merged

def filter_by_review_count(df, n):
    filtered_df = df[df['count'] > n]
    states_with_multiple_entries = filtered_df['state'].value_counts()
    states_to_keep = states_with_multiple_entries[states_with_multiple_entries > 1].index
    filtered_df = filtered_df[filtered_df['state'].isin(states_to_keep)]
    return filtered_df

def generate_choropleth(data, color_column='correlation', p_value_threshold=0.05):
    """
    Generates a choropleth map for the given data.

    Parameters:
        data (pd.DataFrame): The input data containing correlation and state information.
        title (str): The title of the choropleth map.
        color_column (str): The column to use for coloring the map. Default is 'correlation'.
        p_value_threshold (float): The threshold for p-value to filter significant correlations. Default is 0.05.

    Returns:
        plotly.graph_objects.Figure: The generated choropleth map.
    """
    # Filter the data by p-value threshold
    filtered_data = merge_with_states(data[data['p_value'] <= p_value_threshold])

    # Define hover data customization
    hover_data = {
        'state': True,
        color_column: ':.2f',
        'p_value': ':.2e',
        'abbreviation': False
    }

    # Add 'count' to hover data if present in the dataset
    if 'count' in filtered_data.columns:
        hover_data['count'] = True

    # Create the choropleth map
    fig = px.choropleth(
        filtered_data, 
        locations='abbreviation', 
        locationmode='USA-states', 
        color=color_column, 
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0,
        range_color=[-1, 1],  # Set the scale range
        scope="usa",
        hover_data=hover_data
    )

    return fig


def calculate_correlation(df, x_column, y_column, group_column, p_value_threshold=None):
    """
    Calculates Pearson correlation between two columns grouped by a specified column.

    Parameters:
        df (pd.DataFrame): The input dataframe.
        x_column (str): The first column for correlation.
        y_column (str): The second column for correlation.
        group_column (str): The column to group by.
        p_value_threshold (float, optional): If specified, filters results by p-value threshold.

    Returns:
        pd.DataFrame: A dataframe with correlation, p-value, and count.
    """
    corr_df = df.groupby(group_column).apply(
        lambda x: pd.Series(
            pearsonr(x[x_column], x[y_column]),
            index=['correlation', 'p_value']
        ),
        include_groups=False
    )
    corr_df['count'] = df.groupby(group_column)[x_column].count().values
    corr_df.reset_index(inplace=True)

    if p_value_threshold is not None:
        corr_df = corr_df[corr_df['p_value'] <= p_value_threshold]

    return corr_df

def merge_with_state_data(main_df, additional_df, merge_columns, drop_columns=None, rename_columns=None):
    """
    Merges a main dataframe with additional state-related data.

    Parameters:
        main_df (pd.DataFrame): The primary dataframe.
        additional_df (pd.DataFrame): The additional dataframe to merge with.
        merge_columns (list): List of columns to merge on.
        drop_columns (list, optional): Columns to drop after merging.
        rename_columns (dict, optional): Mapping of columns to rename.

    Returns:
        pd.DataFrame: The merged dataframe.
    """
    merged_df = main_df.merge(additional_df, on=merge_columns, how='left')
    
    if drop_columns:
        merged_df.drop(columns=drop_columns, inplace=True)
    
    if rename_columns:
        merged_df.rename(columns=rename_columns, inplace=True)

    return merged_df

# Function to calculate p-values for correlations

def calculate_p_values(data, x_vars=None, y_vars=None):
    """
    Calculate p-values for Pearson correlation between columns in a dataframe.

    Parameters:
        data (pd.DataFrame): The input dataframe.
        x_vars (list, optional): List of column names for x variables. If None, all columns are used.
        y_vars (list, optional): List of column names for y variables. If None, all columns are used.

    Returns:
        pd.DataFrame: A dataframe with p-values.
            - If x_vars and y_vars are None, returns a square dataframe with p-values for all columns.
            - If x_vars and y_vars are provided, returns a dataframe where the index is y_vars and the columns are x_vars.
    """
    if x_vars is None and y_vars is None:
        # Full correlation matrix
        data = data.dropna()
        p_values = pd.DataFrame(np.zeros((data.shape[1], data.shape[1])), 
                                columns=data.columns, index=data.columns)
        for col1 in data.columns:
            for col2 in data.columns:
                _, p_value = stats.pearsonr(data[col1], data[col2])
                p_values.loc[col1, col2] = p_value
    else:
        # Correlation matrix for specified x_vars and y_vars
        p_values = pd.DataFrame(index=y_vars, columns=x_vars)
        for y in y_vars:
            for x in x_vars:
                _, p_value = stats.pearsonr(data[x], data[y])
                p_values.loc[y, x] = p_value
    return p_values.round(4)


def calculate_correlation_stats(df, target='rating'):
    """
    Calculates Pearson correlation, R-squared, and ANOVA statistics between a target column and all other numeric columns in a dataframe.

    Parameters:
        df (pd.DataFrame): The input dataframe.
        target (str): The target column name. Defaults to 'rating'.

    Returns:
        dict: A dictionary with the following keys for each numeric column:
            'correlation': Pearson correlation coefficient.
            'p_value': p-value for the Pearson correlation.
            'r_squared': R-squared value.
            'f_statistic': F-statistic from ANOVA.
            'anova_p': p-value from ANOVA.
    """
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    numeric_columns = [col for col in numeric_columns if col != target]
    
    results = {}
    for column in numeric_columns:
        # Calculate Pearson correlation
        correlation, p_value = stats.pearsonr(df[target], df[column])
        
        # Calculate R-squared
        r_squared = correlation ** 2
        
        # Perform ANOVA
        groups = []
        for value in df[column].unique():
            group = df[df[column] == value][target]
            groups.append(group)
        f_stat, anova_p = stats.f_oneway(*groups)
        
        results[column] = {
            'correlation': correlation,
            'p_value': p_value,
            'r_squared': r_squared,
            'f_statistic': f_stat,
            'anova_p': anova_p
        }
    
    return results

def perform_cluster_analysis(data, vars_to_cluster, n_clusters_range=range(2, 7)):
    """
    Performs k-means clustering on a given set of variables.

    Parameters:
        data (pd.DataFrame): The input dataframe.
        vars_to_cluster (list): List of column names to cluster.
        n_clusters_range (list, optional): List of numbers of clusters to evaluate. Defaults to range(2, 7).

    Returns:
        tuple: Contains the following:
            - data_with_clusters (pd.DataFrame): The original dataframe with a new 'Cluster' column.
            - cluster_stats (pd.DataFrame): A dataframe with mean and standard deviation of the clustered variables for each cluster.
            - f_stat (float): The F-statistic from ANOVA.
            - p_value (float): The p-value from ANOVA.

    Notes:
        ANOVA is used to test the significance of the clusters. The F-statistic and p-value are returned.
    """
    X = data[vars_to_cluster].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=vars_to_cluster)
    
    # Evaluate different numbers of clusters
    metrics = []
    for n_clusters in n_clusters_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Calculate evaluation metrics
        silhouette = silhouette_score(X_scaled, clusters)
        calinski = calinski_harabasz_score(X_scaled, clusters)
        
        metrics.append({
            'n_clusters': n_clusters,
            'silhouette_score': silhouette,
            'calinski_score': calinski
        })
    
    # Plot evaluation metrics
    metrics_df = pd.DataFrame(metrics)
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Silhouette Score', 'Calinski-Harabasz Score'))
    
    fig.add_trace(
        go.Scatter(x=metrics_df['n_clusters'], y=metrics_df['silhouette_score'],
                  mode='lines+markers', name='Silhouette'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=metrics_df['n_clusters'], y=metrics_df['calinski_score'],
                  mode='lines+markers', name='Calinski-Harabasz'),
        row=1, col=2
    )
    
    fig.update_layout(title='Cluster Evaluation Metrics')
    fig.write_html("src/plotly-html-graphs/nicolas/cluster_evaluation_metrics.html")
    fig.show()
    
    # Choose optimal number of clusters (using silhouette score)
    optimal_n_clusters = metrics_df.loc[metrics_df['silhouette_score'].idxmax(), 'n_clusters']
    
    # Perform clustering with optimal number
    kmeans = KMeans(n_clusters=int(optimal_n_clusters), random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Add clusters to original data
    data_with_clusters = data.copy()
    data_with_clusters['Cluster'] = clusters
    
    # Analyze clusters
    cluster_stats = data_with_clusters.groupby('Cluster')[vars_to_cluster].agg(['mean', 'std']).round(3)
    
    # Create 3D scatter plot
    fig = px.scatter_3d(data_with_clusters, 
                        x='Temperature', 
                        y='Precipitation', 
                        z='rating',
                        color='Cluster',
                        title='3D Cluster Visualization')
    fig.write_html("src/plotly-html-graphs/nicolas/cluster_3d.html")
    fig.show()
    
    # Perform ANOVA to test significance of clusters
    f_stat, p_value = stats.f_oneway(*[group['rating'].values 
                                     for name, group in data_with_clusters.groupby('Cluster')])
    
    return data_with_clusters, cluster_stats, f_stat, p_value

def compute_mutual_information(data, target='rating', n_bins=10):
    """
    Compute mutual information between target and other variables
    
    Parameters:
    -----------
    data : pandas DataFrame
    target : str, target variable name
    n_bins : int, number of bins for discretization
    
    Returns:
    --------
    DataFrame with MI scores and normalized MI scores
    """
    # Get numeric columns
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    numeric_cols = [col for col in numeric_cols if col != target]
    
    # Initialize discretizer
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
    
    # Discretize target variable
    target_discrete = discretizer.fit_transform(data[[target]])
    
    # Calculate MI and normalized MI for each variable
    mi_scores = []
    for col in numeric_cols:
        # Discretize feature
        feature_discrete = discretizer.fit_transform(data[[col]])
        
        # Calculate MI scores
        mi = mutual_info_score(target_discrete.ravel(), feature_discrete.ravel())
        nmi = normalized_mutual_info_score(target_discrete.ravel(), feature_discrete.ravel())
        
        mi_scores.append({
            'Variable': col,
            'MI': mi,
            'Normalized_MI': nmi
        })
    
    # Convert to DataFrame
    mi_df = pd.DataFrame(mi_scores)
    mi_df = mi_df.sort_values('MI', ascending=False)
    
    return mi_df

def analyze_mi_relationship(data, feature, target='rating', n_bins=10):
    """
    Analyze the relationship between a feature and target using mutual information.

    Parameters:
    -----------
    data : pandas DataFrame
        The input dataframe.
    feature : str
        The column name of the feature.
    target : str, optional
        The column name of the target. Defaults to 'rating'.
    n_bins : int, optional
        The number of bins for discretization. Defaults to 10.

    Returns:
    -------
    pd.DataFrame
        A normalized contingency matrix with the following structure:
            Index: feature bins
            Columns: target bins
            Values: probability of each combination
    """
    # Create bins for both variables
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
    feature_bins = discretizer.fit_transform(data[[feature]])
    target_bins = discretizer.fit_transform(data[[target]])
    
    # Create contingency matrix
    contingency = pd.crosstab(
        pd.Categorical(feature_bins.ravel(), categories=range(n_bins)),
        pd.Categorical(target_bins.ravel(), categories=range(n_bins))
    )
    
    # Normalize contingency matrix
    contingency_norm = contingency / contingency.sum().sum()
    
    return contingency_norm

def compute_gain_ratio(data, feature, target='rating', n_bins=10):
    """
    Compute the gain ratio between a feature and target using mutual information.

    Parameters:
        data (pd.DataFrame): The input dataframe.
        feature (str): The column name of the feature.
        target (str, optional): The column name of the target. Defaults to 'rating'.
        n_bins (int, optional): The number of bins for discretization. Defaults to 10.

    Returns:
        dict: A dictionary with the following keys:
            'Mutual_Information': The mutual information between the feature and target.
            'Feature_Entropy': The entropy of the feature.
            'Target_Entropy': The entropy of the target.
            'Gain_Ratio': The gain ratio between the feature and target.
    """
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
    
    # Discretize variables
    feature_discrete = discretizer.fit_transform(data[[feature]])
    target_discrete = discretizer.fit_transform(data[[target]])
    
    # Calculate entropies
    def entropy(labels):
        _, counts = np.unique(labels, return_counts=True)
        probabilities = counts / len(labels)
        return -np.sum(probabilities * np.log2(probabilities))
    
    target_entropy = entropy(target_discrete)
    feature_entropy = entropy(feature_discrete)
    
    # Calculate mutual information
    mi = mutual_info_score(target_discrete.ravel(), feature_discrete.ravel())
    
    # Calculate gain ratio
    gain_ratio = mi / feature_entropy if feature_entropy != 0 else 0
    
    return {
        'Mutual_Information': mi,
        'Feature_Entropy': feature_entropy,
        'Target_Entropy': target_entropy,
        'Gain_Ratio': gain_ratio
    }