import pandas as pd

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
    Load an Excel file and extract its data.

    Parameters:
    file_path (str): Path to the Excel file to load.

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
    Add weather data to the beer data.
    
    Parameters:
    df_beer (pd.DataFrame): DataFrame with columns 'State', 'Year', 'Month'
    df_weather (pd.DataFrame): DataFrame with columns 'state', 'year', 'month', and 'value'
    
    Returns:
    pd.DataFrame: A DataFrame with the weather data added to the beer data
    """
    df_beer[column_name] = df_beer.apply(lambda row: value_from_date_and_state(df_weather, row['year'], row['month'], row['state']), axis=1)
    return df_beer
