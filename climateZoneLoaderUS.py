# file: data_processing.py

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