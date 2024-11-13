import pandas as pd
import numpy as np
import plotly.express as px


def calculate_monthly_means(df, year_highpass=None):
    """
    Calculates the mean rating for each month in every year, with an optional cutoff year.

    Parameters:
    - df (pd.DataFrame): DataFrame containing 'rating', 'month', and 'year' columns.
    - year_highpass (int, optional): The cutoff year. If specified, only data from this year onward is included.

    Returns:
    - pd.DataFrame: A DataFrame with 'year', 'month', and 'mean_rating' columns.
    """
    # Apply the year cutoff if specified
    if year_highpass is not None:
        df = df[df['year'] >= year_highpass]
    
    # Group by 'year' and 'month' and calculate the mean of 'rating'
    monthly_means = df.groupby(['year', 'month'])['rating'].mean().reset_index()
    
    # Rename the column for clarity
    monthly_means.rename(columns={'rating': 'mean_rating'}, inplace=True)
    
    return monthly_means

def convert_to_timeseries(df):
    """
    Adds a 'month_number' column to the DataFrame, representing each month as a sequential month count.
    For example, month 1 of the second year will be month 13.

    Parameters:
    - df (pd.DataFrame): DataFrame containing 'year' and 'month' columns.

    Returns:
    - pd.DataFrame: The input DataFrame with an added 'month_number' column.
    """
    # Calculate the minimum year in the dataset to use as a reference
    start_year = df['year'].min()
    
    # Calculate the month number as the difference in years times 12 plus the month
    df['month_number'] = (df['year'] - start_year) * 12 + df['month']
    
    df = df.sort_values('month_number')
    return df



def plot_frequency_spectrum(ratings, cutoff_freq=0.15):
    """
    Perform Fourier analysis on the input ratings time series and plot the frequency spectrum.
    
    Parameters:
    - ratings (array-like): The time series of ratings (monthly mean ratings).
    - cutoff_freq (float): The frequency cutoff for the plot (default is 0.15).
    
    Returns:
    - None: Displays the frequency spectrum plot.
    """
    # Apply Fourier Transform
    fft_result = np.fft.fft(ratings)
    
    # Calculate Frequencies
    freqs = np.fft.fftfreq(len(ratings), d=1)  # d=1 because we're using months as the time unit
    
    # Calculate the Magnitudes
    magnitudes = np.abs(fft_result)
    
    # Create a DataFrame for the frequency and magnitude data
    df_freq = pd.DataFrame({
        'Frequency (cycles per month)': freqs[:len(freqs) // 2],  # Only positive frequencies
        'Magnitude': magnitudes[:len(magnitudes) // 2]
    })
    
    # Filter the frequencies and magnitudes to only include values up to the cutoff frequency
    df_freq_filtered = df_freq[df_freq['Frequency (cycles per month)'] <= cutoff_freq]
    
    # Plot using Plotly Express with log scale for Magnitude
    fig = px.line(df_freq_filtered, x='Frequency (cycles per month)', y='Magnitude', 
                  title="Frequency Spectrum of Beer Ratings Time Series",
                  labels={'Frequency (cycles per month)': 'Frequency', 'Magnitude': 'Magnitude'})
    
    # Set y-axis to log scale
    fig.update_layout(yaxis_type="log")
    
    # Show the plot
    fig.show()