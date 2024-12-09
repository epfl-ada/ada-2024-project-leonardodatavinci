import pandas as pd
import numpy as np
import plotly.express as px
from statsmodels.tsa.seasonal import STL
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# PREPROCESSING FUNCTIONS ---------------------------------------------------------------------------------------------
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

def time_formatting_for_stl(df):
    """
    Adds a 'date' column to the DataFrame based on 'year' and 'month'.
    The date is always set to the first day of the month.
    
    Args:
        df (pd.DataFrame): DataFrame with 'year' and 'month' columns.
    
    Returns:
        pd.DataFrame: Updated DataFrame with a 'date' column.
    """
    df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
    stl_data = df.set_index('date')[['mean_rating']]

    return stl_data

def time_formatting_of_timeseries(df, data_column_name):
    """
    Adds a 'date' column to the DataFrame based on 'year' and 'month'.
    The date is always set to the first day of the month.
    
    Args:
        df (pd.DataFrame): DataFrame with 'year' and 'month' columns.
    
    Returns:
        pd.DataFrame: Updated DataFrame with a 'date' column.
    """
    df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
    stl_data = df.set_index('date')[[data_column_name]]

    return stl_data

def data_preprocessing_mean_rating_per_month(df, data_column_name):
    # must contain 'rating', 'month', and 'year' columns.
    df_means = calculate_monthly_means(df)
    df_means_timeseries = convert_to_timeseries(df_means)
    stl_data = time_formatting_of_timeseries(df_means_timeseries, data_column_name)

    return stl_data

def process_to_timeseries_number_of_ratings(df):
    """
    Processes the given DataFrame to return a timeseries of the number of ratings per month.
    The output DataFrame contains ['year', 'month', 'N_ratings'].

    Input:
    - df: A DataFrame containing 'year' and 'month' columns (one row per rating).

    Output:
    - A DataFrame with ['year', 'month', 'N_ratings'].
    """
    # Group by 'year' and 'month' and count the number of rows (ratings) for each month
    monthly_ratings = df.groupby(['year', 'month']).size().reset_index(name='N_ratings')

    return monthly_ratings

def data_preprocessing_number_of_ratings_per_month(df):

    # must contain 'rating', 'month', and 'year' columns.
    df_number_ratings = process_to_timeseries_number_of_ratings(df)
    df_number_ratings_timeseries = convert_to_timeseries(df_number_ratings)
    stl_data = time_formatting_of_timeseries(df_number_ratings_timeseries, 'N_ratings')

    return stl_data

# STL FUNCTIONS  ---------------------------------------------------------------------------------------------

def plot_STL(data):
    """
    Plots the components of an STL decomposition (Original Series, Trend, Seasonal, Residual)
    using Plotly.

    Parameters:
    ----------
    data : pandas.DataFrame
        A DataFrame containing the time series data with:
        - An index representing the time points.
        - A column named 'mean_rating' representing the series to be decomposed.

    Returns:
    -------
    None
        The function generates an interactive Plotly figure and displays it in the browser.
    """


    # Perform STL decomposition
    stl = STL(data)
    result = stl.fit()

    # Create the figure with subplots
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                        subplot_titles=("Original Series", "Trend", "Seasonal", "Residual"))

    # Add the original series
    fig.add_trace(go.Scatter(x=data.index, y=data[data.columns[0]], name='Original'), row=1, col=1)

    # Add the trend component
    fig.add_trace(go.Scatter(x=data.index, y=result.trend, name='Trend'), row=2, col=1)

    # Add the seasonal component
    fig.add_trace(go.Scatter(x=data.index, y=result.seasonal, name='Seasonal'), row=3, col=1)

    # Add the residual component
    fig.add_trace(go.Scatter(x=data.index, y=result.resid, name='Residual'), row=4, col=1)

    # Update layout
    fig.update_layout(height=800, title="STL Decomposition", showlegend=False)
    fig.update_yaxes(title_text=data.columns[0])

    # Show the plot
    fig.show()

    return fig


def calculate_avg_yearly_amplitude(df):
    """
    Calculates amplitude of every year in an oscillating signal then takes the average. 

    Parameters:
        df (pd.DataFrame): DataFrame containing 'seasonal' and 'date' columns.

    Returns:
        float: The average of yearly mean amplitudes.
    """
    # Ensure 'date' is datetime
    #df["date"] = pd.to_datetime(df["date"])
    
    # Extract the year from the date column
    df["year"] = df["date"].dt.year
    
    # Group by year and calculate the amplitude for each year
    yearly_amplitudes = (
        df.groupby("year")["seasonal"]
        .apply(lambda x: (x.max() - x.min()) / 2)  # Compute amplitude per year
    )
    
    # Calculate the average of the yearly amplitudes
    average_amplitude = yearly_amplitudes.mean()
    
    return average_amplitude

def report_STL_amplitude_seasonality_score(seasonal, print_report = True):
    """
    takes the STL.seasonal output and reports a seasonality intensity metric. 
    if print_report is true then a report will be printed
    """
    # Convert Series to DataFrame
    df = seasonal.reset_index()
    df.columns = ["date", "seasonal"]
    avg_yearly_amplitude = calculate_avg_yearly_amplitude(df)

    if print_report:
        print(f"Average Yearly Mean Amplitude of STL seasonality: {np.round(avg_yearly_amplitude, 4)}")
    
    return np.round(avg_yearly_amplitude, 4)


# Fourier Transform functions based on STL data -----------------------------------------------------------------------------------

def FFT_dataframe(df, cutoff_freq=0.5):

    """
    Perform Fourier analysis on the input ratings time series and return them as a dataframe.
    """

    # Apply Fourier Transform
    fft_result = np.fft.fft(df)

    # Calculate Frequencies
    freqs = np.fft.fftfreq(len(df), d=1)  # d=1 because we're using months as the time unit

    # Calculate the Magnitudes and Normalize
    magnitudes = np.abs(fft_result) / len(df)  # Normalize by dividing by the length

    # Create a DataFrame for frequency and magnitude data
    df_freq = pd.DataFrame({
        'Frequency (cycles per month)': freqs[:len(freqs) // 2],  # Only positive frequencies
        'Magnitude': magnitudes[:len(magnitudes) // 2]
    })
    df_freq_filtered = df_freq[df_freq['Frequency (cycles per month)'] <= cutoff_freq]

    return df_freq


def fourier_analysis(ratings, cutoff_freq=0.5):
    """
    Perform Fourier analysis on the input ratings time series
    """

    df_freq = FFT_dataframe(ratings)

    # Filter the frequencies and magnitudes to only include values up to the cutoff frequency
    df_freq_filtered = df_freq[df_freq['Frequency (cycles per month)'] <= cutoff_freq]
    return df_freq_filtered

def FFT_magnitude_of_closest_freq_to_target_freq(df_FFT, target_frequency):
    """
    finds the frequency closest to a 12-month cycle 
    (i.e., 0.083 cycles per month, corresponding to an annual cycle). The function prints out the 
    maximum frequency with its corresponding magnitude and the magnitude of the frequency closest 
    to 0.083 cycles per month.
    """
    tolerance = 1e-1  # Define tolerance for closest frequency search

    df_FFT_diff = df_FFT.copy()
    # Calculate the absolute difference between each frequency and the target frequency
    df_FFT_diff['Freq_diff'] = (df_FFT_diff['Frequency (cycles per month)'] - target_frequency).abs()

    # Find the frequency closest to 12-month cycle (0.083)
    closest_freq_row = df_FFT_diff.loc[df_FFT_diff['Freq_diff'].idxmin()]
    closest_freq = closest_freq_row['Frequency (cycles per month)']
    closest_magnitude = closest_freq_row['Magnitude']

    return closest_magnitude, closest_freq

    
def find_second_FFT_peak(df_FFT, freq_to_exlude, window_size):
    """
    to find the biggest peak that is not a given peak (here not the peak of 12-month period). This is the freq_to_exclude.
    the window_size defines the area around the freq_to_exclude that is also excluded.
    """
    # Step 1: Exclude the values around the freq_to_exclude.

    # Create a mask to exclude the highest peak and its adjacent frequencies
    df_freq_excluding_adjacent = df_FFT[
    ~df_FFT['Frequency (cycles per month)'].between(freq_to_exlude - window_size, freq_to_exlude + window_size)
    ]

    # Identify the largest magnitude from the remaining data.
    second_max_magnitude_row = df_freq_excluding_adjacent.loc[df_freq_excluding_adjacent['Magnitude'].idxmax()]
    second_max_magnitude = second_max_magnitude_row['Magnitude']
    second_max_freq = second_max_magnitude_row['Frequency (cycles per month)']

    return second_max_magnitude, second_max_freq

def report_fourier_analysis(ratings, cutoff_freq=0.15):
    """
    Perform Fourier analysis on the input ratings time series and report the frequency with 
    the maximum magnitude and the magnitude of the frequency closest to 12-month period.

    This function performs the Fourier transform of the given time series of ratings, identifies 
    the frequency with the highest magnitude, and finds the frequency closest to a 12-month cycle 
    (i.e., 0.083 cycles per month, corresponding to an annual cycle). The function prints out the 
    maximum frequency with its corresponding magnitude and the magnitude of the frequency closest 
    to 0.083 cycles per month.

    Parameters:
    - ratings (array-like): The time series of ratings (typically monthly mean ratings).
    - cutoff_freq (float): The frequency cutoff for the analysis (default is 0.15).

    Returns:
    - max_freq (float): The frequency with the maximum magnitude.
    - max_magnitude (float): The magnitude corresponding to `max_freq`.
    - closest_magnitude (float): The magnitude of the frequency closest to the 12-month cycle (0.083).
    """
    
    # Perform Fourier analysis on the input ratings
    df_freq = fourier_analysis(ratings, cutoff_freq=cutoff_freq)
    
    # Identify the frequency with the maximum magnitude
    max_magnitude_row = df_freq.loc[df_freq['Magnitude'].idxmax()]
    max_freq = max_magnitude_row['Frequency (cycles per month)']
    max_magnitude = max_magnitude_row['Magnitude']

    target_freq = 1/12
    closest_magnitude, closest_freq = FFT_magnitude_of_closest_freq_to_target_freq(df_freq, target_freq)

    # Output results

    print(f"The frequency with the maximum magnitude is {max_freq:.6f} cycles per month "
          f"with a magnitude of {max_magnitude:.6f}.")
    print(f"The magnitude of the frequency closest to a 12-month period (0.083) is: {closest_magnitude:.6f}.")
    print()
    max_period = 1/max_freq
    print(f"This means the most significant period is: {max_period:.6f} months.")
    return max_freq, max_magnitude, closest_magnitude


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
    df_freq_filtered = fourier_analysis(ratings, cutoff_freq=cutoff_freq)
    


    # Plot using Plotly Express with log scale for Magnitude
    fig = px.line(df_freq_filtered, x='Frequency (cycles per month)', y='Magnitude', 
                  title="Frequency Spectrum of Beer Ratings Time Series",
                  labels={'Frequency (cycles per month)': 'Frequency', 'Magnitude': 'Magnitude'})
    
    # Set y-axis to log scale
    fig.update_layout(yaxis_type="log")
    
    # Show the plot
    fig.show()

    return fig


# FULL PIPELINE ---------------------------------------------------------------------------------------------------------------

  
def full_seasonality_report(df):
      """
      combines all STL and Fourier Transform functions to one analysis pipeline.
      1) preps the raw ratings data
      2) splitting timeseries into seasonal and non-seasonal components
      3) gives a seasonality amplitude score based on only the seasonal component
      4) returns frequency specrum of the seasonal signal, to verify 12-month periodicity

      input: 
      -  df containing 'rating', 'month', and 'year' columns.
      output:
      -  none

      """
   
      # Perform STL decomposition
      stl = STL(df)
      result = stl.fit()
      seasonal = result.seasonal

      fig_stl = plot_STL(df)
      report_STL_amplitude_seasonality_score(seasonal, print_report = True)
      
      fig_fft = plot_frequency_spectrum(seasonal, cutoff_freq=0.5)
      report_fourier_analysis(seasonal, cutoff_freq=0.15)

      return fig_stl, fig_fft

def timeseries_seasonality_metric(df):
    """
    Calculates two metrics that quantify the seasonality of a time series.

    Parameters:
    df (pd.DataFrame): A dataframe where the index is a datetime object representing months (e.g., '1-12-2013') 
                       and the first column contains the values of interest (e.g., gene expression, temperature, etc.).

    Returns:
    tuple: 
        - A float representing the ratio of the peak corresponding to the 12-month period (seasonal frequency) to the second-highest peak in the frequency spectrum. A value greater than 5 indicates strong seasonality.
        - A float representing the mean amplitude of the seasonal component of the signal, calculated based on STL decomposition.

    Notes:
    - The function uses STL (Seasonal-Trend decomposition using Loess) to extract the seasonal component of the time series.
    - Fourier Transform (FFT) is used to identify the primary and secondary peaks in the frequency spectrum.
    - The seasonal metric is based on comparing the magnitude of the 12-month periodicity (or annual frequency) to the next largest frequency peak.
    - The mean amplitude reflects the strength of the seasonal signal over the period.
    """
    
    # extract seasonal part from signal using STL decomposition
    stl = STL(df)
    result = stl.fit()
    seasonal = result.seasonal

    # Calculate the mean amplitude of the seasonal component
    seasonal_df = seasonal.reset_index()
    seasonal_df.columns = ["date", "seasonal"]
    avg_amplitude = calculate_avg_yearly_amplitude(seasonal_df)
    avg_amplitude = np.round(avg_amplitude, 4)

    # Compute the Fourier Transform of the seasonal component
    df_Fourier = FFT_dataframe(seasonal, cutoff_freq=0.5)

    df_Fourier

    # Identify the magnitude and frequency of the primary peak (12-month period)
    target_freq = 1/12
    closest_magnitude, closest_freq = FFT_magnitude_of_closest_freq_to_target_freq(df_Fourier, target_freq)

    # Find the second largest peak that does not correspond to the 12-month peak
    window_size = 0.01
    second_max_magnitude, second_max_freq = find_second_FFT_peak(df_Fourier, target_freq, window_size)
    
    # Compute the ratio of the 12-month peak to the second largest peak

    peak_ratio = closest_magnitude / second_max_magnitude
    peak_ratio = np.round(peak_ratio, 4)
    return peak_ratio, avg_amplitude
