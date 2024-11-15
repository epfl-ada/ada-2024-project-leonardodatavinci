import rasterio
import pandas as pd
import numpy as np
import plotly.express as px

# Define climate colors and classes
climate_colors = [
    "#960000", "#FF0000", "#FF6E6E", "#FFCCCC", "#CC8D14", "#CCAA54", "#FFCC00", "#FFFF64",
    "#007800", "#005000", "#003200", "#96FF00", "#00D700", "#00AA00", "#BEBE00", "#8C8C00",
    "#5A5A00", "#550055", "#820082", "#C800C8", "#FF6EFF", "#646464", "#8C8C8C", "#BEBEBE",
    "#E6E6E6", "#6E28B4", "#B464FA", "#C89BFA", "#C8C8FF", "#6496FF", "#64FFFF", "#F5FFFF",
]
climate_classes = [
    'Af', 'Am', 'As', 'Aw', 'BSh', 'BSk', 'BWh', 'BWk', 'Cfa', 'Cfb', 'Cfc', 'Csa', 'Csb',
    'Csc', 'Cwa', 'Cwb', 'Cwc', 'Dfa', 'Dfb', 'Dfc', 'Dfd', 'Dsa', 'Dsb', 'Dsc', 'Dsd',
    'Dwa', 'Dwb', 'Dwc', 'Dwd', 'EF', 'ET', 'Ocean'
]

# Define period and file path
period = '1986-2010'
raster_file = f'data/climatezones/Map_KG-Global/KG_{period}.grd'


def generate_climate_map(raster_file, downsample_factor=10):
    # Ensure the downsample factor is at least 1
    downsample_factor = max(1, downsample_factor)
    print(f"Using downsample factor: {downsample_factor}")

    # Open and read the raster file
    with rasterio.open(raster_file) as src:
        raster_data = src.read(1)[::downsample_factor, ::downsample_factor]  # Downsample by the factor

    # Extract non-empty data points
    rows, cols = np.where(raster_data != src.nodata)
    print(f"Non-empty points identified: {len(rows)}")

    lats, lons, classes = [], [], []

    for i, (row, col) in enumerate(zip(rows, cols)):
        if i % 10000 == 0:  # Adjusting print frequency for downsampled data
            print(f"Processing point {i} of {len(rows)}...")
        # Calculate latitude and longitude for the downsampled row and column
        lon, lat = src.xy(row * downsample_factor, col * downsample_factor)  # Scale row and col back to original coordinates
        lats.append(lat)
        lons.append(lon)
        class_index = raster_data[row, col] - 1  # Adjust for zero-based index
        if 0 <= class_index < len(climate_classes):
            classes.append(climate_classes[class_index])
        else:
            classes.append('Unknown')

    # Create DataFrame
    df = pd.DataFrame({'Latitude': lats, 'Longitude': lons, 'Climate_Class': classes})

    # Create plotly figure
    fig = px.scatter_geo(
        df,
        lat='Latitude',
        lon='Longitude',
        color='Climate_Class',
        title=f'Köppen-Geiger Climate Classification ({period})',
        color_discrete_sequence=climate_colors,
        category_orders={'Climate_Class': climate_classes},
        opacity=0.5,  # Adjust opacity to make borders more visible
    )

    # Update map layout
    fig.update_geos(
        showcountries=True,
        showcoastlines=True,
        projection_type='natural earth'
    )

    fig.write_html("illustrations/Nicolas/climate_map.html")


def get_climate_class(lon, lat, raster_file):
    with rasterio.open(raster_file) as src:
        # Get the row and column for the given coordinates
        row, col = src.index(lon, lat)
        climate_code = src.read(1)[row, col]
        
        if 1 <= climate_code <= len(climate_classes):
            return climate_classes[climate_code - 1]
        else:
            return "Unknown"


# Function to load the station data from a fixed-width text file
def load_station_data(file_path):
    # Define the column specifications and names
    colspecs = [
        (0, 7),    # Wmo#
        (8, 9),    # R
        (10, 15),  # Lat
        (16, 22),  # Lon
        (23, 47),  # Country Name
        (48, 73),  # Station Name
        (74, 78),  # Statn Elev (m)
        (79, 85),  # Barom Elev (.1m)
        (86, 96),  # Local Statn #
        (97, 99),  # Separator
        (100, 105),  # Mean Stn Pres
        (106, 110), # Mean Slvl Pres
        (111, 115), # Mean Temp
        (116, 120), # Dly Totl Prec
        (121, 125), # Mean Max Temp
        (126, 130), # Mean Min Temp
        (131, 135), # Mean RH
        (136, 141)  # Total # Obs
    ]

    column_names = [
        'Wmo#', 'R', 'Lat', 'Lon', 'Country Name', 'Station Name',
        'Statn Elev (m)', 'Barom Elev (.1m)', 'Local Statn #', 'Separator',
        'Mean Stn Pres', 'Mean Slvl Pres', 'Mean Dly Temp', 'Totl Prec',
        'Mean Max Temp', 'Mean Min Temp', 'Mean RH', 'Total # Obs'
    ]

    # Load the data using read_fwf
    df = pd.read_fwf(
        file_path,
        colspecs=colspecs,
        skiprows=4,  # Skip header rows
        names=column_names,
        encoding='ISO-8859-1'  # Specify encoding to handle special characters
    )
    return df

# Function to convert latitude and longitude from DMS format to decimal degrees
def convert_to_decimal(coord):
    degrees = int(coord[:-1][:-2])  # Extract degrees
    minutes = int(coord[:-1][-2:])  # Extract minutes
    decimal = degrees + minutes / 60
    if coord[-1] in ['S', 'W']:
        decimal = -decimal
    return decimal

# Function to generate a map with all weather stations from the DataFrame
def generate_station_map(df):
    # Convert latitude and longitude columns to decimal degrees
    df['Latitude'] = df['Lat'].apply(convert_to_decimal)
    df['Longitude'] = df['Lon'].apply(convert_to_decimal)

    # Create the Plotly map
    fig = px.scatter_geo(
        df,
        lat='Latitude',
        lon='Longitude',
        hover_name='Station Name',
        hover_data={'Country Name': True, 'Statn Elev (m)': True, 'Mean Dly Temp': True, 'Totl Prec': True},
        title="Global Weather Stations",
    )
    
    fig.update_geos(
        showcountries=True,
        showcoastlines=True,
        projection_type="natural earth"
    )

    # Save map as HTML
    fig.write_html("illustrations/Nicolas/weather_stations.html")


