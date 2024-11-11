import rasterio
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
import numpy as np
from rasterio.plot import show
from rasterio.mask import mask
from shapely.geometry import box

# Define the period and file path
period = '1986-2010'
raster_file = f'data/climatezones/Map_KG-Global/KG_{period}.grd'

src = rasterio.open(raster_file)

raster_data = src.read(1)
raster_meta = src.meta

# Define the color palette for climate classification
climate_colors = [
    "#960000", "#FF0000", "#FF6E6E", "#FFCCCC", "#CC8D14", "#CCAA54", "#FFCC00", "#FFFF64",
    "#007800", "#005000", "#003200", "#96FF00", "#00D700", "#00AA00", "#BEBE00", "#8C8C00",
    "#5A5A00", "#550055", "#820082", "#C800C8", "#FF6EFF", "#646464", "#8C8C8C", "#BEBEBE",
    "#E6E6E6", "#6E28B4", "#B464FA", "#C89BFA", "#C8C8FF", "#6496FF", "#64FFFF", "#F5FFFF",
]

# Define the climate classes corresponding to the color palette
climate_classes = [
    'Af', 'Am', 'As', 'Aw', 'BSh', 'BSk', 'BWh', 'BWk', 'Cfa', 'Cfb', 'Cfc', 'Csa', 'Csb',
    'Csc', 'Cwa', 'Cwb', 'Cwc', 'Dfa', 'Dfb', 'Dfc', 'Dfd', 'Dsa', 'Dsb', 'Dsc', 'Dsd',
    'Dwa', 'Dwb', 'Dwc', 'Dwd', 'EF', 'ET', 'Ocean'
]

# Plot the raster data
plt.figure(figsize=(13, 10))
plt.imshow(raster_data, cmap='tab20', vmin=0, vmax=len(climate_colors) - 1)
plt.colorbar(ticks=range(len(climate_colors)), label='Climate Class')
plt.clim(-0.5, len(climate_colors) - 0.5)
plt.title(f'Köppen-Geiger Climate Classification ({period})')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

# Load high-resolution world map for overlay
world = gpd.read_file('data/climatezones/ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp')

# Plot with country borders
fig, ax = plt.subplots(figsize=(13, 10))
show(raster_data, ax=ax, transform=src.transform, cmap='tab20', vmin=0, vmax=len(climate_colors) - 1)
world.boundary.plot(ax=ax, linewidth=0.5)
plt.title(f'Köppen-Geiger Climate Classification with Country Borders ({period})')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

# Determine the climate class for a specific location (e.g., Vienna, Austria)
lon, lat = 16.375, 48.210
row, col = src.index(lon, lat)
climate_code = raster_data[row, col]
climate_class = climate_classes[climate_code - 1]  # Adjust for zero-based index
print(f'The climate class for Vienna, Austria (Lon: {lon}, Lat: {lat}) is: {climate_class}')

# Export a subset of the data to a CSV file
# Define the bounding box for the subset (adjust as needed)
min_lon, min_lat = 16.0, 48.0
max_lon, max_lat = 17.0, 49.0
bbox = box(min_lon, min_lat, max_lon, max_lat)

# Mask the raster with the bounding box
geo = gpd.GeoDataFrame({'geometry': bbox}, index=[0], crs=src.crs)
out_image, out_transform = mask(src, geo.geometry, crop=True)
out_image = out_image[0]  # Extract the first band

# Create a DataFrame with latitude, longitude, and climate class
rows, cols = np.where(out_image != src.nodata)
lats, lons, classes = [], [], []
for row, col in zip(rows, cols):
    lat, lon = src.xy(row, col)
    lats.append(lat)
    lons.append(lon)
    classes.append(climate_classes[out_image[row, col] - 1])  # Adjust for zero-based index

df = pd.DataFrame({'Latitude': lats, 'Longitude': lons, 'Climate_Class': classes})
df.to_csv(f'KG_{period}_subset.csv', index=False)
print(f'Subset data exported to KG_{period}_subset.csv')

src.close()