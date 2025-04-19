import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# Load NetCDF dataset
ds = xr.open_dataset(r'C:\Users\Lenovo\Downloads\monnthly sea surface temp.nc')

# Rename dimensions to lowercase for convenience
ds = ds.rename({'TIME': 'time', 'LAT': 'lat', 'LON': 'lon'})

# Extract the SST variable
sst = ds['SST']

# Handle missing values using the _FillValue attribute if present
if '_FillValue' in sst.attrs:
    sst = sst.where(sst != sst.attrs['_FillValue'])

# Select the data from 1950 to 2023
sst = sst.sel(time=slice('1950-01', '2023-12'))

# Ensure we remove or handle all NaNs
sst = sst.where(~sst.isnull(), drop=False)

# Calculate the standard deviation over time at each grid point
sst_std = sst.std(dim='time', skipna=True)

# Create the plot
plt.figure(figsize=(12, 6))
ax = plt.axes(projection=ccrs.PlateCarree())
std_plot = ax.contourf(
    sst.lon, sst.lat, sst_std,
    levels=60, transform=ccrs.PlateCarree(),
    cmap='viridis'
)
ax.coastlines()
plt.title('Standard Deviation of SST (1950–2023)')
cbar = plt.colorbar(std_plot, orientation='horizontal', label='SST Std Dev (°C)')

# Save the figure
plt.savefig('sst_std_map_1950_2023.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()
