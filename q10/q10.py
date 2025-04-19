import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy import stats
import seaborn as sns

# File paths
u_wind_file = r"C:\Users\Lenovo\Downloads\u wind.nc"
v_wind_file = r"C:\Users\Lenovo\Downloads\v wind.nc"
sst_file = r"C:\Users\Lenovo\Downloads\monnthly sea surface temp.nc"
index_file = r"D:\placement project\asl assg 2\Q7_nino_dmi_indices.csv"

# Load data
print("Loading SST data...")
sst = xr.open_dataset(sst_file)
print("Loading U-wind data...")
u_wind = xr.open_dataset(u_wind_file)
print("Loading V-wind data...")
v_wind = xr.open_dataset(v_wind_file)
print("Loading indices data...")
indices = pd.read_csv(index_file)

# From the dataset inspection, we can see the correct variable names
# Use the correct variable names instead of the first ones
sst_var = sst['SST']
u_var = u_wind['UWND']
v_var = v_wind['VWND']

print(f"SST variable shape: {sst_var.shape}")
print(f"U-wind variable shape: {u_var.shape}")
print(f"V-wind variable shape: {v_var.shape}")

# Convert indices to xarray DataArrays
indices['time'] = pd.to_datetime(indices['Date'])  # Assuming 'time' column exists in the CSV
indices.set_index('time', inplace=True)
nino = xr.DataArray(indices['Nino3.4'], dims="time", coords={"time": indices.index})
dmi = xr.DataArray(indices['DMI'], dims="time", coords={"time": indices.index})

# Align time coordinates
print("Aligning time coordinates...")
# Convert all time coordinates to pandas datetime for consistent comparison
sst_times = pd.to_datetime(sst_var.TIME.values)
u_times = pd.to_datetime(u_var.TIME.values)
v_times = pd.to_datetime(v_var.TIME.values)

# Create DataFrames with year-month information
sst_dates = pd.DataFrame({'date': sst_times})
sst_dates['year_month'] = sst_dates['date'].dt.strftime('%Y-%m')

u_dates = pd.DataFrame({'date': u_times})
u_dates['year_month'] = u_dates['date'].dt.strftime('%Y-%m')

v_dates = pd.DataFrame({'date': v_times})
v_dates['year_month'] = v_dates['date'].dt.strftime('%Y-%m')

index_dates = pd.DataFrame({'date': indices.index})
index_dates['year_month'] = index_dates['date'].dt.strftime('%Y-%m')

# Find common year-months across all datasets
common_year_months = list(set(sst_dates['year_month']) & 
                          set(u_dates['year_month']) & 
                          set(v_dates['year_month']) &
                          set(index_dates['year_month']))

print(f"Found {len(common_year_months)} common months")

# Helper function to filter times by year-month
def filter_by_year_month(times, common_year_months):
    dates_df = pd.DataFrame({'date': times})
    dates_df['year_month'] = dates_df['date'].dt.strftime('%Y-%m')
    return dates_df[dates_df['year_month'].isin(common_year_months)]['date'].values

# Get dates for each dataset that fall within common year-months
sst_common_times = filter_by_year_month(sst_times, common_year_months)
u_common_times = filter_by_year_month(u_times, common_year_months)
v_common_times = filter_by_year_month(v_times, common_year_months)
index_common_times = filter_by_year_month(indices.index, common_year_months)

# Subset data to common time period
sst_common = sst_var.sel(TIME=sst_common_times)
u_common = u_var.sel(TIME=u_common_times)
v_common = v_var.sel(TIME=v_common_times)

# Subset indices
nino_common = nino.sel(time=index_common_times)
dmi_common = dmi.sel(time=index_common_times)

# Function to perform regression analysis
def regress_field_on_index(field, index):
    """Regress a spatial field on an index time series."""
    # Get time dimension name (assumes it's the first dimension)
    time_dim = field.dims[0]
    spatial_dims = field.dims[1:]
    
    # Extract values and reshape for regression
    field_values = field.values
    index_values = index.values
    
    # Initialize output arrays
    reg_shape = field.isel({time_dim: 0}).shape
    reg_coef = np.full(reg_shape, np.nan)
    p_values = np.full(reg_shape, np.nan)
    
    # Flatten spatial dimensions for easier iteration
    orig_shape = field_values.shape
    field_reshaped = field_values.reshape(orig_shape[0], -1)
    
    # Perform regression for each spatial point
    for i in range(field_reshaped.shape[1]):
        y = field_reshaped[:, i]
        
        # Skip if all NaN
        if np.all(np.isnan(y)):
            continue
        
        # Get valid data points
        valid = ~np.isnan(y)
        if np.sum(valid) > 10:  # Require at least 10 valid points
            x_valid = index_values[valid]
            y_valid = y[valid]
            
            # Perform regression
            try:
                slope, intercept, r_value, p_value, std_err = stats.linregress(x_valid, y_valid)
                
                # Store in flattened array
                flat_idx = i
                reg_coef.flat[flat_idx] = slope
                p_values.flat[flat_idx] = p_value
            except:
                pass
    
    # Create DataArrays with proper coordinates
    coords = {dim: field[dim].values for dim in spatial_dims}
    
    reg_da = xr.DataArray(
        data=reg_coef,
        dims=spatial_dims,
        coords=coords,
        name="regression_coefficient"
    )
    
    p_da = xr.DataArray(
        data=p_values,
        dims=spatial_dims,
        coords=coords,
        name="p_value"
    )
    
    return reg_da, p_da

# If wind data has a level dimension, select 1000 hPa
if 'LEV17_17' in u_common.dims:
    print("Selecting 1000 hPa pressure level for winds")
    u_common = u_common.sel(LEV17_17=1000)
    v_common = v_common.sel(LEV17_17=1000)

print(f"SST common shape: {sst_common.shape}")
print(f"U-wind common shape: {u_common.shape}")
print(f"V-wind common shape: {v_common.shape}")
print(f"Nino3.4 common shape: {nino_common.shape}")
print(f"DMI common shape: {dmi_common.shape}")

# Perform regression for Nino3.4
print("Performing regression for Nino3.4...")
sst_nino_reg, sst_nino_p = regress_field_on_index(sst_common, nino_common)
u_nino_reg, u_nino_p = regress_field_on_index(u_common, nino_common)
v_nino_reg, v_nino_p = regress_field_on_index(v_common, nino_common)

# Perform regression for DMI
print("Performing regression for DMI...")
sst_dmi_reg, sst_dmi_p = regress_field_on_index(sst_common, dmi_common)
u_dmi_reg, u_dmi_p = regress_field_on_index(u_common, dmi_common)
v_dmi_reg, v_dmi_p = regress_field_on_index(v_common, dmi_common)

print(f"SST Nino regression shape: {sst_nino_reg.shape}")
print(f"U-wind Nino regression shape: {u_nino_reg.shape}")
print(f"V-wind Nino regression shape: {v_nino_reg.shape}")

# Function to plot regression maps
def plot_regression_map(sst_reg, u_reg, v_reg, p_values=None, title="", output_file=""):
    """Plot regression maps with wind vectors."""
    fig = plt.figure(figsize=(15, 8))
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
    
    # Define colormap levels with appropriate range
    # Calculate max absolute value for symmetric color scale
    vmax = max(1.0, float(np.nanpercentile(np.abs(sst_reg.values), 95)))
    levels = np.linspace(-vmax, vmax, 21)
    
    # Plot SST regression coefficients
    sst_plot = ax.contourf(sst_reg.LON, sst_reg.LAT, sst_reg, 
                           levels=levels, cmap='RdBu_r', extend='both',
                           transform=ccrs.PlateCarree())
    
    # Add wind vectors (subsample for clarity)
    skip = 4  # Adjust based on your grid resolution
    
    # Create meshgrid for quiver plot
    lons = u_reg.LONN71_73.values[::skip]
    lats = u_reg.LAT.values[::skip]
    lons_grid, lats_grid = np.meshgrid(lons, lats)
    
    # Subsample wind fields
    u_plot = u_reg.values[::skip, ::skip]
    v_plot = v_reg.values[::skip, ::skip]
    
    # Scale wind vectors for better visualization
    scale = 20  # Adjust as needed
    
    # Add wind vectors
    quiv = ax.quiver(lons_grid, lats_grid, u_plot, v_plot,
                     scale=scale, width=0.002, transform=ccrs.PlateCarree())
    
    # Add quiver key for scale reference
    key_value = 2  # Adjust based on your data
    plt.quiverkey(quiv, 0.92, 0.95, key_value, f'{key_value} m/s', 
                  labelpos='E', coordinates='figure')
    
    # Add map features
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False
    
    # Add colorbar
    cbar = plt.colorbar(sst_plot, orientation='horizontal', pad=0.05, aspect=40)
    cbar.set_label('SST Regression Coefficient (°C per index unit)')
    
    # Set title
    plt.title(title, fontsize=16)
    
    # Save figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

# Try plotting with error handling
try:
    print("Plotting Nino3.4 regression map...")
    plot_regression_map(
        sst_nino_reg, u_nino_reg, v_nino_reg, sst_nino_p,
        title='Regression of Nino3.4 Index on SST and Winds (1950-2023)',
        output_file='nino34_regression_sst_winds.png'
    )
    
    print("Plotting DMI regression map...")
    plot_regression_map(
        sst_dmi_reg, u_dmi_reg, v_dmi_reg, sst_dmi_p,
        title='Regression of DMI Index on SST and Winds (1950-2023)',
        output_file='dmi_regression_sst_winds.png'
    )
    
    print("Analysis complete!")
except Exception as e:
    print(f"Error in plotting: {e}")
    
    # Simplified fallback plot
    try:
        plt.figure(figsize=(12, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())
        
        # Just plot SST regression for Nino3.4
        levels = np.linspace(-1, 1, 21)
        im = ax.contourf(sst_nino_reg.LON, sst_nino_reg.LAT, sst_nino_reg, 
                         levels=levels, cmap='RdBu_r', extend='both',
                         transform=ccrs.PlateCarree())
        
        ax.coastlines()
        ax.gridlines(draw_labels=True)
        plt.colorbar(im, orientation='horizontal', label='SST Regression (°C per index unit)')
        plt.title('Regression of Nino3.4 Index on SST (Fallback Plot)')
        plt.savefig('nino34_regression_fallback.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Created fallback SST-only plot.")
    except Exception as e2:
        print(f"Even the fallback plot failed: {e2}")