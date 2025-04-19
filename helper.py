import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
from eofs.xarray import Eof
import pandas as pd
from scipy import signal
import seaborn as sns
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')

# Helper functions for plotting maps
def create_map(projection=ccrs.PlateCarree(), figsize=(12, 8)):
    """Create a map with Cartopy"""
    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=projection)
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    return fig, ax

def plot_spatial_map(data, title, cmap='viridis', vmin=None, vmax=None, projection=ccrs.PlateCarree(), 
                     figsize=(12, 8), add_colorbar=True, cbar_label=None):
    """Plot spatial map using Cartopy"""
    # Add cyclic point to prevent gap at the dateline
    if 'lon' in data.dims:
        data_cyclic, lon_cyclic = add_cyclic_point(data.values, coord=data.lon)
        lons = lon_cyclic
        lats = data.lat
    else:
        data_cyclic = data.values
        lons = data.longitude if 'longitude' in data.dims else data.lon
        lats = data.latitude if 'latitude' in data.dims else data.lat
    
    fig, ax = create_map(projection=projection, figsize=figsize)
    
    # Plot the data
    im = ax.contourf(lons, lats, data_cyclic, transform=ccrs.PlateCarree(),
                     cmap=cmap, vmin=vmin, vmax=vmax, extend='both')
    
    # Add colorbar
    if add_colorbar:
        cbar = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
        if cbar_label:
            cbar.set_label(cbar_label)
    
    plt.title(title, fontsize=14)
    return fig, ax, im

def plot_spatial_map_with_vectors(data, u_wind, v_wind, title, cmap='viridis', vmin=None, vmax=None, 
                                  projection=ccrs.PlateCarree(), figsize=(12, 8)):
    """Plot spatial map with wind vectors overlaid"""
    fig, ax, im = plot_spatial_map(data, title, cmap, vmin, vmax, projection, figsize)
    
    # Add wind vectors with appropriate scaling and subsampling
    step = 4  # Adjust based on your data resolution
    ax.quiver(u_wind.lon[::step], u_wind.lat[::step], 
              u_wind.values[::step, ::step], v_wind.values[::step, ::step], 
              transform=ccrs.PlateCarree(), scale=50, color='black', alpha=0.6)
    
    return fig, ax

# Function for regression analysis
def regress_timeseries_on_spatial(time_series, spatial_data):
    """
    Regress a time series onto spatial data.
    
    Parameters:
    -----------
    time_series : xarray.DataArray or numpy.ndarray
        The time series to regress
    spatial_data : xarray.DataArray
        The spatial data to regress onto, with time as one dimension
    
    Returns:
    --------
    xarray.DataArray
        Regression coefficients for each spatial point
    """
    # Ensure time series is a numpy array
    if isinstance(time_series, xr.DataArray):
        time_series = time_series.values
    
    # Standardize the time series (subtract mean and divide by std)
    time_series_std = (time_series - np.mean(time_series)) / np.std(time_series)
    
    # For each spatial point, calculate regression coefficient
    result = spatial_data.copy()
    
    # Get dimensions without time
    non_time_dims = [dim for dim in spatial_data.dims if dim != 'time']
    
    # Prepare arrays for computation
    X = time_series_std
    coef = np.zeros(spatial_data.isel(time=0).shape)
    
    # Create a meshgrid of spatial indices
    indices = np.array(np.meshgrid(*[range(spatial_data[dim].size) for dim in non_time_dims])).T.reshape(-1, len(non_time_dims))
    
    # Perform regression for each point
    for idx in indices:
        # Get indices as tuples for each dimension
        idx_dict = {dim: idx[i] for i, dim in enumerate(non_time_dims)}
        
        # Get the time series at this spatial point
        Y = spatial_data.isel(time=slice(None), **idx_dict).values
        
        # Calculate regression coefficient: cov(X,Y)/var(X)
        # Since X is standardized, var(X) = 1, so coef = cov(X,Y)
        coef_at_point = np.sum((Y - np.mean(Y)) * X) / len(X)
        
        # Store in the result array
        coef_idx = tuple(idx)
        coef[coef_idx] = coef_at_point
    
    # Create a new DataArray with the coefficients
    coords = {dim: spatial_data[dim] for dim in non_time_dims}
    result = xr.DataArray(coef, coords=coords, dims=non_time_dims)
    
    return result

# Function to apply a low-pass filter to a time series
def apply_lowpass_filter(data, cutoff_years=8, fs=12):
    """
    Apply a Butterworth low-pass filter to a time series.
    
    Parameters:
    -----------
    data : numpy.ndarray
        The time series to filter
    cutoff_years : float
        The cutoff period in years
    fs : float
        Sampling frequency in samples per year (12 for monthly data)
    
    Returns:
    --------
    numpy.ndarray
        Filtered time series
    """
    # Convert cutoff from years to frequency
    cutoff_freq = 1 / cutoff_years
    nyquist = fs / 2
    cutoff_norm = cutoff_freq / nyquist
    
    # Design the Butterworth filter
    b, a = signal.butter(4, cutoff_norm, btype='low')
    
    # Apply the filter
    filtered = signal.filtfilt(b, a, data)
    
    return filtered

# Function to apply a high-pass filter to a time series
def apply_highpass_filter(data, cutoff_years=8, fs=12):
    """
    Apply a Butterworth high-pass filter to a time series.
    
    Parameters:
    -----------
    data : numpy.ndarray
        The time series to filter
    cutoff_years : float
        The cutoff period in years
    fs : float
        Sampling frequency in samples per year (12 for monthly data)
    
    Returns:
    --------
    numpy.ndarray
        Filtered time series
    """
    # Convert cutoff from years to frequency
    cutoff_freq = 1 / cutoff_years
    nyquist = fs / 2
    cutoff_norm = cutoff_freq / nyquist
    
    # Design the Butterworth filter
    b, a = signal.butter(4, cutoff_norm, btype='high')
    
    # Apply the filter
    filtered = signal.filtfilt(b, a, data)
    
    return filtered

# Function to calculate correlation and print results
def calculate_correlation(ts1, ts2, name1, name2):
    """Calculate correlation between two time series and print results"""
    corr, p_value = pearsonr(ts1, ts2)
    print(f"Correlation between {name1} and {name2}: {corr:.4f} (p-value: {p_value:.4f})")
    return corr, p_value



# Questions 3-4: Follow methodology from the paper to reproduce Figure 1 for 1950-2023
# and explain the methodology in detail

# Load SST data
# Load SST data (fix file path formatting)
sst_data = xr.open_dataset(r"C:\Users\Lenovo\Downloads\monnthly sea surface temp.nc")
print("Dimensions:", sst_data.dims)  # Should show TIME, LAT, LON

# Fix 1: Use correct dimension names
sst = sst_data['SST'].rename({'TIME': 'time', 'LAT': 'lat', 'LON': 'lon'}) # Should show dimensions like TIME, LAT, LON

# Adjust dimension names in selection
# sst = sst_data['SST']  # Verify variable name matches your dataset


sst = sst.sel(time=slice('1950-01-01', '2023-12-31'))

# Define the Indian Ocean domain
io_sst = sst.sel(lon=slice(40, 120), lat=slice(-30, 30))

# Calculate anomalies
clim_monthly = io_sst.groupby('time.month').mean('time')
sst_anom = io_sst.groupby('time.month') - clim_monthly

# Area weights
# Calculate area weights (cosine of latitude in radians)
coslat = np.cos(np.deg2rad(sst_anom['lat']))  # Access 'lat' as coordinate
wgts = np.sqrt(coslat.values)[..., np.newaxis]  # Convert to numpy array


# Create an EOF solver
solver = Eof(sst_anom, weights=wgts)

# Get PC time series
pcs = solver.pcs(npcs=2, pcscaling=1)

# Calculate correlation between PC1 and PC2
pc1 = pcs[:, 0]
pc2 = pcs[:, 1]
corr, p_value = pearsonr(pc1, pc2)

print(f"Correlation between EOF1 and EOF2: {corr:.4f}")
print(f"P-value: {p_value:.4f}")

# Plot the correlation
plt.figure(figsize=(10, 6))
plt.scatter(pc1, pc2, alpha=0.5)
plt.xlabel('PC1', fontsize=12)
plt.ylabel('PC2', fontsize=12)
plt.title(f'Correlation between PC1 and PC2: {corr:.4f}', fontsize=14)
plt.grid(True)
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)

# Add regression line
z = np.polyfit(pc1, pc2, 1)
p = np.poly1d(z)
plt.plot(np.sort(pc1), p(np.sort(pc1)), "r--", linewidth=1.5)

plt.savefig('correlation_pc1_pc2.png', dpi=300, bbox_inches='tight')
plt.close()

# Explanation
print("""
Explanation of the correlation between EOF1 and EOF2:

The correlation coefficient between EOF1 and EOF2 is {:.4f}, which indicates a {}.

EOF1 typically represents the Indian Ocean Basin Mode (IOBM), characterized by a basin-wide warming or cooling pattern. This mode is often associated with ENSO influences where El Niño events lead to basin-wide warming in the Indian Ocean with some delay.

EOF2 typically represents the Indian Ocean Dipole (IOD) mode, characterized by anomalously cool SST in the eastern equatorial Indian Ocean and warm SST in the western equatorial Indian Ocean during its positive phase.

The correlation between these two modes suggests:
1. They are not completely independent phenomena despite being mathematically orthogonal in the EOF analysis.
2. There may be physical processes linking these two modes, possibly through their respective relationships with ENSO.
3. The sign of correlation helps understand whether these modes tend to occur in phase or out of phase with each other.
4. The magnitude of correlation indicates how strongly these modes are related.

This relationship is important for understanding the dynamics of the Indian Ocean and its teleconnections with global climate patterns.
""".format(corr, "relatively strong" if abs(corr) > 0.5 else "moderate" if abs(corr) > 0.3 else "weak"))