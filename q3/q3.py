import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from eofs.xarray import Eof

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd

def load_and_preprocess_sst():
    """
    Load and preprocess SST data for EOF analysis
    """
    # Load SST data
    sst_data = xr.open_dataset(r'C:\Users\Lenovo\Downloads\monnthly sea surface temp.nc')
    sst = sst_data['SST'].rename({'TIME': 'time', 'LAT': 'lat', 'LON': 'lon'})
    
    # Select Indian Ocean domain (as per the paper)
    # Adjust coordinates based on your data
    io_sst = sst.sel(lon=slice(40, 120), lat=slice(-30, 30))
    
    # Calculate monthly climatology and remove seasonal cycle
    climatology = io_sst.groupby('time.month').mean('time')
    anomalies = io_sst.groupby('time.month') - climatology
    
    # Create weights based on cosine of latitude
    weights = np.sqrt(np.cos(np.deg2rad(io_sst.lat)))
    weights_array = weights.values[:, np.newaxis] * np.ones(len(io_sst.lon))
    
    return anomalies, weights_array

def perform_eof_analysis(data, weights):
    """
    Perform EOF analysis and return EOFs and PC time series
    """
    # Apply weights to data
    weighted_data = data * weights
    
    # Create solver
    solver = Eof(weighted_data)

# Set number of modes
    n_modes = 2

    # Extract EOFs, PCs, and variance
    eofs = solver.eofs(neofs=n_modes)
    pcs = solver.pcs(npcs=n_modes)
    var_frac = solver.varianceFraction(neigs=n_modes)
    
    return eofs, pcs, var_frac

def plot_eofs_and_pcs(eofs, pcs, var_frac):
    """
    Plot EOFs spatial patterns and PC time series
    """
    # Create figure with 2 subplots for EOFs
    fig = plt.figure(figsize=(15, 12))
    
    # Plot EOF1 spatial pattern
    ax1 = plt.subplot(2, 1, 1, projection=ccrs.PlateCarree())
    eofs[0].plot(ax=ax1, transform=ccrs.PlateCarree(), cmap='RdBu_r')
    ax1.add_feature(cfeature.COASTLINE)
    ax1.set_title(f'EOF1 ({var_frac[0]*100:.1f}% variance explained)')
    
    # Plot EOF2 spatial pattern
    ax2 = plt.subplot(2, 1, 2, projection=ccrs.PlateCarree())
    eofs[1].plot(ax=ax2, transform=ccrs.PlateCarree(), cmap='RdBu_r')
    ax2.add_feature(cfeature.COASTLINE)
    ax2.set_title(f'EOF2 ({var_frac[1]*100:.1f}% variance explained)')
    
    plt.tight_layout()
    plt.show()
    
    # Plot PC time series
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))
    
    # PC1 time series
    ax1.plot(pcs.time, pcs[:, 0], 'b-', label='PC1')
    ax1.set_title('PC1 Time Series')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True)
    
    # PC2 time series
    ax2.plot(pcs.time, pcs[:, 1], 'r-', label='PC2')
    ax2.set_title('PC2 Time Series')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Amplitude')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def save_pc_timeseries(pcs):
    """
    Save PC time series to CSV file
    """
    # Create DataFrame
    df = pd.DataFrame({
        'PC1': pcs[:, 0],
        'PC2': pcs[:, 1]
    }, index=pcs.time.values)
    
    # Save to CSV
    df.to_csv('eof_timeseries.csv')
    
    return df

def main():
    # Load and preprocess data
    print("Loading and preprocessing data...")
    sst_anomalies, weights = load_and_preprocess_sst()
    
    # Perform EOF analysis
    print("Performing EOF analysis...")
    eofs, pcs, var_frac = perform_eof_analysis(sst_anomalies, weights)
    
    # Plot results
    print("Plotting results...")
    plot_eofs_and_pcs(eofs, pcs, var_frac)
    
    # Save time series
    print("Saving PC time series...")
    pc_df = save_pc_timeseries(pcs)
    
    # Print variance explained
    print("\nVariance explained by each mode:")
    print(f"EOF1: {var_frac[0]*100:.1f}%")
    print(f"EOF2: {var_frac[1]*100:.1f}%")
    
    return eofs, pcs, var_frac, pc_df

if __name__ == "__main__":
    eofs, pcs, var_frac, pc_df = main()
def save_eof_results(eofs, pcs, var_frac, eof_filename='eofs.nc', pc_filename='pcs.csv', var_frac_filename='variance_fraction.csv'):
    """
    Save EOF spatial patterns, PC time series, and variance fractions to disk.
    
    Parameters:
    -----------
    eofs : xarray.DataArray
        EOF spatial patterns (usually 3D: mode x lat x lon)
    pcs : xarray.DataArray
        Principal component time series (usually 2D: time x mode)
    var_frac : numpy.ndarray or list
        Variance fraction explained by each EOF mode
    eof_filename : str
        Filename to save EOF spatial patterns (NetCDF)
    pc_filename : str
        Filename to save PC time series (CSV)
    var_frac_filename : str
        Filename to save variance fractions (CSV)
    """
    # Save EOFs as NetCDF
    eofs.to_netcdf(eof_filename)
    print(f"EOF spatial patterns saved to {eof_filename}")
    
    # Convert PCs to DataFrame and save as CSV
    pc_df = pcs.to_dataframe(name='PC').unstack(level=1)  # time x mode DataFrame
    # After unstacking, columns may be tuples (e.g., (0,), (1,))
    pc_df.columns = [f'PC{int(col[0])+1}' if isinstance(col, tuple) and isinstance(col[0], int)
                 else f'PC{int(col)+1}' if isinstance(col, int)
                 else f'PC{col}' for col in pc_df.columns]


    pc_df.to_csv(pc_filename)
    print(f"PC time series saved to {pc_filename}")
    
    # Save variance fractions as CSV
    import pandas as pd
    var_df = pd.DataFrame({
        'EOF Mode': [f'EOF{i+1}' for i in range(len(var_frac))],
        'Variance Fraction': var_frac
    })
    var_df.to_csv(var_frac_filename, index=False)
    print(f"Variance fractions saved to {var_frac_filename}")

# Example usage after your EOF analysis:
save_eof_results(eofs, pcs, var_frac)
