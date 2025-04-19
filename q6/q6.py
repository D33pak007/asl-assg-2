import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from eofs.xarray import Eof
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy import stats

def rename_dims(ds, lon_name, lat_name):
    rename_dict = {}
    for dim in ds.dims:
        if dim == 'TIME':
            rename_dict[dim] = 'time'
        elif dim == lon_name:
            rename_dict[dim] = 'lon'
        elif dim == lat_name:
            rename_dict[dim] = 'lat'
    return ds.rename(rename_dict)

def load_all_data():
    print("Loading datasets...")
    sst = xr.open_dataset(r"C:\Users\Lenovo\Downloads\monnthly sea surface temp.nc")
    u_wind = xr.open_dataset(r"C:\Users\Lenovo\Downloads\u wind.nc")
    v_wind = xr.open_dataset(r"C:\Users\Lenovo\Downloads\v wind.nc")
    
    sst = rename_dims(sst, 'LON', 'LAT')
    u_wind = rename_dims(u_wind, 'LONN71_73', 'LAT')
    v_wind = rename_dims(v_wind, 'LONN71_73', 'LAT')
    
    return sst, u_wind, v_wind

def preprocess_data_for_eof(sst_data):
    print("Preprocessing global SST data for EOF analysis...")
    
    global_sst = sst_data
    
    # Calculate and remove climatology
    climatology = global_sst.groupby('time.month').mean('time')
    anomalies = global_sst.groupby('time.month') - climatology
    
    # Create latitude weights for EOF analysis
    weights = np.sqrt(np.cos(np.deg2rad(global_sst.lat)))
    weights_da = xr.DataArray(weights, coords=[global_sst.lat], dims=['lat'])
    
    return anomalies, weights_da

def perform_eof_analysis(data, weights):
    """
    Perform EOF analysis
    """
    print("Performing EOF analysis...")
    
    # Select SST variable if input is a Dataset
    if isinstance(data, xr.Dataset):
        # Replace 'SST' with your actual SST variable name
        data = data['SST']
    
    # Apply weights (broadcast over lon)
    weighted_data = data * weights
    
    solver = Eof(weighted_data)
    
    n_modes = 2
    eofs = solver.eofs(neofs=n_modes)
    pcs = solver.pcs(npcs=n_modes)
    var_frac = solver.varianceFraction(neigs=n_modes)
    
    return eofs, pcs, var_frac

def plot_eof_patterns(eofs, var_frac):
    """
    Plot EOF spatial patterns
    """
    print("Plotting EOF patterns...")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10),
                                  subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Plot EOF1
    cmap = plt.cm.RdBu_r
    levels = np.linspace(-0.05, 0.05, 21)  # Adjust based on your data range
    
    c1 = eofs[0].plot.contourf(ax=ax1, transform=ccrs.PlateCarree(), 
                              cmap=cmap, levels=levels, add_colorbar=False)
    fig.colorbar(c1, ax=ax1, orientation='horizontal', pad=0.05, shrink=0.8)
    ax1.add_feature(cfeature.COASTLINE)
    ax1.set_title(f'EOF1 ({var_frac[0]*100:.1f}% variance)')
    ax1.set_global()
    ax1.gridlines(draw_labels=True)
    
    # Plot EOF2
    c2 = eofs[1].plot.contourf(ax=ax2, transform=ccrs.PlateCarree(), 
                              cmap=cmap, levels=levels, add_colorbar=False)
    fig.colorbar(c2, ax=ax2, orientation='horizontal', pad=0.05, shrink=0.8)
    ax2.add_feature(cfeature.COASTLINE)
    ax2.set_title(f'EOF2 ({var_frac[1]*100:.1f}% variance)')
    ax2.set_global()
    ax2.gridlines(draw_labels=True)
    
    plt.tight_layout()
    plt.savefig('eof_patterns.png', dpi=300, bbox_inches='tight')
    plt.show()

def calculate_anomalies(data, variable_name):
    """
    Calculate anomalies by removing monthly climatology
    """
    if isinstance(data, xr.Dataset):
        data = data[variable_name]
    
    # Calculate and remove monthly climatology
    climatology = data.groupby('time.month').mean('time')
    anomalies = data.groupby('time.month') - climatology
    
    return anomalies

def perform_regression_manual(pc, field_array, field_dims=None):
    """
    Perform regression analysis manually with scipy.stats.linregress
    
    Parameters:
    -----------
    pc : numpy.ndarray
        Principal component time series (1D array)
    field_array : numpy.ndarray
        Field to regress PC onto (3D array: time x lat x lon)
    field_dims : dict, optional
        Dictionary with dimension coordinates
        
    Returns:
    --------
    reg_coef : numpy.ndarray or xarray.DataArray
        Regression coefficients (2D array: lat x lon)
    """
    # Initialize output arrays
    reg_coef = np.full((field_array.shape[1], field_array.shape[2]), np.nan)
    
    # Loop over spatial grid points
    for i in range(field_array.shape[1]):
        for j in range(field_array.shape[2]):
            # Extract time series at this grid point
            y = field_array[:, i, j]
            
            # Skip if any NaN values
            if np.any(np.isnan(y)):
                continue
                
            # Perform linear regression
            try:
                # Make sure we have the same number of time points for both arrays
                min_length = min(len(pc), len(y))
                slope, intercept, r_value, p_value, std_err = stats.linregress(pc[:min_length], y[:min_length])
                reg_coef[i, j] = slope
            except:
                # Handle any errors (e.g., constant values)
                continue
    
    # Return as DataArray if dimensions are provided
    if field_dims is not None:
        reg_coef_da = xr.DataArray(
            reg_coef,
            dims=['lat', 'lon'],
            coords={'lat': field_dims['lat'], 'lon': field_dims['lon']}
        )
        return reg_coef_da
    
    return reg_coef

def plot_sst_wind_regression(pc_number, sst_reg, u_reg, v_reg, 
                            sst_lat, sst_lon, wind_lat, wind_lon):
    """
    Plot SST regression with wind vectors
    """
    print(f"Plotting SST and wind regression for PC{pc_number}...")
    
    fig, ax = plt.subplots(figsize=(14, 10), subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})
    
    # Set contour levels for SST
    max_val = max(abs(np.nanmin(sst_reg)), abs(np.nanmax(sst_reg)))
    levels = np.linspace(-max_val, max_val, 21)
    
    # Create meshgrid for SST
    lon_mesh_sst, lat_mesh_sst = np.meshgrid(sst_lon, sst_lat)
    
    # Plot SST regression as filled contours
    sst_plot = ax.contourf(lon_mesh_sst, lat_mesh_sst, sst_reg, 
                         levels=levels, cmap='RdBu_r', transform=ccrs.PlateCarree(),
                         extend='both')
    
    # Add colorbar for SST
    cbar = plt.colorbar(sst_plot, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
    cbar.set_label('SST Change (°C/std dev)')
    
    # Prepare wind data for quiver plot
    # Subsample for better visualization
    step = 4  # Increase this value to show fewer arrows
    
    # Create meshgrid for wind
    lon_mesh, lat_mesh = np.meshgrid(wind_lon[::step], wind_lat[::step])
    
    # Subset wind regression fields
    u_plot = u_reg[::step, ::step]
    v_plot = v_reg[::step, ::step]
    
    # Calculate vector magnitudes for scaling
    wind_mag = np.sqrt(u_plot**2 + v_plot**2)
    scale_factor = 50  # Adjust this to make arrows more visible
    
    # Plot wind vectors
    quiv = ax.quiver(lon_mesh, lat_mesh, u_plot, v_plot, 
                   scale=scale_factor, width=0.003, 
                   transform=ccrs.PlateCarree(), zorder=5)
    
    # Add a key for wind vectors
    ref_value = 1.0  # Set a round reference value in m/s
    ax.quiverkey(quiv, 0.9, 0.95, ref_value, f'{ref_value} m/s', 
               labelpos='E', coordinates='figure', fontproperties={'size': 10})
    
    # Add map features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=':')
    
    # Add gridlines
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                    linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    
    # Set plot extent - global view
    ax.set_global()
    
    # Add title
    if pc_number == 1:
        pattern_name = "ENSO"
    elif pc_number == 2:
        pattern_name = "PDO-like Pattern"
    else:
        pattern_name = f"PC{pc_number} Pattern"
        
    ax.set_title(f'PC{pc_number} ({pattern_name}): SST (colors) and Wind (arrows) Regression Patterns', 
                fontsize=14)
    
    plt.tight_layout()
    plt.savefig(f'pc{pc_number}_sst_wind_regression.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    # Load required datasets
    sst, u_wind, v_wind = load_all_data()
    
    # Preprocess SST data for EOF analysis
    sst_anomalies, weights = preprocess_data_for_eof(sst)
    
    # Perform EOF analysis
    eofs, pcs, var_frac = perform_eof_analysis(sst_anomalies, weights)
    
    # Plot EOF patterns
    plot_eof_patterns(eofs, var_frac)
    
    # Extract principal components
    print("Standardizing PC time series...")
    pc1 = pcs[:, 0].values
    pc2 = pcs[:, 1].values
    
    # Make sure PCs are standardized
    pc1 = (pc1 - np.mean(pc1)) / np.std(pc1)
    pc2 = (pc2 - np.mean(pc2)) / np.std(pc2)
    
    # Extract variables as numpy arrays for regression
    sst_array = sst_anomalies['SST'].values
    u_array = u_wind['UWND'].squeeze(dim='LEV17_17', drop=True).values
    v_array = v_wind['VWND'].squeeze(dim='LEV17_17', drop=True).values
    
    # Create dimension dictionaries
    sst_dims = {
        'lat': sst.lat.values,
        'lon': sst.lon.values
    }
    
    wind_dims = {
        'lat': u_wind.lat.values,
        'lon': u_wind.lon.values
    }
    
    # Calculate regressions manually for PC1
    print("Calculating regressions for PC1...")
    sst_reg_pc1 = perform_regression_manual(pc1, sst_array, sst_dims)
    u_reg_pc1 = perform_regression_manual(pc1, u_array, wind_dims)
    v_reg_pc1 = perform_regression_manual(pc1, v_array, wind_dims)
    
    # Calculate regressions manually for PC2
    print("Calculating regressions for PC2...")
    sst_reg_pc2 = perform_regression_manual(pc2, sst_array, sst_dims)
    u_reg_pc2 = perform_regression_manual(pc2, u_array, wind_dims)
    v_reg_pc2 = perform_regression_manual(pc2, v_array, wind_dims)
    
    # Plot regression patterns
    if isinstance(sst_reg_pc1, xr.DataArray):
        # Using xarray plotting
        plot_sst_wind_regression(1, 
                               sst_reg_pc1.values, u_reg_pc1.values, v_reg_pc1.values,
                               sst_dims['lat'], sst_dims['lon'],
                               wind_dims['lat'], wind_dims['lon'])
    else:
        # Using numpy arrays
        plot_sst_wind_regression(1, 
                               sst_reg_pc1, u_reg_pc1, v_reg_pc1,
                               sst_dims['lat'], sst_dims['lon'],
                               wind_dims['lat'], wind_dims['lon'])
    
    if isinstance(sst_reg_pc2, xr.DataArray):
        # Using xarray plotting
        plot_sst_wind_regression(2, 
                               sst_reg_pc2.values, u_reg_pc2.values, v_reg_pc2.values,
                               sst_dims['lat'], sst_dims['lon'],
                               wind_dims['lat'], wind_dims['lon'])
    else:
        # Using numpy arrays
        plot_sst_wind_regression(2, 
                               sst_reg_pc2, u_reg_pc2, v_reg_pc2,
                               sst_dims['lat'], sst_dims['lon'],
                               wind_dims['lat'], wind_dims['lon'])
    
    # Explanation of the patterns
    print("\nEXPLANATION OF RESULTS:")
    print("----------------------")
    print("PC1 Regression Pattern:")
    print("The SST pattern shows the characteristic ENSO signal with warming/cooling")
    print("in the tropical Pacific. The wind vectors reveal the associated changes in")
    print("atmospheric circulation, with weakened trade winds during El Niño phases.")
    print("\nPC2 Regression Pattern:")
    print("The SST pattern likely represents another mode of climate variability such")
    print("as the Pacific Decadal Oscillation (PDO). The wind patterns show how")
    print("atmospheric circulation responds to this mode of variability.")

if __name__ == "__main__":
    main()