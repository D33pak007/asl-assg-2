import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import urllib.request
import io
import re

# URLs for the data
nino34_url = "https://psl.noaa.gov/data/timeseries/month/data/nino34.long.anom.data"
dmi_url = "https://psl.noaa.gov/gcos_wgsp/Timeseries/Data/dmi.had.long.data"

# Function to read Nino3.4 data specifically
def read_nino34_data(url):
    with urllib.request.urlopen(url) as response:
        data = response.read().decode('utf-8')
    
    lines = data.strip().split('\n')
    years = []
    values = []
    
    for line in lines:
        if line.strip() and not line.startswith('#'):
            parts = line.split()
            if len(parts) >= 13:  # Year + 12 months
                year = int(parts[0])
                monthly_values = [float(val) for val in parts[1:13]]
                years.extend([year] * 12)
                values.extend(monthly_values)
    
    months = np.tile(np.arange(1, 13), len(years) // 12)
    
    df = pd.DataFrame({
        'Year': years,
        'Month': months,
        'Value': values
    })
    
    return df

# Function to read DMI data specifically
def read_dmi_data(url):
    with urllib.request.urlopen(url) as response:
        data = response.read().decode('utf-8')
    
    lines = data.strip().split('\n')
    years = []
    values = []
    
    for line in lines:
        if line.strip() and not line.startswith('#'):
            parts = line.split()
            if len(parts) >= 13:  # Year + 12 months
                year = int(parts[0])
                monthly_values = [float(val) for val in parts[1:13]]
                years.extend([year] * 12)
                values.extend(monthly_values)
    
    months = np.tile(np.arange(1, 13), len(years) // 12)
    
    df = pd.DataFrame({
        'Year': years,
        'Month': months,
        'Value': values
    })
    
    return df

# Read Nino3.4 data
nino34_ts = read_nino34_data(nino34_url)

# Read DMI data
dmi_ts = read_dmi_data(dmi_url)

# Create datetime indices
nino34_ts['Date'] = pd.to_datetime(nino34_ts['Year'].astype(str) + '-' + nino34_ts['Month'].astype(str))
nino34_ts = nino34_ts.set_index('Date')

dmi_ts['Date'] = pd.to_datetime(dmi_ts['Year'].astype(str) + '-' + dmi_ts['Month'].astype(str))
dmi_ts = dmi_ts.set_index('Date')

# Filter data for the period 1950-2023
start_date = '1950-01-01'
end_date = '2023-12-31'
nino34_filtered = nino34_ts[(nino34_ts.index >= start_date) & (nino34_ts.index <= end_date)]
dmi_filtered = dmi_ts[(dmi_ts.index >= start_date) & (dmi_ts.index <= end_date)]

# Create a figure with two subplots
plt.figure(figsize=(14, 8))

# Plot Nino3.4 time series
plt.subplot(2, 1, 1)
plt.plot(nino34_filtered.index, nino34_filtered['Value'], 'r-', linewidth=1)
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.fill_between(nino34_filtered.index, nino34_filtered['Value'], 0, 
                 where=nino34_filtered['Value'] > 0, facecolor='red', alpha=0.3)
plt.fill_between(nino34_filtered.index, nino34_filtered['Value'], 0, 
                 where=nino34_filtered['Value'] < 0, facecolor='blue', alpha=0.3)
plt.title('ENSO (Nino 3.4) Index (1950-2023)', fontsize=14)
plt.ylabel('Temperature Anomaly (°C)', fontsize=12)
plt.grid(True, alpha=0.3)

# Plot DMI time series
plt.subplot(2, 1, 2)
plt.plot(dmi_filtered.index, dmi_filtered['Value'], 'b-', linewidth=1)
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.fill_between(dmi_filtered.index, dmi_filtered['Value'], 0, 
                 where=dmi_filtered['Value'] > 0, facecolor='red', alpha=0.3)
plt.fill_between(dmi_filtered.index, dmi_filtered['Value'], 0, 
                 where=dmi_filtered['Value'] < 0, facecolor='blue', alpha=0.3)
plt.title('Indian Ocean Dipole/Dipole Mode Index (IOD/DMI) (1950-2023)', fontsize=14)
plt.ylabel('Temperature Anomaly (°C)', fontsize=12)
plt.grid(True, alpha=0.3)

# Enhance the layout
plt.tight_layout()
plt.savefig('ENSO_IOD_timeseries_1950_2023.png', dpi=300, bbox_inches='tight')
plt.show()

# Calculate basic statistics
print("ENSO (Nino 3.4) Statistics (1950-2023):")
print(f"Mean: {nino34_filtered['Value'].mean():.3f}")
print(f"Standard Deviation: {nino34_filtered['Value'].std():.3f}")
print(f"Maximum: {nino34_filtered['Value'].max():.3f}")
print(f"Minimum: {nino34_filtered['Value'].min():.3f}")

print("\nIOD/DMI Statistics (1950-2023):")
print(f"Mean: {dmi_filtered['Value'].mean():.3f}")
print(f"Standard Deviation: {dmi_filtered['Value'].std():.3f}")
print(f"Maximum: {dmi_filtered['Value'].max():.3f}")
print(f"Minimum: {dmi_filtered['Value'].min():.3f}")