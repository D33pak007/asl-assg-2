import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import requests
import re

def download_nino34_data(url):
    """Download and parse the Nino3.4 data from NOAA"""
    response = requests.get(url)
    lines = response.text.strip().split('\n')
    
    # Find where the actual data starts (after header)
    data_start = 0
    for i, line in enumerate(lines):
        # Look for lines with years (4 digits at start)
        if re.match(r'^\d{4}', line.strip()):
            data_start = i
            break
    
    # Parse the data
    data_rows = lines[data_start:]
    years = []
    monthly_values = []
    
    for row in data_rows:
        values = row.split()
        if len(values) >= 13:  # Year + 12 months
            try:
                year = int(values[0])
                # The rest are monthly values
                monthly = [float(v) if v != '-99.99' else np.nan for v in values[1:13]]  # Handle missing values
                years.append(year)
                monthly_values.append(monthly)
            except ValueError:
                continue  # Skip rows that don't have proper numeric data
    
    # Convert to DataFrame
    df = pd.DataFrame(monthly_values, index=years, columns=range(1, 13))
    
    # Reshape to time series
    dates = []
    values = []
    
    for year in df.index:
        for month in range(1, 13):
            dates.append(pd.Timestamp(int(year), month, 15))
            values.append(df.loc[year, month])
    
    # Create a pandas Series with datetime index
    return pd.Series(values, index=dates, name='Nino3.4')

def download_dmi_data(url):
    """Download and parse the DMI data from NOAA"""
    response = requests.get(url)
    lines = response.text.strip().split('\n')
    
    # Find where the actual data starts (after header)
    data_start = 0
    for i, line in enumerate(lines):
        # Look for lines with years (4 digits at start)
        if re.match(r'^\d{4}', line.strip()):
            data_start = i
            break
    
    # Parse the data
    data_rows = lines[data_start:]
    years = []
    monthly_values = []
    
    for row in data_rows:
        values = row.split()
        if len(values) >= 13:  # Year + 12 months
            try:
                year = int(values[0])
                # The rest are monthly values
                monthly = [float(v) if v != '-99.99' else np.nan for v in values[1:13]]  # Handle missing values
                years.append(year)
                monthly_values.append(monthly)
            except ValueError:
                continue  # Skip rows that don't have proper numeric data
    
    # Convert to DataFrame
    df = pd.DataFrame(monthly_values, index=years, columns=range(1, 13))
    
    # Reshape to time series
    dates = []
    values = []
    
    for year in df.index:
        for month in range(1, 13):
            dates.append(pd.Timestamp(int(year), month, 15))
            values.append(df.loc[year, month])
    
    # Create a pandas Series with datetime index
    return pd.Series(values, index=dates, name='DMI')

def butter_highpass(cutoff, fs, order=5):
    """Design a butterworth highpass filter"""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high')
    return b, a

def butter_lowpass(cutoff, fs, order=5):
    """Design a butterworth lowpass filter"""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low')
    return b, a

def apply_filter(data, cutoff, fs, filter_type='low', order=5):
    """Apply a butterworth filter to the data"""
    # Handle NaN values by replacing them with interpolated values for filtering
    # Create a copy of the data to avoid modifying the original
    data_copy = np.copy(data)
    
    # Find NaN values
    nan_mask = np.isnan(data_copy)
    
    # If there are NaN values, interpolate them
    if np.any(nan_mask):
        indices = np.arange(len(data_copy))
        valid_indices = indices[~nan_mask]
        valid_data = data_copy[~nan_mask]
        
        # Perform linear interpolation
        data_copy[nan_mask] = np.interp(indices[nan_mask], valid_indices, valid_data)
    
    if filter_type == 'low':
        b, a = butter_lowpass(cutoff, fs, order=order)
    elif filter_type == 'high':
        b, a = butter_highpass(cutoff, fs, order=order)
    else:
        raise ValueError("Filter type must be 'low' or 'high'")
    
    # Apply filter
    y = filtfilt(b, a, data_copy)
    
    # Restore NaN values where they were in the original data
    y[nan_mask] = np.nan
    
    return y

# URLs for data
nino34_url = "https://psl.noaa.gov/data/timeseries/month/data/nino34.long.anom.data"
dmi_url = "https://psl.noaa.gov/gcos_wgsp/Timeseries/Data/dmi.had.long.data"

# Download data
print("Downloading Nino3.4 data...")
nino34 = download_nino34_data(nino34_url)
print("Downloading DMI data...")
dmi = download_dmi_data(dmi_url)

# Filter data for the 1900-2023 period
start_date = '1900-01-01'
end_date = '2023-12-31'
nino34 = nino34[(nino34.index >= start_date) & (nino34.index <= end_date)]
dmi = dmi[(dmi.index >= start_date) & (dmi.index <= end_date)]

print(f"Filtered data range: {nino34.index.min().year} to {nino34.index.max().year}")

# Fill missing values for filtering
nino34_filled = nino34.interpolate(method='linear')
dmi_filled = dmi.interpolate(method='linear')

# Calculate sampling frequency (12 samples per year)
fs = 12  # 12 months per year

# For interannual filter (highpass for < 8 years)
highpass_cutoff = 1/(8*12)  # Frequency in cycles/month for 8 years

# For interdecadal filter (lowpass for > 8 years)
lowpass_cutoff = 1/(8*12)  # Frequency in cycles/month for 8 years

# Apply filters using the filled data for filtering
nino34_highpass = apply_filter(nino34_filled.values, highpass_cutoff, fs, filter_type='high', order=5)
nino34_lowpass = apply_filter(nino34_filled.values, lowpass_cutoff, fs, filter_type='low', order=5)

dmi_highpass = apply_filter(dmi_filled.values, highpass_cutoff, fs, filter_type='high', order=5)
dmi_lowpass = apply_filter(dmi_filled.values, lowpass_cutoff, fs, filter_type='low', order=5)

# Create DataFrames with the filtered data
filtered_data = pd.DataFrame({
    'Nino3.4': nino34,
    'Nino3.4 Highpass (<8 years)': nino34_highpass,
    'Nino3.4 Lowpass (>8 years)': nino34_lowpass,
    'DMI': dmi,
    'DMI Highpass (<8 years)': dmi_highpass,
    'DMI Lowpass (>8 years)': dmi_lowpass
}, index=nino34.index)

# Create a combined plot of filtered indices
plt.figure(figsize=(15, 10))

# Plot the interannual filtered indices (highpass <8 years)
plt.subplot(2, 1, 1)
plt.plot(filtered_data.index, filtered_data['Nino3.4 Highpass (<8 years)'], label='Nino3.4', color='blue')
plt.plot(filtered_data.index, filtered_data['DMI Highpass (<8 years)'], label='DMI', color='red')
plt.title('(i) Interannually Filtered Indices (<8 years), 1900-2023', fontsize=14, fontweight='bold')
plt.ylabel('Index Value', fontsize=12)
plt.legend(loc='best')
plt.grid(True)
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)

# Plot the interdecadal filtered indices (lowpass >8 years)
plt.subplot(2, 1, 2)
plt.plot(filtered_data.index, filtered_data['Nino3.4 Lowpass (>8 years)'], label='Nino3.4', color='blue')
plt.plot(filtered_data.index, filtered_data['DMI Lowpass (>8 years)'], label='DMI', color='red')
plt.title('(ii) Interdecadally Filtered Indices (>8 years), 1900-2023', fontsize=14, fontweight='bold')
plt.xlabel('Year', fontsize=12)
plt.ylabel('Index Value', fontsize=12)
plt.legend(loc='best')
plt.grid(True)
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)

plt.tight_layout()
# Save the plot
plt.savefig('filtered_climate_indices.png', dpi=300, bbox_inches='tight')
plt.show()

print("Plot saved as 'filtered_climate_indices.png'")