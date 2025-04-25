import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from io import StringIO
import re

# Set plotting parameters for better readability
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# 1. Load ENSO (Nino 3.4) data with updated URL and parsing
enso_url = "https://psl.noaa.gov/data/timeseries/month/data/nino34.long.anom.data"
try:
    response = requests.get(enso_url)
    response.raise_for_status()
    
    # Print the first few lines to inspect the data format
    print("Raw ENSO data (first 5 lines):")
    lines = response.text.strip().split('\n')[:5]
    for line in lines:
        print(line)
    
    # Process the data line by line with improved parsing
    enso_data = []
    year_pattern = re.compile(r'^\d{4}$|^\d{4}\s')
    
    for line in response.text.strip().split('\n'):
        line = line.strip()
        
        # Skip header lines
        if not year_pattern.match(line) or "year" in line.lower():
            continue
        
        parts = line.strip().split()
        try:
            year = int(parts[0])
            # Get monthly values (there should be 12 values after the year)
            for month, value_str in enumerate(parts[1:13], 1):
                try:
                    value = float(value_str)
                    date = pd.Timestamp(year=year, month=month, day=1)
                    enso_data.append({'Date': date, 'ENSO': value})
                except (ValueError, IndexError) as e:
                    print(f"Error processing ENSO value for {year}-{month}: {e}")
        except (ValueError, IndexError) as e:
            print(f"Error processing ENSO year: {e}")
    
    # Create DataFrame
    if enso_data:
        enso_df = pd.DataFrame(enso_data)
        enso_df.set_index('Date', inplace=True)
        
        # Extend period to 1900-2023
        enso_df = enso_df[(enso_df.index >= '1900-01-01') & (enso_df.index <= '2023-12-31')]
        
        print(f"ENSO data loaded successfully. Shape: {enso_df.shape}")
        print("ENSO data - first few rows:")
        print(enso_df.head())
    else:
        raise ValueError("No valid ENSO data found")

except Exception as e:
    print(f"Error loading ENSO data: {e}")
    # Create empty dataframe if error
    enso_df = pd.DataFrame(index=pd.date_range(start='1900-01-01', end='2023-12-31', freq='MS'),
                          columns=['ENSO'])
    enso_df['ENSO'] = np.nan

# 2. Load IOD/DMI data with updated parsing
iod_url = "https://psl.noaa.gov/gcos_wgsp/Timeseries/Data/dmi.had.long.data"
try:
    response = requests.get(iod_url)
    response.raise_for_status()
    
    # Print the first few lines to inspect the data format
    print("\nRaw IOD data (first 5 lines):")
    lines = response.text.strip().split('\n')[:5]
    for line in lines:
        print(line)
    
    # Process the data line by line with improved parsing
    iod_data = []
    year_pattern = re.compile(r'^\d{4}$|^\d{4}\s')
    
    for line in response.text.strip().split('\n'):
        line = line.strip()
        
        # Skip header lines (first line is year range)
        if not year_pattern.match(line) or line == "1870 2025":
            continue
        
        parts = line.strip().split()
        try:
            year = int(parts[0])
            # Get monthly values (there should be 12 values after the year)
            for month, value_str in enumerate(parts[1:13], 1):
                try:
                    value = float(value_str)
                    date = pd.Timestamp(year=year, month=month, day=1)
                    iod_data.append({'Date': date, 'IOD': value})
                except (ValueError, IndexError) as e:
                    print(f"Error processing IOD value for {year}-{month}: {e}")
        except (ValueError, IndexError) as e:
            print(f"Error processing IOD year: {e}")
    
    # Create DataFrame
    if iod_data:
        iod_df = pd.DataFrame(iod_data)
        iod_df.set_index('Date', inplace=True)
        
        # Extend period to 1900-2023
        iod_df = iod_df[(iod_df.index >= '1900-01-01') & (iod_df.index <= '2023-12-31')]
        
        print(f"IOD data loaded successfully. Shape: {iod_df.shape}")
        print("IOD data - first few rows:")
        print(iod_df.head())
    else:
        raise ValueError("No valid IOD data found")

except Exception as e:
    print(f"Error loading IOD data: {e}")
    # Create empty dataframe if error
    iod_df = pd.DataFrame(index=pd.date_range(start='1900-01-01', end='2023-12-31', freq='MS'),
                         columns=['IOD'])
    iod_df['IOD'] = np.nan

# 3. Load PDO data
# For PDO data, we'll read directly from a URL rather than a local file
pdo_url = "https://psl.noaa.gov/pdo/data/pdo.timeseries.ersstv5.csv"
try:
    response = requests.get(pdo_url)
    response.raise_for_status()
    
    pdo_data = pd.read_csv(StringIO(response.text))
    pdo_df = pdo_data.copy()
    
    # Check if the data needs to be processed based on its format
    if 'Date' in pdo_df.columns:
        pdo_df['Date'] = pd.to_datetime(pdo_df['Date'])
        pdo_df.set_index('Date', inplace=True)
        pdo_df.columns = ['PDO']
    else:
        # Format might be different, try to adapt
        # Assuming first column is year, followed by 12 monthly values
        pdo_rows = []
        for _, row in pdo_df.iterrows():
            year = int(row.iloc[0])
            for month, value in enumerate(row.iloc[1:13], 1):
                try:
                    pdo_rows.append({
                        'Date': pd.Timestamp(year=year, month=month, day=1),
                        'PDO': float(value)
                    })
                except:
                    pass
        
        pdo_df = pd.DataFrame(pdo_rows)
        pdo_df.set_index('Date', inplace=True)
    
    pdo_df['PDO'] = pd.to_numeric(pdo_df['PDO'], errors='coerce')
    pdo_df = pdo_df.dropna()
    pdo_df = pdo_df[(pdo_df.index >= '1900-01-01') & (pdo_df.index <= '2023-12-31')]
    print(f"\nPDO data loaded successfully. Shape: {pdo_df.shape}")
    print("PDO data - first few rows:")
    print(pdo_df.head())

except Exception as e:
    print(f"Error loading PDO data: {e}")
    # Try alternative URL if the first one fails
    try:
        pdo_url_alt = "https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/index/ersst.v5.pdo.dat"
        response = requests.get(pdo_url_alt)
        response.raise_for_status()
        
        # Process the data with different format
        lines = response.text.strip().split('\n')
        pdo_rows = []
        
        for line in lines:
            if line.strip() and not line.startswith('#'):
                parts = line.strip().split()
                if len(parts) >= 13:  # Year + 12 months
                    try:
                        year = int(parts[0])
                        for month, value_str in enumerate(parts[1:13], 1):
                            try:
                                value = float(value_str)
                                pdo_rows.append({
                                    'Date': pd.Timestamp(year=year, month=month, day=1),
                                    'PDO': value
                                })
                            except:
                                pass
                    except:
                        pass
        
        pdo_df = pd.DataFrame(pdo_rows)
        if not pdo_df.empty:
            pdo_df.set_index('Date', inplace=True)
            pdo_df = pdo_df[(pdo_df.index >= '1900-01-01') & (pdo_df.index <= '2023-12-31')]
            print(f"PDO data loaded from alternative source. Shape: {pdo_df.shape}")
        else:
            raise ValueError("No valid PDO data found")
    
    except Exception as e2:
        print(f"Error loading PDO data from alternative source: {e2}")
        # Create empty dataframe if all attempts fail
        pdo_df = pd.DataFrame(index=pd.date_range(start='1900-01-01', end='2023-12-31', freq='MS'),
                             columns=['PDO'])
        pdo_df['PDO'] = np.nan

# 4. Load IPO data
ipo_url = "https://psl.noaa.gov/data/timeseries/IPOTPI/tpi.timeseries.hadisst11.data"
try:
    response = requests.get(ipo_url)
    response.raise_for_status()
    
    # Process the data line by line, with improved parsing
    ipo_data = []
    year_pattern = re.compile(r'^\d{4}$|^\d{4}\s')
    
    for line in response.text.strip().split('\n'):
        line = line.strip()
        
        # Skip header lines
        if not year_pattern.match(line) or "year" in line.lower():
            continue
        
        parts = line.strip().split()
        try:
            year = int(parts[0])
            # Process only years in our target range
            if 1900 <= year <= 2023:
                # Get monthly values
                for month, value_str in enumerate(parts[1:13], 1):
                    try:
                        value = float(value_str)
                        date = pd.Timestamp(year=year, month=month, day=1)
                        ipo_data.append({'Date': date, 'IPO': value})
                    except (ValueError, IndexError) as e:
                        print(f"Error processing IPO value for {year}-{month}: {e}")
        except (ValueError, IndexError) as e:
            print(f"Error processing IPO year: {e}")
    
    if not ipo_data:
        raise ValueError("No valid IPO data found")
        
    # Create DataFrame
    ipo_df = pd.DataFrame(ipo_data)
    ipo_df.set_index('Date', inplace=True)
    
    print(f"IPO data loaded successfully. Shape: {ipo_df.shape}")
    print("IPO data - first few rows:")
    print(ipo_df.head())

except Exception as e:
    print(f"Error loading IPO data: {e}")
    # Create empty dataframe if error
    ipo_df = pd.DataFrame(index=pd.date_range(start='1900-01-01', end='2023-12-31', freq='MS'),
                         columns=['IPO'])
    ipo_df['IPO'] = np.nan

# 5. Apply 10-year running mean filter (121 months) as in the first code
def apply_running_mean(series, window=121):  # 121 months = ~10 years
    """Apply running mean to filter the time series."""
    # Check if series has data before filtering
    if series.isna().all():
        return series
    return series.rolling(window=window, center=True).mean()

# Apply filters to all indices
print("\nApplying 10-year running mean filter...")

# Apply filter to ENSO data
if not enso_df['ENSO'].isna().all():
    enso_filtered = apply_running_mean(enso_df['ENSO'])
else:
    enso_filtered = pd.Series(np.nan, index=enso_df.index)

# Apply filter to IOD data
if not iod_df['IOD'].isna().all():
    iod_filtered = apply_running_mean(iod_df['IOD'])
else:
    iod_filtered = pd.Series(np.nan, index=iod_df.index)

# Apply filter to PDO data
if not pdo_df['PDO'].isna().all():
    pdo_filtered = apply_running_mean(pdo_df['PDO'])
else:
    pdo_filtered = pd.Series(np.nan, index=pdo_df.index)

# Apply filter to IPO data
if not ipo_df['IPO'].isna().all():
    ipo_filtered = apply_running_mean(ipo_df['IPO'])
else:
    ipo_filtered = pd.Series(np.nan, index=ipo_df.index)

# 6. Calculate correlations between time series
print("\nCalculating correlations...")

# Create a dictionary to store all original and filtered series
all_series = {
    'ENSO': enso_df['ENSO'] if 'ENSO' in enso_df else pd.Series(),
    'IOD': iod_df['IOD'] if 'IOD' in iod_df else pd.Series(),
    'PDO': pdo_df['PDO'] if 'PDO' in pdo_df else pd.Series(),
    'IPO': ipo_df['IPO'] if 'IPO' in ipo_df else pd.Series(),
    'ENSO_filtered': enso_filtered,
    'IOD_filtered': iod_filtered,
    'PDO_filtered': pdo_filtered,
    'IPO_filtered': ipo_filtered
}

# Create a DataFrame with all series
all_df = pd.DataFrame(all_series)

# Calculate correlations for original data
original_df = all_df[['ENSO', 'IOD', 'PDO', 'IPO']].dropna()
if not original_df.empty:
    original_corr = original_df.corr()
    print("\nCorrelation matrix for original indices:")
    print(original_corr)
else:
    print("\nWarning: Not enough data to calculate correlations for original indices.")

# Calculate correlations for filtered data
filtered_df = all_df[['ENSO_filtered', 'IOD_filtered', 'PDO_filtered', 'IPO_filtered']].dropna()
if not filtered_df.empty:
    filtered_corr = filtered_df.corr()
    print("\nCorrelation matrix for 10-year filtered indices:")
    print(filtered_corr)
else:
    print("\nWarning: Not enough data to calculate correlations for filtered indices.")

# 7. Create plots to visualize the results
# Plot 1: ENSO and IOD (original and filtered)
fig, axs = plt.subplots(2, 1, figsize=(14, 10))

# Plot original time series
axs[0].plot(enso_df.index, enso_df['ENSO'], 'b-', label='ENSO (Nino 3.4)')
axs[0].plot(iod_df.index, iod_df['IOD'], 'r-', label='IOD/DMI')
axs[0].set_title('Original ENSO and IOD Time Series (1900-2023)')
axs[0].set_xlabel('Year')
axs[0].set_ylabel('Index Value')
axs[0].legend()
axs[0].grid(True)

# Plot filtered time series
axs[1].plot(enso_filtered.index, enso_filtered, 'b-', label='ENSO (10-year running mean)')
axs[1].plot(iod_filtered.index, iod_filtered, 'r-', label='IOD (10-year running mean)')
axs[1].set_title('Filtered ENSO and IOD Time Series (1900-2023)')
axs[1].set_xlabel('Year')
axs[1].set_ylabel('Index Value')
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.savefig('ENSO_IOD_Timeseries.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot 2: PDO and IPO (original and filtered)
fig, axs = plt.subplots(2, 1, figsize=(14, 10))

# Plot original time series
axs[0].plot(pdo_df.index, pdo_df['PDO'], 'b-', label='PDO')
axs[0].plot(ipo_df.index, ipo_df['IPO'], 'r-', label='IPO')
axs[0].set_title('Original PDO and IPO Time Series (1900-2023)')
axs[0].set_xlabel('Year')
axs[0].set_ylabel('Index Value')
axs[0].legend()
axs[0].grid(True)

# Plot filtered time series
axs[1].plot(pdo_filtered.index, pdo_filtered, 'b-', label='PDO (10-year running mean)')
axs[1].plot(ipo_filtered.index, ipo_filtered, 'r-', label='IPO (10-year running mean)')
axs[1].set_title('Filtered PDO and IPO Time Series (1900-2023)')
axs[1].set_xlabel('Year')
axs[1].set_ylabel('Index Value')
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.savefig('PDO_IPO_Timeseries.png', dpi=300, bbox_inches='tight')
plt.show()

# 8. Save the filtered data to CSV
filtered_data = pd.DataFrame({
    'ENSO': enso_df['ENSO'] if 'ENSO' in enso_df else np.nan,
    'ENSO_filtered': enso_filtered,
    'IOD': iod_df['IOD'] if 'IOD' in iod_df else np.nan,
    'IOD_filtered': iod_filtered,
    'PDO': pdo_df['PDO'] if 'PDO' in pdo_df else np.nan,
    'PDO_filtered': pdo_filtered,
    'IPO': ipo_df['IPO'] if 'IPO' in ipo_df else np.nan,
    'IPO_filtered': ipo_filtered
})

filtered_data.to_csv('climate_indices_10yr_filtered.csv')

print("\nFiltered data saved to 'climate_indices_10yr_filtered.csv'")
print("\nTask completed successfully!")