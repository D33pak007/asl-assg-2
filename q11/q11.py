import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from io import StringIO
import re

# Set plotting parameters for better readability
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# 1. Load PDO data from local file
pdo_file_path = r"C:\Users\Lenovo\Downloads\pdo.timeseries.sstens.csv"

# Load PDO data
pdo_data = pd.read_csv(pdo_file_path)
pdo_df = pdo_data.copy()
pdo_df['Date'] = pd.to_datetime(pdo_df['Date'])
pdo_df.set_index('Date', inplace=True)
pdo_df.columns = ['PDO']
pdo_df['PDO'] = pd.to_numeric(pdo_df['PDO'], errors='coerce')
pdo_df = pdo_df.dropna()
pdo_df = pdo_df[(pdo_df.index >= '1900-01-01') & (pdo_df.index <= '2023-12-31')]
print(f"PDO data loaded successfully. Shape: {pdo_df.shape}")

# 2. Load IPO data from provided URL with more robust parsing
ipo_url = "https://psl.noaa.gov/data/timeseries/IPOTPI/tpi.timeseries.hadisst11.data"
try:
    response = requests.get(ipo_url)
    response.raise_for_status()
    
    # Print the first few lines of data to inspect
    print("Raw IPO data (first 5 lines):")
    for line in response.text.strip().split('\n')[:5]:
        print(line)
    
    # Process the data line by line, but more cautiously
    years = []
    months = []
    values = []
    
    for line in response.text.strip().split('\n'):
        # Skip lines that don't start with a numeric year
        parts = line.strip().split()
        if not parts or not re.match(r'^\d{4}$', parts[0]):
            print(f"Skipping non-year line: {line[:50]}...")
            continue
        
        year = int(parts[0])
        # Check if there are at least 13 columns (year + 12 months)
        if len(parts) < 13:
            print(f"Skipping incomplete line for year {year}")
            continue
            
        # Extract monthly values
        for month, value_str in enumerate(parts[1:13], 1):
            try:
                value = float(value_str)
                years.append(year)
                months.append(month)
                values.append(value)
            except ValueError:
                print(f"Skipping invalid value '{value_str}' for {year}-{month}")
    
    if not years:
        raise ValueError("No valid data found in IPO file")
    
    # Create a DataFrame from the parsed data
    ipo_data = pd.DataFrame({
        'Year': years,
        'Month': months,
        'IPO': values
    })
    
    # Create proper date index
    ipo_data['Date'] = pd.to_datetime(ipo_data['Year'].astype(str) + '-' + 
                                     ipo_data['Month'].astype(str) + '-01')
    
    # Set the date as index and filter to required range
    ipo_df = ipo_data[['Date', 'IPO']].copy()
    ipo_df.set_index('Date', inplace=True)
    ipo_df = ipo_df[(ipo_df.index >= '1900-01-01') & (ipo_df.index <= '2023-12-31')]
    
    print(f"IPO data loaded successfully. Shape: {ipo_df.shape}")
    print("IPO data - first few rows:")
    print(ipo_df.head())

except Exception as e:
    print(f"Error loading IPO data: {e}")
    print("Trying alternate method for IPO data...")
    
    # Alternative direct method - try a different approach
    try:
        # Try manual download and processing
        ipo_data_alt = []
        lines = response.text.strip().split('\n')
        year_pattern = re.compile(r'^\d{4}')
        
        for line in lines:
            # Find lines starting with a 4-digit year
            if year_pattern.match(line.strip()):
                parts = line.strip().split()
                year = int(parts[0])
                
                # Only process years within our range
                if 1900 <= year <= 2023:
                    # Get monthly values
                    try:
                        monthly_values = [float(v) for v in parts[1:13]]
                        
                        # Create row for each month
                        for month, value in enumerate(monthly_values, 1):
                            date = f"{year}-{month:02d}-01"
                            ipo_data_alt.append({'Date': date, 'IPO': value})
                    except (ValueError, IndexError) as e:
                        print(f"Error processing year {year}: {e}")
        
        if not ipo_data_alt:
            raise ValueError("No valid data found using alternate method")
            
        # Create DataFrame from the processed data
        ipo_df = pd.DataFrame(ipo_data_alt)
        ipo_df['Date'] = pd.to_datetime(ipo_df['Date'])
        ipo_df.set_index('Date', inplace=True)
        
        print(f"IPO data loaded using alternate method. Shape: {ipo_df.shape}")
        print(ipo_df.head())
    
    except Exception as e2:
        print(f"Alternative method also failed: {e2}")
        
        # As a last resort, try to manually parse from the raw text
        try:
            # Download data directly from their website to a temporary file
            import tempfile
            import os
            import urllib.request
            
            temp_dir = tempfile.gettempdir()
            temp_file = os.path.join(temp_dir, 'ipo_data.txt')
            
            # Try using a different URL format or direct download
            urllib.request.urlretrieve("https://psl.noaa.gov/data/timeseries/IPOTPI/tpi.timeseries.hadisst11.data", temp_file)
            
            print(f"Downloaded IPO data to {temp_file}")
            
            # Read the file and parse manually
            with open(temp_file, 'r') as f:
                content = f.read()
            
            # Create a structured dataframe
            dates = []
            values = []
            
            for line in content.strip().split('\n'):
                parts = line.strip().split()
                if len(parts) >= 13 and parts[0].isdigit():
                    year = int(parts[0])
                    if 1900 <= year <= 2023:
                        for m, val in enumerate(parts[1:13], 1):
                            try:
                                dates.append(pd.Timestamp(year=year, month=m, day=1))
                                values.append(float(val))
                            except:
                                pass
            
            if dates:
                ipo_df = pd.DataFrame({'IPO': values}, index=dates)
                print(f"IPO data loaded using direct file parsing. Shape: {ipo_df.shape}")
            else:
                raise ValueError("No valid data found in downloaded file")
                
        except Exception as e3:
            print(f"All methods failed. Creating empty IPO dataframe: {e3}")
            # Create empty dataframe with proper date index if all methods fail
            ipo_df = pd.DataFrame(index=pd.date_range(start='1900-01-01', end='2023-12-31', freq='MS'),
                                columns=['IPO'])
            ipo_df['IPO'] = np.nan

# 3. Apply low-pass filter (10-year running mean)
def apply_running_mean(series, window=121):  # 121 months = ~10 years
    """Apply running mean to filter the time series."""
    return series.rolling(window=window, center=True).mean()

# Apply filter to both time series
print("\nApplying filters...")
pdo_filtered = apply_running_mean(pdo_df['PDO'])
ipo_filtered = apply_running_mean(ipo_df['IPO'])

# 4. Calculate correlation between the time series
print("\nCalculating correlations...")
# Merge dataframes on date index
merged_df = pd.merge(pdo_df, ipo_df, left_index=True, right_index=True, how='inner')
merged_df = merged_df.dropna()

if not merged_df.empty:
    corr_original = merged_df['PDO'].corr(merged_df['IPO'])
    print(f"Correlation between unfiltered series: {corr_original:.4f}")
    print(f"Number of overlapping data points: {len(merged_df)}")
else:
    corr_original = None
    print("No overlapping data points for correlation calculation")

# For filtered data, merge and drop NaN values
merged_filtered = pd.DataFrame({
    'PDO_filtered': pdo_filtered,
    'IPO_filtered': ipo_filtered
}, index=pdo_df.index)
merged_filtered = merged_filtered.dropna()

if not merged_filtered.empty:
    corr_filtered = merged_filtered['PDO_filtered'].corr(merged_filtered['IPO_filtered'])
    print(f"Correlation between filtered series: {corr_filtered:.4f}")
    print(f"Number of overlapping filtered data points: {len(merged_filtered)}")
else:
    corr_filtered = None
    print("No overlapping filtered data points for correlation calculation")

# 5. Create plots
print("\nCreating plots...")
fig, axs = plt.subplots(2, 1, figsize=(14, 10))

# Plot original time series
axs[0].plot(pdo_df.index, pdo_df['PDO'], 'b-', label='PDO')
if not ipo_df['IPO'].isna().all():  # Only plot if there's data
    axs[0].plot(ipo_df.index, ipo_df['IPO'], 'r-', label='IPO')
axs[0].set_title('Original PDO and IPO Time Series (1900-2023)')
axs[0].set_xlabel('Year')
axs[0].set_ylabel('Index Value')
axs[0].legend()
axs[0].grid(True)

# Plot filtered time series
axs[1].plot(pdo_filtered.index, pdo_filtered, 'b-', label='PDO (10-year running mean)')
if not ipo_filtered.isna().all():  # Only plot if there's data
    axs[1].plot(ipo_filtered.index, ipo_filtered, 'r-', label='IPO (10-year running mean)')
axs[1].set_title('Filtered PDO and IPO Time Series (1900-2023)')
axs[1].set_xlabel('Year')
axs[1].set_ylabel('Index Value')
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()

# Print correlation results
print("\nCorrelation Results:")
print(f"Correlation between unfiltered PDO and IPO: {corr_original:.4f}" if corr_original is not None else "Could not calculate correlation for unfiltered data")
print(f"Correlation between filtered PDO and IPO: {corr_filtered:.4f}" if corr_filtered is not None else "Could not calculate correlation for filtered data")

# Save the figure
plt.savefig('PDO_IPO_Timeseries.png', dpi=300, bbox_inches='tight')
plt.show()