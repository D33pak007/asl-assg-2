import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Load AMO data from local file
file_path = r"C:\Users\Lenovo\Downloads\amon.sm.long.data"

# Read the raw text data
with open(file_path, 'r') as f:
    raw_data = f.read()

# Process the file by extracting only the data rows
lines = raw_data.strip().split('\n')
cleaned_lines = []

# Skip header lines (usually contain metadata)
start_parsing = False
for line in lines:
    # Skip empty lines
    if not line.strip():
        continue
    
    # If line starts with a year (4 digits), it's likely a data row
    parts = line.strip().split()
    if parts and len(parts[0]) == 4 and parts[0].isdigit():
        start_parsing = True
        cleaned_lines.append(line)
    elif start_parsing:  # Continue adding lines after we've found the first data row
        cleaned_lines.append(line)

# Manual parsing approach for maximum reliability
amo_data = []
years = []
all_values = []

for line in cleaned_lines:
    parts = line.strip().split()
    if len(parts) >= 1:
        try:
            year = int(parts[0])
            years.append(year)
            
            # Get the monthly values (up to 12 if available)
            monthly_values = []
            for i in range(1, min(13, len(parts))):
                try:
                    val = float(parts[i])
                    # Replace missing values (often coded as -99.99) with NaN
                    if val < -90:
                        val = np.nan
                    monthly_values.append(val)
                except ValueError:
                    monthly_values.append(np.nan)
            
            # Ensure we have 12 months (fill with NaN if needed)
            while len(monthly_values) < 12:
                monthly_values.append(np.nan)
                
            all_values.append(monthly_values)
        except ValueError:
            # If we can't convert the first field to an integer year, skip this line
            pass

# Convert to numpy array
all_values_array = np.array(all_values)

# Flatten the monthly data
amo_values = all_values_array.flatten()

# Create date range starting from the first year in the data
dates = pd.date_range(start=f"{years[0]}-01-01", periods=len(amo_values), freq='MS')
amo_series = pd.Series(amo_values, index=dates)

# Filter the dataset to the required period: 1900-01 to 2023-12
start_date = '1900-01-01'
end_date = '2023-12-31'

amo_filtered = amo_series[(amo_series.index >= start_date) & (amo_series.index <= end_date)]
print(f"Data filtered to {start_date} through {end_date}, with {len(amo_filtered)} data points")

# Apply a low-pass filter (121-month moving average) to see decadal variability
amo_smoothed = amo_filtered.rolling(window=121, center=True).mean()

# Calculate some basic statistics
mean_amo = amo_filtered.mean()
std_amo = amo_filtered.std()
max_amo = amo_filtered.max()
min_amo = amo_filtered.min()

print("\nAMO Index Summary Statistics (1900-2023):")
print(f"Mean: {mean_amo:.4f}")
print(f"Standard Deviation: {std_amo:.4f}")
print(f"Maximum: {max_amo:.4f}")
print(f"Minimum: {min_amo:.4f}")

# Plot the time series
fig, ax = plt.subplots(figsize=(14, 8))

# Original AMO time series
ax.plot(amo_filtered.index, amo_filtered, label='Monthly AMO Index', 
        alpha=0.5, linewidth=0.8, color='darkblue')

# Smoothed AMO time series
ax.plot(amo_smoothed.index, amo_smoothed, label='AMO (10-year moving avg)', 
        linewidth=2.5, color='red')

# Add horizontal line at zero
ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)

# Create numeric date markers for annotations and shading
year_1900 = pd.Timestamp('1900-01-01')
year_1925 = pd.Timestamp('1925-01-01')
year_1965 = pd.Timestamp('1965-01-01')
year_1995 = pd.Timestamp('1995-01-01')
year_2023 = pd.Timestamp('2023-12-31')

year_1912 = pd.Timestamp('1912-01-01')
year_1940 = pd.Timestamp('1940-01-01')
year_1980 = pd.Timestamp('1980-01-01')
year_2010 = pd.Timestamp('2010-01-01')

# Add vertical shading for different periods
ax.axvspan(year_1900, year_1925, alpha=0.2, color='lightblue')
ax.axvspan(year_1925, year_1965, alpha=0.2, color='indianred')
ax.axvspan(year_1965, year_1995, alpha=0.2, color='lightblue')
ax.axvspan(year_1995, year_2023, alpha=0.2, color='indianred')

# Add annotations using specific x-coordinates
ax.annotate('Cool Phase', xy=(mdates.date2num(year_1912), 0.4), xytext=(mdates.date2num(year_1912), 0.4),
            fontsize=12)
ax.annotate('Warm Phase', xy=(mdates.date2num(year_1940), 0.4), xytext=(mdates.date2num(year_1940), 0.4),
            fontsize=12)
ax.annotate('Cool Phase', xy=(mdates.date2num(year_1980), 0.4), xytext=(mdates.date2num(year_1980), 0.4),
            fontsize=12)
ax.annotate('Warm Phase', xy=(mdates.date2num(year_2010), 0.4), xytext=(mdates.date2num(year_2010), 0.4),
            fontsize=12)

# Add title and labels
ax.set_title('Atlantic Multi-decadal Oscillation (AMO) Index: 1900-2023', fontsize=16)
ax.set_xlabel('Year', fontsize=14)
ax.set_ylabel('AMO Index Value', fontsize=14)

# Add grid and legend
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend(fontsize=12)

# Format x-axis to show years
ax.xaxis.set_major_locator(mdates.YearLocator(10))  # Show every 10 years
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Format as year
plt.xticks(rotation=45)

# Add statistics as text box
stats_text = (f"Mean: {mean_amo:.3f}\nStd Dev: {std_amo:.3f}\n"
              f"Max: {max_amo:.3f}\nMin: {min_amo:.3f}")
props = dict(boxstyle='round', facecolor='white', alpha=0.8)
ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=12,
        verticalalignment='top', bbox=props)

fig.tight_layout()
plt.savefig('amo_timeseries.png', dpi=300)
plt.show()

# Create annual means for clearer visualization of the overall trend
amo_annual = amo_filtered.resample('YE').mean()

# Plot annual AMO index
fig, ax = plt.subplots(figsize=(14, 6))
bars = ax.bar(amo_annual.index, amo_annual.values, width=365, 
        color=np.where(amo_annual.values >= 0, 'darkred', 'steelblue'))
ax.axhline(y=0, color='black', linestyle='-', alpha=0.7)
ax.set_title('Annual Atlantic Multi-decadal Oscillation (AMO) Index: 1900-2023', fontsize=16)
ax.set_xlabel('Year', fontsize=14)
ax.set_ylabel('AMO Index Value', fontsize=14)
ax.grid(True, linestyle='--', alpha=0.7)

# Format x-axis to show years
ax.xaxis.set_major_locator(mdates.YearLocator(10))  # Show every 10 years
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Format as year
plt.xticks(rotation=45)

fig.tight_layout()
plt.savefig('amo_annual.png', dpi=300)
plt.show()