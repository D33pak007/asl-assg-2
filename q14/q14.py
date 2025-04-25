import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import requests
from io import StringIO
from scipy import stats

# Function to download data from URLs
def download_data(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        print(f"Failed to download data from {url}")
        return None

# Function to properly parse climate indices data based on the AMO parser approach
def parse_climate_data(raw_data):
    lines = raw_data.strip().split('\n')
    cleaned_lines = []
    
    # Skip header lines
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
                pass
    
    # Convert to numpy array
    all_values_array = np.array(all_values)
    
    # Flatten the monthly data
    values = all_values_array.flatten()
    
    # Create date range starting from the first year in the data
    dates = pd.date_range(start=f"{years[0]}-01-01", periods=len(values), freq='MS')
    series = pd.Series(values, index=dates)
    
    return series

# Download the datasets
print("Downloading data...")
nino34_url = "https://psl.noaa.gov/data/timeseries/month/data/nino34.long.anom.data"
ipo_url = "https://psl.noaa.gov/data/timeseries/IPOTPI/tpi.timeseries.hadisst11.data"
amo_url = "https://psl.noaa.gov/data/correlation/amon.sm.long.data"

nino34_data = download_data(nino34_url)
ipo_data = download_data(ipo_url)
amo_data = download_data(amo_url)

# Process each dataset using the proper parsing method
print("Processing climate indices data...")
nino34_series = parse_climate_data(nino34_data)
ipo_series = parse_climate_data(ipo_data)
amo_series = parse_climate_data(amo_data)

# Filter each dataset to the required period: 1900-01 to 2023-12
start_date = '1900-01-01'
end_date = '2023-12-31'

nino34_filtered = nino34_series[(nino34_series.index >= start_date) & (nino34_series.index <= end_date)]
ipo_filtered = ipo_series[(ipo_series.index >= start_date) & (ipo_series.index <= end_date)]
amo_filtered = amo_series[(amo_series.index >= start_date) & (amo_series.index <= end_date)]

print(f"Data filtered to {start_date} through {end_date}")
print(f"NINO3.4 data points: {len(nino34_filtered)}")
print(f"IPO data points: {len(ipo_filtered)}")
print(f"AMO data points: {len(amo_filtered)}")

# Create a DataFrame with common dates
df = pd.DataFrame({
    'NINO3.4': nino34_filtered,
    'IPO': ipo_filtered,
    'AMO': amo_filtered
})

# Apply a low-pass filter (121-month moving average) to see decadal variability
print("Applying 10-year filtering...")
window = 121
df['NINO3.4_filtered'] = df['NINO3.4'].rolling(window=window, center=True).mean()
df['IPO_filtered'] = df['IPO'].rolling(window=window, center=True).mean()
df['AMO_filtered'] = df['AMO'].rolling(window=window, center=True).mean()

# Calculate correlations
print("Calculating correlations...")
corr_amo_enso = df['AMO'].corr(df['NINO3.4'])
corr_amo_ipo = df['AMO'].corr(df['IPO'])
corr_amo_enso_filtered = df['AMO_filtered'].dropna().corr(df['NINO3.4_filtered'].dropna())
corr_amo_ipo_filtered = df['AMO_filtered'].dropna().corr(df['IPO_filtered'].dropna())

print(f"Correlation between AMO and ENSO: {corr_amo_enso:.4f}")
print(f"Correlation between AMO and IPO: {corr_amo_ipo:.4f}")
print(f"Correlation between AMO and ENSO (10-year filtered): {corr_amo_enso_filtered:.4f}")
print(f"Correlation between AMO and IPO (10-year filtered): {corr_amo_ipo_filtered:.4f}")

# Calculate recent period (1950-2020) correlations
df_recent = df.loc['1950-01-01':'2020-12-31'].copy()
recent_corr_amo_enso = df_recent['AMO'].corr(df_recent['NINO3.4'])
recent_corr_amo_ipo = df_recent['AMO'].corr(df_recent['IPO'])
recent_corr_amo_enso_filtered = df_recent['AMO_filtered'].dropna().corr(df_recent['NINO3.4_filtered'].dropna())
recent_corr_amo_ipo_filtered = df_recent['IPO_filtered'].dropna().corr(df_recent['AMO_filtered'].dropna())

print(f"Recent correlation between AMO and ENSO (1950-2020): {recent_corr_amo_enso:.4f}")
print(f"Recent correlation between AMO and IPO (1950-2020): {recent_corr_amo_ipo:.4f}")
print(f"Recent filtered correlation between AMO and ENSO (1950-2020): {recent_corr_amo_enso_filtered:.4f}")
print(f"Recent filtered correlation between AMO and IPO (1950-2020): {recent_corr_amo_ipo_filtered:.4f}")

# Create plots
print("Creating plots...")
plt.figure(figsize=(14, 10))

# Plot original time series
plt.subplot(3, 1, 1)
plt.plot(df.index, df['NINO3.4'], 'b-', alpha=0.5, linewidth=0.8, label='NINO3.4')
plt.plot(df.index, df['AMO'], 'r-', alpha=0.5, linewidth=0.8, label='AMO')
plt.title(f'ENSO (NINO3.4) vs AMO (1900-2023, Correlation: {corr_amo_enso:.4f})')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.ylabel('Anomaly (°C)')
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)

# Format x-axis
plt.gca().xaxis.set_major_locator(mdates.YearLocator(20))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.subplot(3, 1, 2)
plt.plot(df.index, df['IPO'], 'g-', alpha=0.5, linewidth=0.8, label='IPO')
plt.plot(df.index, df['AMO'], 'r-', alpha=0.5, linewidth=0.8, label='AMO')
plt.title(f'IPO vs AMO (1900-2023, Correlation: {corr_amo_ipo:.4f})')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.ylabel('Anomaly (°C)')
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)

# Format x-axis
plt.gca().xaxis.set_major_locator(mdates.YearLocator(20))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# Plot filtered time series
plt.subplot(3, 1, 3)
plt.plot(df.index, df['NINO3.4_filtered'], 'b-', linewidth=2.5, label='NINO3.4 (10-yr filtered)')
plt.plot(df.index, df['IPO_filtered'], 'g-', linewidth=2.5, label='IPO (10-yr filtered)')
plt.plot(df.index, df['AMO_filtered'], 'r-', linewidth=2.5, label='AMO (10-yr filtered)')
plt.title(f'10-year Filtered Time Series\nAMO-ENSO Corr: {corr_amo_enso_filtered:.4f}, AMO-IPO Corr: {corr_amo_ipo_filtered:.4f}')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.ylabel('Anomaly (°C)')
plt.xlabel('Year')
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)

# Format x-axis
plt.gca().xaxis.set_major_locator(mdates.YearLocator(20))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('amo_enso_ipo_correlations.png', dpi=300)
print("Time series plot saved as 'amo_enso_ipo_correlations.png'")

# Create scatter plots
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.scatter(df['AMO'], df['NINO3.4'], alpha=0.5, color='blue')
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlabel('AMO Index')
plt.ylabel('NINO3.4 Index')
plt.title(f'AMO vs NINO3.4 (r = {corr_amo_enso:.4f})')

# Add regression line
x = df['AMO'].values
y = df['NINO3.4'].values
mask = ~np.isnan(x) & ~np.isnan(y)
x = x[mask]
y = y[mask]
m, b = np.polyfit(x, y, 1)
plt.plot(x, m*x + b, 'r-', linewidth=2)

plt.subplot(1, 2, 2)
plt.scatter(df['AMO'], df['IPO'], alpha=0.5, color='green')
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlabel('AMO Index')
plt.ylabel('IPO Index')
plt.title(f'AMO vs IPO (r = {corr_amo_ipo:.4f})')

# Add regression line
x = df['AMO'].values
y = df['IPO'].values
mask = ~np.isnan(x) & ~np.isnan(y)
x = x[mask]
y = y[mask]
m, b = np.polyfit(x, y, 1)
plt.plot(x, m*x + b, 'r-', linewidth=2)

plt.tight_layout()
plt.savefig('amo_enso_ipo_scatter.png', dpi=300)
print("Scatter plot saved as 'amo_enso_ipo_scatter.png'")

# Similar to the AMO visualization in the first script, create annual plots for better visualization
# Create annual means
nino34_annual = df['NINO3.4'].resample('YE').mean()
ipo_annual = df['IPO'].resample('YE').mean()
amo_annual = df['AMO'].resample('YE').mean()

# Plot annual indices with color-coded bars
fig, axs = plt.subplots(3, 1, figsize=(14, 12))

# NINO3.4 Annual
axs[0].bar(nino34_annual.index, nino34_annual.values, width=365, 
          color=np.where(nino34_annual.values >= 0, 'darkred', 'steelblue'))
axs[0].axhline(y=0, color='black', linestyle='-', alpha=0.7)
axs[0].set_title('Annual NINO3.4 Index: 1900-2023', fontsize=16)
axs[0].set_ylabel('NINO3.4 Index Value', fontsize=14)
axs[0].grid(True, linestyle='--', alpha=0.7)
axs[0].xaxis.set_major_locator(mdates.YearLocator(10))
axs[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.setp(axs[0].get_xticklabels(), rotation=45)

# IPO Annual
axs[1].bar(ipo_annual.index, ipo_annual.values, width=365, 
          color=np.where(ipo_annual.values >= 0, 'darkgreen', 'lightgreen'))
axs[1].axhline(y=0, color='black', linestyle='-', alpha=0.7)
axs[1].set_title('Annual IPO Index: 1900-2023', fontsize=16)
axs[1].set_ylabel('IPO Index Value', fontsize=14)
axs[1].grid(True, linestyle='--', alpha=0.7)
axs[1].xaxis.set_major_locator(mdates.YearLocator(10))
axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.setp(axs[1].get_xticklabels(), rotation=45)

# AMO Annual
axs[2].bar(amo_annual.index, amo_annual.values, width=365, 
          color=np.where(amo_annual.values >= 0, 'darkred', 'steelblue'))
axs[2].axhline(y=0, color='black', linestyle='-', alpha=0.7)
axs[2].set_title('Annual AMO Index: 1900-2023', fontsize=16)
axs[2].set_xlabel('Year', fontsize=14)
axs[2].set_ylabel('AMO Index Value', fontsize=14)
axs[2].grid(True, linestyle='--', alpha=0.7)
axs[2].xaxis.set_major_locator(mdates.YearLocator(10))
axs[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.setp(axs[2].get_xticklabels(), rotation=45)

fig.tight_layout()
plt.savefig('climate_indices_annual.png', dpi=300)
print("Annual indices plot saved as 'climate_indices_annual.png'")

# Add vertical shading for AMO phases (as in the first script)
fig, ax = plt.subplots(figsize=(14, 8))

# Plot AMO time series
ax.plot(amo_filtered.index, amo_filtered, label='Monthly AMO Index', 
        alpha=0.5, linewidth=0.8, color='darkblue')

# Smoothed AMO time series
amo_smoothed = amo_filtered.rolling(window=121, center=True).mean()
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
mean_amo = amo_filtered.mean()
std_amo = amo_filtered.std()
max_amo = amo_filtered.max()
min_amo = amo_filtered.min()

stats_text = (f"Mean: {mean_amo:.3f}\nStd Dev: {std_amo:.3f}\n"
              f"Max: {max_amo:.3f}\nMin: {min_amo:.3f}")
props = dict(boxstyle='round', facecolor='white', alpha=0.8)
ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=12,
        verticalalignment='top', bbox=props)

fig.tight_layout()
plt.savefig('amo_timeseries_phases.png', dpi=300)
print("AMO time series with phases plot saved as 'amo_timeseries_phases.png'")

# Provide physical interpretation
print("\nPhysical Interpretation of Correlations:")
print("----------------------------------------")
print("1. AMO and ENSO Correlation:")
if corr_amo_enso < 0:
    print(f"   The negative correlation ({corr_amo_enso:.4f}) suggests an inverse relationship between Atlantic and Pacific variability.")
    print("   When the North Atlantic is anomalously warm (positive AMO), there is an increased tendency for")
    print("   La Niña-like conditions in the tropical Pacific (negative ENSO values).")
    print("   This basin-scale seesaw pattern operates through atmospheric teleconnections:")
    print("   - Changes in the Walker circulation that affect Pacific Ocean upwelling")
    print("   - Shifts in the Intertropical Convergence Zone (ITCZ) position")
    print("   - Modifications to global wind patterns that can enhance or suppress ENSO development")
else:
    print(f"   The positive correlation ({corr_amo_enso:.4f}) suggests some connection between Atlantic and Pacific warming/cooling cycles,")
    print("   potentially indicating a common external forcing mechanism affecting both ocean basins similarly.")

print("\n2. AMO and IPO Correlation:")
if corr_amo_ipo < 0:
    print(f"   The negative correlation ({corr_amo_ipo:.4f}) indicates that decadal variability in the")
    print("   Atlantic and Pacific often operates in opposition, with warm phases in one basin")
    print("   coinciding with cool phases in the other. This interbasin interaction suggests:")
    print("   - A thermohaline circulation influence connecting both ocean basins")
    print("   - Modulation of the Walker circulation across the tropical oceans")
    print("   - Global-scale redistribution of heat energy between ocean basins")
    print("   - Cross-basin influences through atmospheric bridges")
    print("   This relationship has significant implications for multi-decadal climate prediction.")
else:
    print(f"   The positive correlation ({corr_amo_ipo:.4f}) suggests coordinated warming/cooling patterns")
    print("   between Atlantic and Pacific basins at decadal timescales, which may indicate:")
    print("   - Common external forcing mechanisms like greenhouse gases or solar activity")
    print("   - Global-scale circulation patterns affecting multiple ocean basins simultaneously")
    print("   - Potential positive feedbacks between ocean basins reinforcing temperature anomalies")

print("\n3. Filtered Correlations (Decadal Timescale):")
print(f"   AMO-ENSO filtered correlation: {corr_amo_enso_filtered:.4f}")
print(f"   AMO-IPO filtered correlation: {corr_amo_ipo_filtered:.4f}")
if abs(corr_amo_enso_filtered) > abs(corr_amo_enso) or abs(corr_amo_ipo_filtered) > abs(corr_amo_ipo):
    print("   The stronger correlations in the filtered data suggest that the relationships between")
    print("   these climate modes are more pronounced at decadal timescales than at interannual timescales.")
    print("   This highlights the importance of low-frequency climate variability in interbasin interactions,")
    print("   potentially through slower oceanic processes like the meridional overturning circulation.")
else:
    print("   The correlations at decadal timescales reveal the long-term relationships between these")
    print("   climate modes that may be masked by higher-frequency variability in the unfiltered data.")