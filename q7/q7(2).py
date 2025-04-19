import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import urllib.request
from scipy.stats import pearsonr

# === STEP 1: Load SST Data ===
sst = xr.open_dataset(r"C:\Users\Lenovo\Downloads\monnthly sea surface temp.nc")

# Rename dimensions to standard names
sst = sst.rename({'TIME': 'time', 'LAT': 'lat', 'LON': 'lon'})

# Select time range
sst = sst.sel(time=slice("1950-01", "2023-12"))

# Convert longitudes to 0–360
sst = sst.assign_coords(lon=(sst.lon % 360))

# === STEP 2: Region-wise Mean Helper ===
def area_mean(ds, lat_bounds, lon_bounds):
    subset = ds.sel(lat=slice(*sorted(lat_bounds)), lon=slice(*lon_bounds))
    weights = np.cos(np.deg2rad(subset.lat))
    return subset.weighted(weights).mean(dim=["lat", "lon"])

# === STEP 3: Calculate Indices ===
nino34 = area_mean(sst['SST'], (-5, 5), (190.5, 240.5))
ceio = area_mean(sst['SST'], (-5, 5), (70.5, 90.5))
sumatra = area_mean(sst['SST'], (-13, -3), (100.5, 110.5))
dmi = ceio - sumatra

# === STEP 4: Compute Monthly Anomalies ===
def monthly_anomaly(da):
    climatology = da.groupby("time.month").mean("time")
    return da.groupby("time.month") - climatology

nino34_anom = monthly_anomaly(nino34)
dmi_anom = monthly_anomaly(dmi)

# === STEP 5: Load Official Q8 Data from NOAA ===
def read_psl_index(url):
    with urllib.request.urlopen(url) as response:
        data = response.read().decode('utf-8')
    lines = data.strip().split('\n')
    years, values = [], []
    for line in lines:
        if line.strip() and not line.startswith('#'):
            parts = line.split()
            if len(parts) >= 13:
                year = int(parts[0])
                monthly = [float(val) for val in parts[1:13]]
                years.extend([year] * 12)
                values.extend(monthly)
    months = np.tile(np.arange(1, 13), len(years) // 12)
    df = pd.DataFrame({"Year": years, "Month": months, "Value": values})
    df["Date"] = pd.to_datetime(df["Year"].astype(str) + "-" + df["Month"].astype(str))
    return df.set_index("Date")["Value"]

nino34_q8 = read_psl_index("https://psl.noaa.gov/data/timeseries/month/data/nino34.long.anom.data")
dmi_q8 = read_psl_index("https://psl.noaa.gov/gcos_wgsp/Timeseries/Data/dmi.had.long.data")

# Filter to match time range
nino34_q8 = nino34_q8["1950-01":"2023-12"]
dmi_q8 = dmi_q8["1950-01":"2023-12"]

# === STEP 6: Compare & Correlate ===
# Convert to pandas series
nino34_own = nino34_anom.to_series().rename("Nino3.4_own")
dmi_own = dmi_anom.to_series().rename("DMI_own")

# Normalize time index to monthly (important!)
nino34_own.index = nino34_own.index.to_period("M").to_timestamp()
dmi_own.index = dmi_own.index.to_period("M").to_timestamp()
nino34_q8.index = nino34_q8.index.to_period("M").to_timestamp()
dmi_q8.index = dmi_q8.index.to_period("M").to_timestamp()

# Debug time ranges
print("🌐 Your Nino3.4 index range:", nino34_own.index.min(), "to", nino34_own.index.max())
print("📊 NOAA Nino3.4 Q8 range:", nino34_q8.index.min(), "to", nino34_q8.index.max())
print("🌐 Your DMI index range:", dmi_own.index.min(), "to", dmi_own.index.max())
print("📊 NOAA DMI Q8 range:", dmi_q8.index.min(), "to", dmi_q8.index.max())

# Combine and drop NaNs
nino_compare = pd.concat([nino34_own, nino34_q8.rename("Nino3.4_Q8")], axis=1).dropna()
dmi_compare = pd.concat([dmi_own, dmi_q8.rename("DMI_Q8")], axis=1).dropna()

# Debug lengths
print("✅ Nino Compare Length:", len(nino_compare))
print(nino_compare.head())
print("✅ DMI Compare Length:", len(dmi_compare))
print(dmi_compare.head())

# Correlations only if valid
if len(nino_compare) >= 2:
    corr_nino = pearsonr(nino_compare["Nino3.4_own"], nino_compare["Nino3.4_Q8"])[0]
    print(f"🔁 Correlation between your Nino3.4 and Q8: {corr_nino:.3f}")
else:
    print("❌ Not enough data for Nino3.4 correlation")

if len(dmi_compare) >= 2:
    corr_dmi = pearsonr(dmi_compare["DMI_own"], dmi_compare["DMI_Q8"])[0]
    print(f"🔁 Correlation between your new DMI and Q8: {corr_dmi:.3f}")
else:
    print("❌ Not enough data for DMI correlation")

# === STEP 7: Plot ===
plt.figure(figsize=(14, 6))

plt.subplot(2, 1, 1)
plt.plot(nino_compare.index, nino_compare["Nino3.4_own"], label="Your Nino3.4", color="blue")
plt.plot(nino_compare.index, nino_compare["Nino3.4_Q8"], label="Q8 Nino3.4", color="red", alpha=0.6)
plt.title("Nino3.4 Index Comparison (1950–2023)")
plt.ylabel("SST Anomaly (°C)")
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(dmi_compare.index, dmi_compare["DMI_own"], label="Your DMI (New Definition)", color="blue")
plt.plot(dmi_compare.index, dmi_compare["DMI_Q8"], label="Q8 DMI (Original)", color="red", alpha=0.6)
plt.title("DMI Index Comparison (1950–2023)")
plt.ylabel("SST Anomaly Difference (°C)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
q7_df = pd.concat([nino34_own, dmi_own], axis=1)
q7_df.columns = ["Nino3.4", "DMI"]
q7_df.index.name = "Date"

# Save to CSV
q7_df.to_csv("Q7_nino_dmi_indices.csv")

print("💾 Saved custom Nino3.4 and DMI anomalies to 'Q7_nino_dmi_indices.csv'")
