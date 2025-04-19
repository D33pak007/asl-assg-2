import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import urllib.request
from eofs.xarray import Eof
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.stats import pearsonr

# === Load SST and preprocess ===
def load_and_preprocess_sst():
    sst_data = xr.open_dataset(r'C:\Users\Lenovo\Downloads\monnthly sea surface temp.nc')
    sst = sst_data['SST'].rename({'TIME': 'time', 'LAT': 'lat', 'LON': 'lon'})
    sst = sst.sel(time=slice("1950-01", "2023-12"))

    # Indian Ocean region
    sst_io = sst.sel(lon=slice(40, 120), lat=slice(-30, 30))

    # Monthly anomalies
    climatology = sst_io.groupby('time.month').mean('time')
    anomalies = sst_io.groupby('time.month') - climatology

    # Weights
    weights = np.sqrt(np.cos(np.deg2rad(anomalies.lat)))
    return anomalies, weights

# === EOF Analysis ===
def perform_eof_analysis(data, weights):
    solver = Eof(data * weights)
    eofs = solver.eofs(neofs=2)
    pcs = solver.pcs(npcs=2)
    var_frac = solver.varianceFraction(neigs=2)
    return eofs, pcs, var_frac

# === Plot EOFs & PCs ===
def plot_eofs_and_pcs(eofs, pcs, var_frac):
    fig = plt.figure(figsize=(15, 12))
    ax1 = plt.subplot(2, 1, 1, projection=ccrs.PlateCarree())
    eofs[0].plot(ax=ax1, transform=ccrs.PlateCarree(), cmap='RdBu_r')
    ax1.add_feature(cfeature.COASTLINE)
    ax1.set_title(f'EOF1 ({var_frac[0]*100:.1f}% variance explained)')

    ax2 = plt.subplot(2, 1, 2, projection=ccrs.PlateCarree())
    eofs[1].plot(ax=ax2, transform=ccrs.PlateCarree(), cmap='RdBu_r')
    ax2.add_feature(cfeature.COASTLINE)
    ax2.set_title(f'EOF2 ({var_frac[1]*100:.1f}% variance explained)')
    plt.tight_layout()
    plt.show()

    # PC Time Series
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6))
    ax1.plot(pcs.time, pcs[:, 0], 'b-')
    ax1.set_title('PC1 Time Series')
    ax1.grid(True)
    ax2.plot(pcs.time, pcs[:, 1], 'r-')
    ax2.set_title('PC2 Time Series')
    ax2.grid(True)
    plt.tight_layout()
    plt.show()

# === Save PCs as DataFrame ===
def save_pc_timeseries(pcs):
    pc_df = pd.DataFrame({
        'PC1': pcs[:, 0].values,
        'PC2': pcs[:, 1].values
    }, index=pd.to_datetime(pcs.time.values))
    pc_df.to_csv('pcs.csv')
    return pc_df

# === Read NOAA indices ===
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
    df = pd.DataFrame({
        'Year': years,
        'Month': months,
        'Value': values
    })
    df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'].astype(str))
    return df.set_index("Date")["Value"]

# === Main function ===
def main():
    print("Loading SST and computing anomalies...")
    sst_anom, weights = load_and_preprocess_sst()

    print("Performing EOF analysis...")
    eofs, pcs, var_frac = perform_eof_analysis(sst_anom, weights)

    print("Plotting EOF patterns and PCs...")
    plot_eofs_and_pcs(eofs, pcs, var_frac)

    print("Saving PC time series...")
    pc_df = save_pc_timeseries(pcs)

    print(f"\nEOF Variance Explained:\nEOF1: {var_frac[0]*100:.2f}%, EOF2: {var_frac[1]*100:.2f}%")
    return eofs, pcs, var_frac, pc_df

# === Run everything ===
if __name__ == "__main__":
    eofs, pcs, var_frac, pc_df = main()

    # === Load Nino3.4 and DMI ===
    nino34_url = "https://psl.noaa.gov/data/timeseries/month/data/nino34.long.anom.data"
    dmi_url = "https://psl.noaa.gov/gcos_wgsp/Timeseries/Data/dmi.had.long.data"

    nino34_filtered = read_psl_index(nino34_url)["1950-01":"2023-12"]
    dmi_filtered = read_psl_index(dmi_url)["1950-01":"2023-12"]

    # Align time index
    nino34_filtered.index = nino34_filtered.index.to_period("M").to_timestamp()
    dmi_filtered.index = dmi_filtered.index.to_period("M").to_timestamp()
    pc_df.index = pc_df.index.to_period("M").to_timestamp()

    # Combine all time series
    combined = pd.concat([
        nino34_filtered.rename("Nino3.4"),
        dmi_filtered.rename("DMI"),
        pc_df
    ], axis=1).dropna()

    # === Q9: Correlation Results ===
    print("\n📊 Correlation Matrix:")
    print(combined.corr())

    print(f"\n🔁 Correlation between Nino3.4 and DMI: {pearsonr(combined['Nino3.4'], combined['DMI'])[0]:.3f}")
    print(f"🔁 Correlation between Nino3.4 and EOF1 (PC1): {pearsonr(combined['Nino3.4'], combined['PC1'])[0]:.3f}")
    print(f"🔁 Correlation between Nino3.4 and EOF2 (PC2): {pearsonr(combined['Nino3.4'], combined['PC2'])[0]:.3f}")
    print(f"🔁 Correlation between DMI and EOF1 (PC1): {pearsonr(combined['DMI'], combined['PC1'])[0]:.3f}")
    print(f"🔁 Correlation between DMI and EOF2 (PC2): {pearsonr(combined['DMI'], combined['PC2'])[0]:.3f}")
