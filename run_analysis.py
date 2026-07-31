import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import plotly.graph_objects as go

# ==========================================
# 1. CONFIGURATION (MATCHED TO YOUR PLATE)
# ==========================================
EXCEL_FILE = "tecan_raw.xlsx" # Or change to whatever your file name is

# Sample Columns (Green): Columns 4, 5, 6
SAMPLE_COLS = [4, 5, 6]

# DMSO Control Columns (Blue): Columns 7, 8, 9
DMSO_COLS = [7, 8, 9]

# 11-point dilution series running down Rows A through K (in Molar):
concentrations_molar = np.array([
    3.00e-7,   # Row A: 300 nM
    1.00e-7,   # Row B: 100 nM
    3.33e-8,   # Row C: 33.3 nM
    1.11e-8,   # Row D: 11.1 nM
    3.70e-9,   # Row E: 3.7 nM
    1.23e-9,   # Row F: 1.23 nM
    4.12e-10,  # Row G: 412 pM
    1.37e-10,  # Row H: 137 pM
    4.57e-11,  # Row I: 45.7 pM
    1.52e-11,  # Row J: 15.2 pM
    5.07e-12,  # Row K: 5.07 pM
])

row_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
row_to_conc = {letter: conc for letter, conc in zip(row_letters, concentrations_molar)}

# ==========================================
# 2.      AUTO-DETECT TECAN GRID
# ==========================================
df_raw = pd.read_excel(EXCEL_FILE, header=None)

# Find row where '<>' or row 'A' exists
grid_start_idx = None
for idx, row in df_raw.iterrows():
    vals = [str(v).strip() for v in row.dropna().values]
    if '<>' in vals or 'A' in vals[:2]:
        grid_start_idx = idx
        break

if grid_start_idx is None:
    raise ValueError("Could not locate plate grid in Excel sheet.")

header_row = df_raw.iloc[grid_start_idx]
grid_data = df_raw.iloc[grid_start_idx + 1:].copy()

grid_data.columns = header_row
grid_data.rename(columns={grid_data.columns[0]: 'Row'}, inplace=True)
grid_data['Row'] = grid_data['Row'].astype(str).str.strip()

# ==========================================
# 3. UNPIVOT & NORMALIZE (% RESPONSE)
# ==========================================
df = grid_data.melt(id_vars=['Row'], var_name='Column', value_name='RLU')
df['Column'] = pd.to_numeric(df['Column'], errors='coerce')
df['RLU'] = pd.to_numeric(df['RLU'], errors='coerce')
df = df.dropna(subset=['Column', 'RLU']).copy()
df['Column'] = df['Column'].astype(int)

# Calculate DMSO baseline average from Columns 7, 8, 9
dmso_baseline = df[df['Column'].isin(DMSO_COLS)]['RLU'].mean()
print(f"Calculated DMSO Mean Baseline: {dmso_baseline:.2f} RLU")

# % Response: 0% = DMSO baseline, -100% = complete signal loss
df['RESP'] = ((df['RLU'] - dmso_baseline) / dmso_baseline) * 100

# Filter for Sample wells (Columns 4, 5, 6 across Rows A-K)
sample_df = df[
    (df['Column'].isin(SAMPLE_COLS)) & 
    (df['Row'].isin(row_letters))
].copy()

sample_df['Concentration_M'] = sample_df['Row'].map(row_to_conc)
sample_df['Log_Conc'] = np.log10(sample_df['Concentration_M'])

# ==========================================
# 4. FIT 4PL DOSE-RESPONSE CURVE
# ==========================================
def four_pl(x, top, bottom, log_ic50, hill_slope):
    return bottom + (top - bottom) / (1 + 10 ** ((log_ic50 - x) * hill_slope))

popt, _ = curve_fit(
    four_pl, 
    sample_df['Log_Conc'], 
    sample_df['RESP'], 
    p0=[0, -100, -8.0, -1.0]
)

top_fit, bottom_fit, log_ic50_fit, hill_fit = popt
ic50_nM = (10 ** log_ic50_fit) * 1e9

print("\n================ FIT RESULTS ================")
print(f" Calculated IC50 : {ic50_nM:.3f} nM")
print(f" Top Baseline    : {top_fit:.2f} %")
print(f" Bottom Plateau  : {bottom_fit:.2f} %")
print(f" Hill Slope      : {hill_fit:.2f}")
print("=============================================\n")

# ==========================================
# 5. PLOT & EXPORT
# ==========================================
x_smooth = np.linspace(sample_df['Log_Conc'].min() - 0.2, sample_df['Log_Conc'].max() + 0.2, 200)
y_smooth = four_pl(x_smooth, *popt)

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=sample_df['Log_Conc'],
    y=sample_df['RESP'],
    mode='markers',
    name='T24 HiBiT Replicates',
    marker=dict(color='#2ca02c', size=9, symbol='square')
))

fig.add_trace(go.Scatter(
    x=x_smooth,
    y=y_smooth,
    mode='lines',
    name=f'4PL Fit (IC50 = {ic50_nM:.2f} nM)',
    line=dict(color='#1f77b4', width=2.5)
))

fig.update_layout(
    title=f'T24 HiBiT Knockdown Curve — IC50: {ic50_nM:.2f} nM',
    xaxis_title='Log(Concentration), M',
    yaxis_title='% Response (Relative to DMSO)',
    template='plotly_white'
)

fig.show()

# Save tidy data for Prism
sample_df[['Row', 'Column', 'Concentration_M', 'Log_Conc', 'RLU', 'RESP']].to_csv(
    "cleaned_data_for_prism.csv", index=False
)
print("Saved clean table to 'cleaned_data_for_prism.csv'!")
