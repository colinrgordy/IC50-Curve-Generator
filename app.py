import io
import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import curve_fit
import streamlit as st

st.set_page_config(
    page_title="High-Throughput Dose Response Analyzer",
    page_icon="",
    layout="wide",
)

# ==========================================
# CUSTOM CSS FOR ENLARGED SIDEBAR & METRICS
# ==========================================
st.markdown(
    """
<style>
    /* 1. Sidebar Typography (Labels, Inputs, Radio Options, Headers) */
    [data-testid="stSidebar"] label, 
    [data-testid="stSidebar"] .stWidgetLabel,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span {
        font-size: 20px !important;
        font-weight: 600 !important;
    }
    
    [data-testid="stSidebar"] input {
        font-size: 20px !important;
        font-weight: 600 !important;
    }

    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        font-size: 24px !important;
        font-weight: 700 !important;
    }

    /* 2. Metric Cards (Side Statistics) */
    [data-testid="stMetricValue"] {
        font-size: 34px !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 22px !important;
        font-weight: 600 !important;
        color: #cccccc !important;
    }
    
    /* 3. Global Headers */
    h1 { font-size: 38px !important; font-weight: 800 !important; }
    h2 { font-size: 30px !important; font-weight: 700 !important; }
    h3 { font-size: 26px !important; font-weight: 700 !important; }
    
    /* 4. Export Button */
    .stDownloadButton button {
        font-size: 22px !important;
        font-weight: 700 !important;
        padding: 16px 32px !important;
        border-radius: 8px !important;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ==========================================
# 1. PARSING ENGINES (TECAN, VIEWLUX, CSV)
# ==========================================


def parse_tecan_xlsx(file_bytes):
    df_raw = pd.read_excel(file_bytes, header=None)
    grid_start_idx = None
    for idx, row in df_raw.iterrows():
        vals = [str(v).strip() for v in row.dropna().values]
        if "<>" in vals or "A" in vals[:2]:
            grid_start_idx = idx
            break

    if grid_start_idx is None:
        return {}

    header_row = df_raw.iloc[grid_start_idx]
    grid_data = df_raw.iloc[grid_start_idx + 1 :].copy()
    grid_data.columns = header_row
    grid_data.rename(columns={grid_data.columns[0]: "Row"}, inplace=True)
    grid_data["Row"] = grid_data["Row"].astype(str).str.strip()

    valid_rows = [chr(i) for i in range(ord("A"), ord("Q"))]
    grid_data = grid_data[grid_data["Row"].isin(valid_rows)]

    df = grid_data.melt(id_vars=["Row"], var_name="Column", value_name="RLU")
    df["Column"] = pd.to_numeric(df["Column"], errors="coerce")
    df["RLU"] = pd.to_numeric(df["RLU"], errors="coerce")
    df = df.dropna(subset=["Column", "RLU"])
    df["Column"] = df["Column"].astype(int)

    return {"Plate_1": df}


def parse_viewlux_txt(file_bytes):
    content = file_bytes.read().decode("utf-8", errors="ignore")
    plates = {}
    blocks = re.split(r"(?:Plate|PLATE)\s*(\d+)", content)

    if len(blocks) > 1:
        for i in range(1, len(blocks), 2):
            plate_num = f"Plate_{blocks[i]}"
            block_text = blocks[i + 1]
            df_grid = pd.read_csv(
                io.StringIO(block_text), sep="\t", header=None
            )
            df_grid = df_grid.dropna(how="all").dropna(how="all", axis=1)
            plates[plate_num] = format_grid_to_long(df_grid)
    else:
        df_grid = pd.read_csv(io.StringIO(content), sep="\t", header=None)
        plates["Plate_1"] = format_grid_to_long(df_grid)

    return plates


def format_grid_to_long(df_grid):
    df_grid = df_grid.dropna(how="all").reset_index(drop=True)
    if df_grid.iloc[0, 0] in ["<>", "Row", None] or str(
        df_grid.iloc[0, 0]
    ).isdigit():
        df_grid.columns = df_grid.iloc[0]
        df_grid = df_grid.iloc[1:].reset_index(drop=True)

    df_grid.rename(columns={df_grid.columns[0]: "Row"}, inplace=True)
    df_grid["Row"] = df_grid["Row"].astype(str).str.strip()

    valid_rows = [chr(i) for i in range(ord("A"), ord("Q"))]
    df_grid = df_grid[df_grid["Row"].isin(valid_rows)]

    df = df_grid.melt(id_vars=["Row"], var_name="Column", value_name="RLU")
    df["Column"] = pd.to_numeric(df["Column"], errors="coerce")
    df["RLU"] = pd.to_numeric(df["RLU"], errors="coerce")
    return df.dropna(subset=["Column", "RLU"])


# ==========================================
# 2. 4PL MATH FUNCTION
# ==========================================


def four_pl(x, top, bottom, log_ic50, hill_slope):
    return bottom + (top - bottom) / (1 + 10 ** ((log_ic50 - x) * hill_slope))


# ==========================================
# 3. STREAMLIT INTERFACE
# ==========================================

st.title("High-Throughput Dose Response & IC50 Analyzer")

# Sidebar Configuration
st.sidebar.header("1. Upload Data File")
uploaded_file = st.sidebar.file_uploader(
    "Choose file", type=["xlsx", "txt", "csv"]
)

if uploaded_file is not None:
    file_name = uploaded_file.name
    if file_name.endswith(".xlsx"):
        plates_dict = parse_tecan_xlsx(uploaded_file)
    elif file_name.endswith(".txt"):
        plates_dict = parse_viewlux_txt(uploaded_file)
    else:
        df_csv = pd.read_csv(uploaded_file)
        plates_dict = {"Plate_1": df_csv}

    if not plates_dict:
        st.error(
            "Could not parse plate data. Please check the file format."
        )
        st.stop()

    st.sidebar.success(f"Loaded {len(plates_dict)} plate(s)!")
    selected_plate = st.sidebar.selectbox(
        "Select Plate to Analyze", list(plates_dict.keys())
    )
    df_plate = plates_dict[selected_plate]

    st.sidebar.header("2. Plate Layout Settings")

    dilution_dir = st.sidebar.radio(
        "Dilution Direction", ["Vertical (Rows A-K)", "Horizontal (Cols 1-12)"]
    )

    st.sidebar.subheader("Concentration Parameters")
    top_dose_nM = st.sidebar.number_input(
        "Top Concentration (nM)", value=300.0, step=10.0
    )
    dilution_fold = st.sidebar.number_input(
        "Dilution Fold (e.g., 3 for 1:3)", value=3.0, step=0.5
    )
    num_points = st.sidebar.slider(
        "Number of Dilution Points", min_value=4, max_value=16, value=11
    )

    concentrations_M = [
        (top_dose_nM * 1e-9) / (dilution_fold**i)
        for i in range(num_points)
    ]

    st.sidebar.subheader("Control & Sample Well Groups")
    sample_cols = st.sidebar.multiselect(
        "Sample Columns (Compound Treated)",
        options=list(range(1, 25)),
        default=[4, 5, 6],
    )
    dmso_cols = st.sidebar.multiselect(
        "Negative Control Columns (DMSO Baseline / 0%)",
        options=list(range(1, 25)),
        default=[7, 8, 9],
    )
    wt_cols = st.sidebar.multiselect(
        "Background / WT Columns (Optional)",
        options=list(range(1, 25)),
        default=[1, 2, 3],
    )

    # ==========================================
    # 4. NORMALIZATION & DATA PROCESSING
    # ==========================================
    dmso_baseline = df_plate[df_plate["Column"].isin(dmso_cols)]["RLU"].mean()

    df_plate["RESP"] = (
        (df_plate["RLU"] - dmso_baseline) / dmso_baseline
    ) * 100

    if "Vertical" in dilution_dir:
        row_letters = [chr(ord("A") + i) for i in range(num_points)]
        row_to_conc = dict(zip(row_letters, concentrations_M))
        sample_df = df_plate[
            (df_plate["Column"].isin(sample_cols))
            & (df_plate["Row"].isin(row_letters))
        ].copy()
        sample_df["Concentration_M"] = sample_df["Row"].map(row_to_conc)
    else:
        col_numbers = list(range(1, num_points + 1))
        col_to_conc = dict(zip(col_numbers, concentrations_M))
        sample_df = df_plate[
            (df_plate["Column"].isin(sample_cols))
            & (df_plate["Column"].isin(col_numbers))
        ].copy()
        sample_df["Concentration_M"] = sample_df["Column"].map(col_to_conc)

    sample_df["Log_Conc"] = np.log10(sample_df["Concentration_M"])

    # ==========================================
    # 5. CURVE FITTING & PLOTLY GRAPH
    # ==========================================
    try:
        popt, _ = curve_fit(
            four_pl,
            sample_df["Log_Conc"],
            sample_df["RESP"],
            p0=[0, -100, -8.0, -1.0],
        )
        top_fit, bottom_fit, log_ic50_fit, hill_fit = popt
        ic50_M = 10**log_ic50_fit
        ic50_nM = ic50_M * 1e9

        ic50_y_val = bottom_fit + (top_fit - bottom_fit) / 2.0

        x_smooth = np.linspace(
            sample_df["Log_Conc"].min() - 0.2,
            sample_df["Log_Conc"].max() + 0.2,
            300,
        )
        y_smooth = four_pl(x_smooth, *popt)

        fig = go.Figure()

        # Replicates
        fig.add_trace(
            go.Scatter(
                x=sample_df["Log_Conc"],
                y=sample_df["RESP"],
                mode="markers",
                name="Replicates",
                marker=dict(color="#2ca02c", size=14, symbol="square"),
            )
        )

        # 4PL Fit Line
        fig.add_trace(
            go.Scatter(
                x=x_smooth,
                y=y_smooth,
                mode="lines",
                name=f"4PL Fit (IC50 = {ic50_nM:.2f} nM)",
                line=dict(color="#1f77b4", width=4),
            )
        )

        # IC50 Coordinate Point (Hidden from legend)
        fig.add_trace(
            go.Scatter(
                x=[log_ic50_fit],
                y=[ic50_y_val],
                mode="markers",
                name="IC50 Point",
                showlegend=False,
                marker=dict(
                    color="#FF4B4B",
                    size=12,
                    symbol="circle",
                    line=dict(color="white", width=2),
                ),
            )
        )

        # Dashed Crosshair Guidelines
        fig.add_shape(
            type="line",
            x0=log_ic50_fit,
            x1=log_ic50_fit,
            y0=sample_df["RESP"].min() - 5,
            y1=ic50_y_val,
            line=dict(color="#FF4B4B", width=1.5, dash="dash"),
        )
        fig.add_shape(
            type="line",
            x0=sample_df["Log_Conc"].min() - 0.2,
            x1=log_ic50_fit,
            y0=ic50_y_val,
            y1=ic50_y_val,
            line=dict(color="#FF4B4B", width=1.5, dash="dash"),
        )

        # SLEEK IC50 PILL (No arrows/shapes blocking data)
        fig.add_annotation(
            x=log_ic50_fit,
            y=ic50_y_val,
            text=f"<b>IC50: {ic50_nM:.2f} nM</b>",
            showarrow=False,
            xanchor="left",
            yanchor="bottom",
            xshift=14,
            yshift=14,
            bgcolor="#1E1E1E",
            font=dict(color="#FF4B4B", size=18),
            bordercolor="#FF4B4B",
            borderwidth=1.5,
            borderpad=8,
            opacity=0.95,
        )

        # Scaled Up Layout
        fig.update_layout(
            title=dict(
                text=f"<b>{selected_plate} — Dose Response Curve</b>",
                font=dict(size=30),
            ),
            xaxis=dict(
                title=dict(
                    text="<b>Log(Concentration), M</b>", font=dict(size=24)
                ),
                tickfont=dict(size=20),
                gridwidth=1.5,
            ),
            yaxis=dict(
                title=dict(
                    text="<b>% Response (Relative to DMSO)</b>",
                    font=dict(size=24),
                ),
                tickfont=dict(size=20),
                gridwidth=1.5,
            ),
            legend=dict(
                font=dict(size=18),
                itemsizing="constant",
                x=0.68,
                y=0.95,
            ),
            height=800,
            margin=dict(l=40, r=40, t=70, b=40),
        )

        # Column Layout Split
        col_chart, col_stats = st.columns([8.2, 1.8])

        with col_chart:
            st.plotly_chart(fig, use_container_width=True)

        with col_stats:
            st.markdown("### Fit Metrics")
            st.markdown("---")
            st.metric("IC50", f"{ic50_nM:.2f} nM")
            st.write("")
            st.metric("Top Plateau", f"{top_fit:.1f} %")
            st.write("")
            st.metric("Bottom Plateau", f"{bottom_fit:.1f} %")
            st.write("")
            st.metric("Hill Slope", f"{hill_fit:.2f}")
            st.write("")
            st.metric("DMSO Baseline", f"{dmso_baseline:.0f} RLU")

        # Export Section
        st.markdown("---")
        st.subheader("Export Clean Data")

        csv_buffer = sample_df[
            [
                "Row",
                "Column",
                "Concentration_M",
                "Log_Conc",
                "RLU",
                "RESP",
            ]
        ].to_csv(index=False)
        st.download_button(
            label="Download Prism-Ready CSV",
            data=csv_buffer,
            file_name=f"{selected_plate}_dose_response_cleaned.csv",
            mime="text/csv",
        )

    except Exception as e:
        st.error(f"Curve fitting error: {e}")

else:
    st.info("Upload a TECAN .xlsx, ViewLux .txt, or CSV file to start.")
