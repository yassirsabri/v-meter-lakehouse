import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import os

# Page Configuration
st.set_page_config(
    page_title="ICARDA - Videometer Analytics",
    page_icon="🌾",
    layout="wide"
)

# Constants
API_URL = os.getenv("API_URL", "http://prediction-api:8000")

# CSS for better design (Fixed for Dark Mode visibility)
st.markdown("""
    <style>
    .main-header { font-size: 2.5rem; color: #2E8B57; font-weight: bold; }
    .metric-card { background-color: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 5px solid #2E8B57; }
    
    /* Custom HTML Table Styling - Dark Mode Safe */
    .dataframe-container { overflow-x: auto; margin-bottom: 1rem; }
    table.custom-dataframe { border-collapse: collapse; width: 100%; font-size: 14px; }
    table.custom-dataframe th { background-color: #2E8B57 !important; color: white !important; text-align: left; padding: 10px; border: 1px solid #555; }
    table.custom-dataframe td { padding: 8px; border: 1px solid #555; }
    table.custom-dataframe tr:nth-child(even) { background-color: rgba(255, 255, 255, 0.05); }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">🌾 ICARDA Analytics Hub: Lakehouse & MLOps</p>', unsafe_allow_html=True)
st.write("This portal directly queries the Bronze layer (MinIO) in real-time via Predicate Pushdown.")

# Tabs creation
tab1, tab2, tab3 = st.tabs(["🎯 Bronze Layer Exploration", "📊 MLOps & API Monitoring", "💡 Architecture Overview"])

def decode_iceberg_columns(df):
    """Decodes Iceberg column names and cleans raw Videometer artifacts."""
    new_cols = {}
    for col in df.columns:
        clean_col = str(col)
        # 1. Decode Iceberg XML-like tags
        clean_col = clean_col.replace("_x20_", " ").replace("_x28_", "(").replace("_x29_", ")")
        clean_col = clean_col.replace("_x5B_", "[").replace("_x5D_", "]").replace("_x2F_", "/")
        
        # 2. FIX CHICKPEA BUG: Remove the annoying Videometer (Unknown) artifact
        clean_col = clean_col.replace("(Unknown)", "").replace("Unknown", "")
        
        # 3. Clean up extra spaces
        new_cols[col] = clean_col.strip()
        
    return df.rename(columns=new_cols)

# ==========================================
# TAB 1: BRONZE LAYER EXPLORATION
# ==========================================
with tab1:
    st.subheader("Instant Data Lake Inspection")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 1. Search Parameters")
        
        crop_options = ["All crops (Auto-detect)", "barley", "chickpea", "wheat"]
        selected_crop = st.selectbox("Crop (Optional)", crop_options)
        api_crop_param = None if "All" in selected_crop else selected_crop

        accessions_input = st.text_input(
            "Accession(s)", 
            value="8586", 
            help="Enter one or multiple accessions separated by commas (e.g., 8586, 16988)"
        )
        
        analyze_btn = st.button("🔍 Extract Data", type="primary", use_container_width=True)

    if analyze_btn and accessions_input:
        with st.spinner('PyArrow is scanning the Data Lake (MinIO) in real-time...'):
            try:
                params = {"accessions": accessions_input, "limit": 500}
                if api_crop_param:
                    params["crop"] = api_crop_param
                    
                bronze_resp = requests.get(f"{API_URL}/data/bronze", params=params)
                
                if bronze_resp.status_code != 200:
                    st.error(f"Data not found: {bronze_resp.json().get('detail')}")
                else:
                    response_data = bronze_resp.json()
                    seeds_data = response_data["data"]
                    st.success(f"✅ {response_data['rows_returned']} record(s) successfully extracted from the Datalake!")
                    
                    results_df = pd.DataFrame(seeds_data)
                    results_df = decode_iceberg_columns(results_df)
                    
                    st.markdown("---")
                    st.markdown("### 2. Morphological Footprint")
                    
                    # --- FIX: Support multiple seeds for the graph ---
                    seed_filenames = results_df['Filename'].astype(str).unique().tolist()
                    
                    if len(seed_filenames) > 1:
                        selected_seed_plot = st.selectbox("Select an accession to visualize its physical profile:", seed_filenames)
                    else:
                        selected_seed_plot = seed_filenames[0]
                        st.write(f"**Visualizing Accession:** {selected_seed_plot}")
                        
                    # Filter the dataframe for the selected seed only
                    seed_plot_df = results_df[results_df['Filename'].astype(str) == selected_seed_plot]
                    
                    cols_to_show = [c for c in results_df.columns if "area" in c.lower() or "length" in c.lower() or "width" in c.lower() or "volume" in c.lower()]
                    
                    if cols_to_show and not seed_plot_df.empty:
                        numeric_chart_data = seed_plot_df[cols_to_show].iloc[0:1].apply(pd.to_numeric, errors='coerce').fillna(0)
                        chart_data = numeric_chart_data.T.reset_index()
                        
                        fig = px.bar(chart_data, x='index', y=0, 
                                     title=f"Physical Profile - Accession {selected_seed_plot}",
                                     labels={'index': 'Feature', '0': 'Value (Raw)'})
                        
                        fig.update_layout(xaxis_tickangle=-45)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("### 3. Raw Data Extracted from MinIO")
                    
                    # Drop completely empty columns
                    light_df = results_df.dropna(axis=1, how='all')
                    
                    # Display a safe HTML table
                    display_df = light_df.head(15).iloc[:, :15].astype(str)
                    html_table = display_df.to_html(index=False, classes="custom-dataframe")
                    st.markdown(f'<div class="dataframe-container">{html_table}</div>', unsafe_allow_html=True)
                    
                    st.caption(f"Displaying {len(display_df)} rows and the first 15 useful columns to ensure browser stability. Total valid columns extracted: {len(light_df.columns)}.")
                    
                    with st.expander("Show raw JSON payload (First seed only)"):
                        st.json(seeds_data[0] if len(seeds_data) > 0 else {})
                    
                    csv = light_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="⬇️ Download Full Dataset (CSV)",
                        data=csv,
                        file_name=f"icarda_bronze_extract.csv",
                        mime="text/csv",
                    )
                            
            except requests.exceptions.ConnectionError:
                st.error("Unable to connect to the FastAPI. Please check if containers are running.")

# ==========================================
# TAB 2: MLOPS & API HEALTH
# ==========================================
with tab2:
    st.subheader("Infrastructure Monitoring")
    
    if st.button("🔄 Refresh Status", use_container_width=True):
        try:
            health_resp = requests.get(f"{API_URL}/health")
            data = health_resp.json()
            
            col1, col2, col3 = st.columns(3)
            col1.metric("API Status", data.get("status", "Unknown").upper())
            col2.metric("ML Model Loaded", "YES ✅" if data.get("model_loaded") else "NO ❌")
            col3.metric("Tracking MLflow", "Connected")
            
            st.markdown("**Active Model Source:**")
            st.code(data.get("model_source"))
            
        except Exception as e:
            st.error(f"API Unreachable: {e}")
            
    st.markdown("---")
    st.markdown("### Hot-Reload (Zero Downtime)")
    st.write("If Airflow just finished a new training, click here to force the API to load the new MLflow model without service interruption.")
    if st.button("🚀 Force Model Reload (Hot-Swap)"):
        with st.spinner("Reloading into memory via MLflow..."):
            try:
                resp = requests.post(f"{API_URL}/reload")
                if resp.status_code == 200:
                    st.success("Model successfully updated!")
                    st.json(resp.json())
                else:
                    st.error("Reload failed.")
            except Exception as e:
                st.error(f"Error: {e}")

# ==========================================
# TAB 3: ARCHITECTURE OVERVIEW
# ==========================================
with tab3:
    st.subheader("How does it work?")
    st.markdown("""
    This application demonstrates the complete integration of the Lakehouse architecture:
    
    1. **Storage (MinIO)**: Videometer data is stored in Parquet format in the Bronze layer.
    2. **Predicate Pushdown (PyArrow)**: The API does not download the whole dataset. It directly queries the Warehouse to fetch only the requested accessions.
    3. **Inference (FastAPI + MLflow)**: Data is sent to the XGBoost model in memory.
    4. **MLOps (Airflow)**: The model can be retrained and hot-reloaded without ever taking this Dashboard offline.
    """)