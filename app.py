import streamlit as st
import pandas as pd
import joblib
from sklearn.neighbors import NearestNeighbors
import numpy as np

import os
import streamlit as st

import streamlit as st
from PIL import Image

# Must be near the top of your file
st.set_page_config(page_title="Recruit Similarity Finder", layout="wide")

# Load and display logo
logo = Image.open("dc.png")  # replace with your file name
st.image(logo, width=150)  # adjust width as needed


st.caption(f"cwd: {os.getcwd()} | file: {__file__}")


# --- Load your saved model and pipeline ---
import joblib
import streamlit as st

assets = joblib.load("recruit_model.joblib")

# Display keys in the sidebar to confirm correct model load
# st.sidebar.write("Assets keys:", list(assets.keys()))

try:
    pipe = assets["pipe"]
    FEATURES = assets["feature_cols"]
    Z_recruits = assets["Z_recruits"]          # <- confirmed correct from your test
    recruits_meta = assets["recruits_meta"]
    kmeans = assets["kmeans"]
except KeyError as e:
    st.error(f"Missing key {e!s} in recruit_model.joblib. "
             f"Available keys: {list(assets.keys())}")
    st.stop()

# --- Load current players data ---
from pathlib import Path
import pandas as pd
import streamlit as st

def load_current_players():
    # 1) Let the user upload, which overrides everything else
    uploaded = st.sidebar.file_uploader(
        "Upload current players sheet (.csv or .xlsx)", type=["csv", "xlsx"]
    )
    if uploaded is not None:
        if uploaded.name.lower().endswith(".xlsx"):
            return pd.read_excel(uploaded, sheet_name="Sheet1")
        else:
            return pd.read_csv(uploaded)

    # 2) Try common local filenames (CSV or Excel) in the same folder as app.py
    candidates = [
        "2025 FB Spring Test Numbers.xlsx - Sheet1.csv",   # your exported CSV name
        "2025 FB Spring Test Numbers - Sheet1.csv",
        "2025 FB Spring Test Numbers.csv",
        "2025 FB Spring Test Numbers.xlsx",                # the original Excel
    ]
    for name in candidates:
        p = Path(name)
        if p.exists():
            if p.suffix.lower() == ".xlsx":
                # If it’s an Excel file, read the Sheet1 tab
                return pd.read_excel(p, sheet_name="Sheet1")
            else:
                return pd.read_csv(p)

    # 3) Last resort: glob anything similar in the folder
    hits = list(Path(".").glob("*Spring Test Numbers*.csv")) + \
           list(Path(".").glob("*Spring Test Numbers*.xlsx"))
    if hits:
        p = hits[0]
        if p.suffix.lower() == ".xlsx":
            return pd.read_excel(p, sheet_name="Sheet1")
        else:
            return pd.read_csv(p)

    # 4) If still nothing found, show a nice message in the app
    st.error("Couldn't find your current players file. "
             "Put the CSV/XLSX in the same folder as `app.py` "
             "or upload it via the sidebar.")
    st.stop()

# ---- use it here ----
current = load_current_players()

# Make the FullName column (trim spaces to avoid mismatches)
current["First Name"] = current["First Name"].astype(str).str.strip()
current["Last Name"]  = current["Last Name"].astype(str).str.strip()
current["FullName"]   = current["First Name"] + " " + current["Last Name"]

# --- Fit nearest-neighbor model for recruits ---
nn = NearestNeighbors(n_neighbors=10, metric="euclidean").fit(Z_recruits)

# --- Streamlit UI ---
st.title("Recruit to Current Player Finder")
st.write("Compare 2024 Davidson players to similar recruits based on test metrics and clustering.")

# Player selection
player_name = st.selectbox("Select a current player:", current["FullName"].unique())
pos_pick = st.selectbox("Filter by position:", ["All"] + sorted(recruits_meta["Pos"].dropna().unique()))

if st.button("Find Similar Recruits"):
    # Extract the player's numeric features
    row = current.loc[current["FullName"] == player_name, FEATURES]
    Xp = row.apply(pd.to_numeric, errors="coerce")

    # Transform into the same PCA/scaled space
    zp = pipe.transform(Xp)

    # Predict player cluster (optional)
    cluster = int(kmeans.predict(zp)[0])
    st.metric("Predicted Cluster", cluster)

    # Find nearest recruits
    dist, idx = nn.kneighbors(zp, n_neighbors=10)
    out = recruits_meta.iloc[idx[0]].copy()
    out.insert(0, "Distance", np.round(dist[0], 3))

    # Filter by position
    if pos_pick != "All" and "Pos" in out.columns:
        out = out[out["Pos"] == pos_pick]

    # Show results
    st.subheader("Most Similar Recruits")
    st.dataframe(out, use_container_width=True)

    st.download_button(
        label="Download as CSV",
        data=out.to_csv(index=False),
        file_name=f"similar_recruits_{player_name.replace(' ', '_')}.csv",
        mime="text/csv"
    )
