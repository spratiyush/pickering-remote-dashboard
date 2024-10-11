import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import math
from pathlib import Path

# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='Water Quality Dashboard - Pickering Lab',
    page_icon=':earth_americas:', # This is an emoji shortcode. Could be a URL too.
)

# -----------------------------------------------------------------------------
# Draw the actual page

# Display in Streamlit
st.title("🚰 Atlas Dashboard")
st.markdown("Welcome to your Atlas Dashboard from the Pickering Lab! Monitor real-time chlorine residual levels, ORP, pH, and temperature directly from your Atlas Device. Our dashboard provides predictive insights, historical trends, and sends timely alerts to ensure optimal water quality and safety.")

# Create dummy data
np.random.seed(42)
num_days = 7
dates = [datetime.now() - timedelta(days=i) for i in range(num_days)][::-1]
chlorine_levels_local = np.random.uniform(0.1, 0.8, size=num_days)  # Local chlorine levels (mg/L)
chlorine_levels_regional = np.random.uniform(0.15, 0.9, size=num_days)  # Regional chlorine levels (mg/L)

# Create a DataFrame with dummy data
data = pd.DataFrame({
    'Date': dates,
    'Local Chlorine Level (mg/L)': chlorine_levels_local,
    'Regional Chlorine Level (mg/L)': chlorine_levels_regional
})

# Current chlorine status
current_local_chlorine = chlorine_levels_local[-1]
current_regional_chlorine = chlorine_levels_regional[-1]
safe_threshold = 0.2  # Safe chlorine level threshold (mg/L)

# Determine status for the current chlorine level
status_local = "Safe" if current_local_chlorine >= safe_threshold else "Unsafe"
status_regional = "Safe" if current_regional_chlorine >= safe_threshold else "Unsafe"

# Current Chlorine Status
st.subheader("Current Chlorine Status")
option = st.selectbox('Select level to display', ['Local', 'Regional'])

if option == 'Local':
    st.metric("Local Chlorine Level", f"{current_local_chlorine:.2f} mg/L", status_local)
else:
    st.metric("Regional Chlorine Level", f"{current_regional_chlorine:.2f} mg/L", status_regional)

# Create a plot for the last 7 days trend
st.subheader("Chlorine Levels Trend - Last 7 Days")
plt.figure(figsize=(10, 5))
plt.plot(data['Date'], data['Local Chlorine Level (mg/L)'], marker='o', label='Local')
plt.plot(data['Date'], data['Regional Chlorine Level (mg/L)'], marker='o', linestyle='--', label='Regional')
plt.axhline(y=safe_threshold, color='gray', linestyle='--', label='Safe Threshold (0.2 mg/L)')
plt.xlabel('Date')
plt.ylabel('Chlorine Level (mg/L)')
plt.title('Chlorine Levels Over the Last 7 Days')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()

# Display the plot in Streamlit
st.pyplot(plt)

# Show data as a table for reference
st.subheader("Chlorine Data")
st.dataframe(data)
