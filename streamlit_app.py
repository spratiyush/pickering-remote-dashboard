import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import math
from pathlib import Path
import requests

# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='Atlas Dashboard',
    page_icon='🚰', 
)

# -----------------------------------------------------------------------------

# ThingSpeak API details
CHANNEL_ID = '2659578'  
READ_API_KEY = 'RZ5A2FF8B3GZIQWQ'  
NUM_RESULTS = 100  

def fetch_thingspeak_data(channel_id, read_api_key, num_results):
    url = f"https://api.thingspeak.com/channels/{channel_id}/feeds.json"
    params = {
        'api_key': read_api_key,
        'results': num_results
    }
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        return data['feeds'], data['channel']
    else:
        st.error("Error fetching data from ThingSpeak")
        return None, None

# Function to process ThingSpeak data into a DataFrame
def process_data(feeds, channel_info):
    df = pd.DataFrame(feeds)
    df['created_at'] = pd.to_datetime(df['created_at'])
    # Map field names from the channel to DataFrame columns
    for field_key in channel_info.keys():
        if field_key.startswith("field"):
            field_name = channel_info[field_key]
            df[field_name] = pd.to_numeric(df[field_key], errors='coerce')
    
    return df

# Display in Streamlit
st.title("🚰 Atlas Dashboard")
st.markdown("Welcome to your Atlas Dashboard from the *Pickering Lab*! Monitor real-time chlorine residual levels, ORP, pH, and temperature directly from your Atlas Device.")

# Fetch and process data
feeds, channel_info = fetch_thingspeak_data(CHANNEL_ID, READ_API_KEY, NUM_RESULTS)

if feeds and channel_info:
    df = process_data(feeds, channel_info)
    st.write(f"This is data from ThingSpeak Channel: {channel_info['name']}")
    
    # Display the raw data
    st.subheader("Raw Data")
    st.dataframe(df)

    # Display charts for each sensor field
    sensor_fields = [col for col in df.columns if col.startswith("field")]
    for field in sensor_fields:
        if field == "field1":
            field = "pH"
        elif field == "field2":
            field = "ORP"
        else:
            field = "Temperature (C)"
        st.subheader(f"{field} Over Time")
        st.line_chart(df[['created_at', field]].set_index('created_at'))
else:
    st.error("Failed to load data from ThingSpeak.")


# =========== DUMMY CODE AND DATA BELOW ==================

# # Create dummy data
# np.random.seed(42)
# num_days = 7
# dates = [datetime.now() - timedelta(days=i) for i in range(num_days)][::-1]
# chlorine_levels_local = np.random.uniform(0.1, 0.8, size=num_days)  # Local chlorine levels (mg/L)
# chlorine_levels_regional = np.random.uniform(0.15, 0.9, size=num_days)  # Regional chlorine levels (mg/L)

# # Create a DataFrame with dummy data
# data = pd.DataFrame({
#     'Date': dates,
#     'Local Chlorine Level (mg/L)': chlorine_levels_local,
#     'Regional Chlorine Level (mg/L)': chlorine_levels_regional
# })

# # Current chlorine status
# current_local_chlorine = chlorine_levels_local[-1]
# current_regional_chlorine = chlorine_levels_regional[-1]
# safe_threshold = 0.2  # Safe chlorine level threshold (mg/L)

# # Determine status for the current chlorine level
# status_local = "Safe" if current_local_chlorine >= safe_threshold else "Unsafe"
# status_regional = "Safe" if current_regional_chlorine >= safe_threshold else "Unsafe"

# # Define a function to display colored status
# def display_status(status):
#     color = "green" if status == "Safe" else "red"
#     st.markdown(f"<span style='color:{color}; font-size:24px;'>{status}</span>", unsafe_allow_html=True)

# # Current Chlorine Status
# st.subheader("Current Chlorine Status")
# option = st.selectbox('Select level to display', ['Local', 'Regional'])

# if option == 'Local':
#     # st.markdown(status_local)
    
#     st.metric("Local Chlorine Level", f"{current_local_chlorine:.2f} mg/L")
#     display_status(status_local)
# else:
#     st.metric("Regional Chlorine Level", f"{current_regional_chlorine:.2f} mg/L")
#     display_status(status_regional)
#     # st.markdown(status_regional)
    

# # Create a plot for the last 7 days trend
# st.subheader("Weekly Trend")
# plt.figure(figsize=(10, 5))
# plt.plot(data['Date'], data['Local Chlorine Level (mg/L)'], marker='o', label='Local')
# plt.plot(data['Date'], data['Regional Chlorine Level (mg/L)'], marker='o', linestyle='--', label='Regional')
# plt.axhline(y=safe_threshold, color='gray', linestyle='--', label='Safe Threshold (0.2 mg/L)')
# plt.xlabel('Date')
# plt.ylabel('Chlorine Level (mg/L)')
# plt.title('Chlorine Levels Over the Last 7 Days')
# plt.xticks(rotation=45)
# plt.legend()
# plt.tight_layout()

# # Display the plot in Streamlit
# st.pyplot(plt)

# # Show data as a table for reference
# st.subheader("Chlorine Data")
# st.dataframe(data)
