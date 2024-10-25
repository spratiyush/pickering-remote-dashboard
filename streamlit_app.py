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
CHANNEL_ID = st.secrets["api_keys"]["channel_id"]
READ_API_KEY = st.secrets["api_keys"]["read_api_key"]
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

def van_haute_model(orp, ph):
    intercept = 0.44
    orp_coeff = -0.015
    orp_squared_coeff = 1.1e-5
    interaction_coeff = 8.4e-4
    
    log_fcr = (intercept 
               + orp_coeff * orp 
               + orp_squared_coeff * (orp ** 2) 
               + interaction_coeff * (orp * ph))
    
    return log_fcr

# Display in Streamlit
st.title("🚰 Atlas Dashboard")
st.markdown("Welcome to your Atlas Dashboard from the *Pickering Lab*! Monitor real-time chlorine residual levels, ORP, pH, and temperature directly from your Atlas Device.")

# Fetch and process data
feeds, channel_info = fetch_thingspeak_data(CHANNEL_ID, READ_API_KEY, NUM_RESULTS)



if feeds and channel_info:
    df = process_data(feeds, channel_info)
    # Remove and simplify columns
    df = df.drop(columns=['entry_id', 'field1', 'field2', 'field3'])
    df['created_at'] = df['created_at'].dt.strftime('%Y-%m-%d %H:%M')
    
    # Check for the presence of ORP and pH columns in the data
    if 'ORP (mV)' in df.columns and 'pH' in df.columns:
        # Calculate FCR using Van Haute's Model
        df['log_FCR'] = df.apply(lambda row: van_haute_model(row['ORP (mV)'], row['pH']), axis=1)
        df['FCR (mg/L)'] = 10 ** df['log_FCR']  # Convert log(FCR) to FCR
        
        # User Role Selection
        user_role = st.selectbox("Tell us about you:", ["Select Your Role", "Community Member", "Technician", "Researcher"])
        
        # Define Safe Threshold
        safe_threshold = 0.2
        
        # Display for Community Member
        if user_role == "Community Member":
            last_fcr = df['FCR (mg/L)'].iloc[-1]
            if last_fcr > safe_threshold:
                st.markdown(f"<span style='color:green; font-size:24px;'>Safe</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"<span style='color:red; font-size:24px;'>Unsafe</span>", unsafe_allow_html=True)
        
        # Display for Technician
        elif user_role == "Technician":
            last_fcr = df['FCR (mg/L)'].iloc[-1]
            st.subheader("Current Chlorine Status")
            color = "green" if last_fcr > safe_threshold else "red"
            st.markdown(f"<span style='color:{color}; font-size:24px;'>{last_fcr:.2f} mg/L</span>", unsafe_allow_html=True)
            
            # Trend of FCR over time
            st.subheader("FCR Trend Today")
            st.line_chart(df[['created_at', 'FCR (mg/L)']].set_index('created_at'))
        
        # Display for Researcher
        elif user_role == "Researcher":
            st.write(f"FCR data from ThingSpeak Channel: {channel_info['name']}")
            # Display raw FCR data
            st.subheader("Raw FCR Data")
            st.dataframe(df)
            
            # Trend of FCR over time
            st.subheader("FCR Trend This Month")
            st.line_chart(df[['created_at', 'FCR (mg/L)']].set_index('created_at'))
    else:
        st.error("The dataset does not contain ORP and pH readings required for FCR calculation.")
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
