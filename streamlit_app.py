# import pandas as pd
# import streamlit as st
# import matplotlib.pyplot as plt
# import numpy as np
# from datetime import datetime, timedelta
# import math
# from pathlib import Path
# import requests
# import base64

# # Streamlit Page Configuration
# st.set_page_config(
#     page_title='Atlas Dashboard',
#     page_icon='🚰', 
# )

# # -----------------------------------------------------------------------------
# # ThingSpeak API details
# CHANNEL_ID = st.secrets["api_keys"]["channel_id"]
# READ_API_KEY = st.secrets["api_keys"]["read_api_key"]
# NUM_RESULTS = 100  

# # Fetch data from ThingSpeak
# def fetch_thingspeak_data(channel_id, read_api_key, num_results):
#     url = f"https://api.thingspeak.com/channels/{channel_id}/feeds.json"
#     params = {
#         'api_key': read_api_key,
#         'results': num_results
#     }
#     response = requests.get(url, params=params)
#     if response.status_code == 200:
#         data = response.json()
#         return data['feeds'], data['channel']
#     else:
#         st.error("Error fetching data from ThingSpeak")
#         return None, None

# # Process data
# def process_data(feeds, channel_info):
#     df = pd.DataFrame(feeds)
#     df['created_at'] = pd.to_datetime(df['created_at'])
#     for field_key in channel_info.keys():
#         if field_key.startswith("field"):
#             field_name = channel_info[field_key]
#             df[field_name] = pd.to_numeric(df[field_key], errors='coerce')
#     return df

# # Van Haute's Model
# def van_haute_model(orp, ph):
#     intercept = 0.44
#     orp_coeff = -0.015
#     orp_squared_coeff = 1.1e-5
#     interaction_coeff = 8.4e-4
#     log_fcr = (intercept 
#                + orp_coeff * orp 
#                + orp_squared_coeff * (orp ** 2) 
#                + interaction_coeff * (orp * ph))
#     return log_fcr

# # Base64 Encoding for Images
# def get_base64_of_bin_file(bin_file):
#     with open(bin_file, 'rb') as f:
#         data = f.read()
#     return base64.b64encode(data).decode()

# # Paths for images
# safe_water_path = "safe_water.png"
# unsafe_water_path = "unsafe_water.png"

# # Display the app title and description
# st.title("🚰 Atlas Dashboard")
# st.markdown("Welcome to your Atlas Dashboard from the *Pickering Lab*! Monitor real-time chlorine residual levels, ORP, pH, and temperature directly from your Atlas Device.")

# # Fetch and process data
# feeds, channel_info = fetch_thingspeak_data(CHANNEL_ID, READ_API_KEY, NUM_RESULTS)

# if feeds and channel_info:
#     df = process_data(feeds, channel_info)
#     df = df.drop(columns=['entry_id', 'field1', 'field2', 'field3'])
#     df['created_at'] = df['created_at'].dt.strftime('%Y-%m-%d %H:%M')

#     if 'ORP (mV)' in df.columns and 'pH' in df.columns:
#         df['log_FCR'] = df.apply(lambda row: van_haute_model(row['ORP (mV)'], row['pH']), axis=1)
#         df['FCR (mg/L)'] = 10 ** df['log_FCR']
        
#         # User Role Selection
#         st.markdown("Tell Us About Yourself:")
#         user_role = st.selectbox("", ["Select Your Role", "Community Member", "Technician", "Researcher", "NGO/Government"])

#         # Safe Thresholds
#         safe_threshold_low = 0.2
#         safe_threshold_high = 0.8

#         if user_role == "NGO/Government":
#             st.title("NGO/Government Dashboard")
            
#             # Local/Regional dropdown
#             view_option = st.selectbox("Select View", ["Local", "Regional"])

#             # Metrics Section
#             current_fcr = df["FCR (mg/L)"].iloc[-1]
#             daily_avg = df["FCR (mg/L)"].iloc[-24:].mean()
#             weekly_avg = df["FCR (mg/L)"].iloc[-168:].mean()

#             col1, col2, col3 = st.columns(3)
#             col1.metric("Current FCR Level (mg/L)", f"{current_fcr:.2f}")
#             col2.metric("Daily FCR Average (mg/L)", f"{daily_avg:.2f}")
#             col3.metric("Weekly FCR Average (mg/L)", f"{weekly_avg:.2f}")

#             # Time Series Graph with Thresholds
#             st.subheader("FCR Time Series with Thresholds")
#             time_window = st.slider("Select Time Window (Hours)", 1, 100, 24)
#             filtered_df = df.tail(time_window)

#             plt.figure(figsize=(10, 5))
#             plt.plot(filtered_df["created_at"], filtered_df["FCR (mg/L)"], label="FCR", color="blue")
#             plt.axhline(y=0.2, color="red", linestyle="--", label="Min Threshold (0.2 mg/L)")
#             plt.axhline(y=0.8, color="red", linestyle="--", label="Max Threshold (0.8 mg/L)")
#             plt.xlabel("Timestamp")
#             plt.ylabel("FCR (mg/L)")
#             plt.title("FCR Over Time")
#             plt.legend()
#             st.pyplot(plt)

#             # Additional Metrics
#             no_chlorine_count = (filtered_df["FCR (mg/L)"] < 0.2).sum()
#             detectable_chlorine_count = (filtered_df["FCR (mg/L)"] >= 0.2).sum()
#             no_chlorine_proportion = no_chlorine_count / len(filtered_df)

#             st.subheader("Chlorine Metrics")
#             col4, col5, col6 = st.columns(3)
#             col4.metric("No Chlorine Detected", no_chlorine_count)
#             col5.metric("Proportion of No Chlorine", f"{no_chlorine_proportion:.2%}")
#             col6.metric("Detectable Chlorine Count", detectable_chlorine_count)







#### original Code ### 
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import math
from pathlib import Path
import requests
import base64

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


# Function to get base64 of the image
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Path to the logo image
safe_water_path = "safe_water.png"
unsafe_water_path = "unsafe_water.png"
safe_logo_base64 = get_base64_of_bin_file(safe_water_path)
unsafe_logo_base64 = get_base64_of_bin_file(unsafe_water_path)

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
        st.markdown("Tell Us About Yourself:")
        user_role = st.selectbox("", ["Select Your Role", "Community Member", "Technician", "Researcher", "NGO/Government"])
        
        # Define Safe Thresholds
        safe_threshold_low = 0.2
        safe_threshold_high = 1.0
        
        # Display for Community Member
        if user_role == "Community Member":
            last_fcr = df['FCR (mg/L)'].iloc[-1]
            if last_fcr > safe_threshold_low and last_fcr < safe_threshold_high:
                # st.markdown(f"<span style='color:green; font-size:24px;'>Safe</span>", unsafe_allow_html=True)
                # Add Safe Water image to top center and subtitle underneath
                st.image("safe_water.png", caption="Water is Safe", use_column_width=True)
                
            else:
                # st.markdown(f"<span style='color:red; font-size:24px;'>Unsafe</span>", unsafe_allow_html=True)
                st.image("unsafe_water.png", caption="Water is Unsafe", use_column_width=True)
        
        # Display for Technician
        elif user_role == "Technician":
            last_fcr = df['FCR (mg/L)'].iloc[-1]
            st.subheader("Current Chlorine Status")
            color = "green" if last_fcr > safe_threshold_low and last_fcr < safe_threshold_high else "red"
            st.markdown(f"<span style='color:{color}; font-size:24px;'>{last_fcr:.2f} mg/L</span>", unsafe_allow_html=True)
            
            # Trend of FCR over time
            st.subheader("FCR Trend Today")
            st.line_chart(df[['created_at', 'FCR (mg/L)']].set_index('created_at')[-10:])
        
        # Display for Researcher
        elif user_role == "Researcher":
            st.write(f"FCR data from ThingSpeak Channel: {channel_info['name']}")
            # Display raw FCR data
            st.subheader("Raw FCR Data")
            st.dataframe(df)
            
            # Trend of FCR over time
            st.subheader("FCR Trend This Month")
            st.line_chart(df[['created_at', 'FCR (mg/L)']].set_index('created_at'))
        
        elif user_role == "NGO/Government":
            # st.title("NGO/Government Dashboard")
            
            # Local/Regional dropdown
            view_option = st.selectbox("Select View", ["Local", "Regional"])

            # Metrics Section
            current_fcr = df["FCR (mg/L)"].iloc[-1]
            daily_avg = df["FCR (mg/L)"].iloc[-24:].mean()
            weekly_avg = df["FCR (mg/L)"].iloc[-168:].mean()

            col1, col2, col3 = st.columns(3)
            col1.metric("Current FCR Level (mg/L)", f"{current_fcr:.2f}")
            col2.metric("Daily FCR Average (mg/L)", f"{daily_avg:.2f}")
            col3.metric("Weekly FCR Average (mg/L)", f"{weekly_avg:.2f}")

            # Time Series Graph with Thresholds
            st.subheader("FCR Time Series with Thresholds")
            time_window = st.slider("Select Time Window (Hours)", 1, 100, 24)
            filtered_df = df.tail(time_window)

            plt.figure(figsize=(10, 5))
            plt.plot(filtered_df["created_at"], filtered_df["FCR (mg/L)"], label="FCR", color="blue")
            plt.axhline(y=0.2, color="red", linestyle="--", label="Min Threshold (0.2 mg/L)")
            plt.axhline(y=0.8, color="red", linestyle="--", label="Max Threshold (0.8 mg/L)")
            plt.xlabel("Timestamp")
            plt.ylabel("FCR (mg/L)")
            plt.title("FCR Over Time")
            plt.legend()
            st.pyplot(plt)

            # Additional Metrics
            no_chlorine_count = (filtered_df["FCR (mg/L)"] < 0.2).sum()
            detectable_chlorine_count = (filtered_df["FCR (mg/L)"] >= 0.2).sum()
            no_chlorine_proportion = no_chlorine_count / len(filtered_df)

            st.subheader("Chlorine Metrics")
            col4, col5, col6 = st.columns(3)
            col4.metric("No Chlorine Detected", no_chlorine_count)
            col5.metric("Proportion of No Chlorine", f"{no_chlorine_proportion:.2%}")
            col6.metric("Detectable Chlorine Count", detectable_chlorine_count)
    else:
        st.error("The dataset does not contain ORP and pH readings required for FCR calculation.")
else:
    st.error("Failed to load data from ThingSpeak.")

