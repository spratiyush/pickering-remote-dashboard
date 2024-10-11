import streamlit as st
import pandas as pd
import requests
from datetime import datetime

# Set up your ThingSpeak API details
CHANNEL_ID = 'YOUR_CHANNEL_ID'  
READ_API_KEY = 'YOUR_READ_API_KEY'  
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

# Streamlit app
st.title('ThingSpeak Data Dashboard')
st.markdown('Displays data from a specified ThingSpeak channel.')

# Fetch and process data
feeds, channel_info = fetch_thingspeak_data(CHANNEL_ID, READ_API_KEY, NUM_RESULTS)

if feeds and channel_info:
    df = process_data(feeds, channel_info)
    st.write(f"Data from ThingSpeak channel: {channel_info['name']}")
    st.write(f"Channel Description: {channel_info['description']}")
    
    # Display the raw data
    st.subheader("Raw Data")
    st.dataframe(df)

    # Display charts for each sensor field
    sensor_fields = [col for col in df.columns if col.startswith("field")]
    for field in sensor_fields:
        st.subheader(f"{field} Over Time")
        st.line_chart(df[['created_at', field]].set_index('created_at'))
else:
    st.error("Failed to load data from ThingSpeak.")
