from sklearn.preprocessing import MinMaxScaler
from collections import Counter
import geopandas as gpd
import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import pickle
import os

def load_processed_data(file_path):
    """
        Loads the processed datasets stored in pickle format. 
        If they don't exist, it preprocesses the raw data.
    """

    if os.path.exists(file_path):
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        return data
    else:
        st.error("You have to run the preprocessing notebook before conducting any visualization.")
        st.stop()

# Loads the data.
summer_collisions_2018 = load_processed_data('summer_collisions_2018.pkl')
summer_collisions_2020 = load_processed_data('summer_collisions_2020.pkl')
summer_weather_2018 = load_processed_data('summer_weather_2018.pkl')
summer_weather_2020 = load_processed_data('summer_weather_2020.pkl')

# Some altair settings.
alt.data_transformers.disable_max_rows()

# Concatenating both years datasets.
summer_collisions = pd.concat([summer_collisions_2018, summer_collisions_2020])

# Custom color palettes for different years
colors_2018 = ['#b2df8a', '#33a02c']
colors_2020 = ['#cab2d6', '#6a3d9a'] 

# Grouping by 'Month' and 'Day_Type' columns and computing the count
grouped = summer_collisions_2018.groupby(['MONTH_YEAR', 'DAY TYPE']).size().reset_index(name='Count')

grouped['Normalized_Count'] = grouped.groupby('MONTH_YEAR')['Count'].transform(lambda x: x / x.sum())

chart2018 = alt.Chart(grouped).mark_bar().encode(
    x=alt.X('MONTH_YEAR:N',title='', sort=['June 2018', 'July 2018', 'August 2018']),
    y=alt.Y('Normalized_Count:Q',title='Proportion of collisions'),
    color=alt.Color('DAY TYPE:N', scale=alt.Scale(range=colors_2018), title='Day Type'),
)

grouped = summer_collisions_2020.groupby(['MONTH_YEAR', 'DAY TYPE']).size().reset_index(name='Count')

grouped['Normalized_Count'] = grouped.groupby('MONTH_YEAR')['Count'].transform(lambda x: x / x.sum())

chart2020 = alt.Chart(grouped).mark_bar().encode(
    x=alt.X('MONTH_YEAR:N',title='', sort=['June 2020', 'July 2020', 'August 2020']),
    y=alt.Y('Normalized_Count:Q',title=''),
    color=alt.Color('DAY TYPE:N',scale=alt.Scale(range=colors_2020), title=''),
)

chart1 = (chart2018+chart2020).resolve_scale(color='independent').properties(title='Proportion of collisions based on the day type per month',
                                                             width=200, height=200)

# Count occurrences of each string and add to a list
flattened = [word for sublist in summer_collisions['VEHICLE TYPE'] for word in sublist]
counter = Counter(flattened)
type_counts = sorted(counter.items(), key=lambda x: x[1], reverse=True)

# Create a DataFrame from the sorted list
data_vehicles = pd.DataFrame(type_counts, columns=['Type', 'Frequency'])
data_filtered = data_vehicles[data_vehicles['Frequency'] > 100]

chart2 = alt.Chart(data_filtered).mark_bar().encode(
    x='Frequency:Q',
    y=alt.Y('Type:N', sort='-x', title='Type of vehicle'),
    color=alt.Color('Frequency:Q', scale=alt.Scale(scheme='orangered')), 
).properties(
    title='Frequency of Vehicle Types in collisions',
    width= 100,
    height=200
)

lines = alt.Chart(summer_collisions).mark_line(strokeDash=[3], strokeWidth=3).encode(
    x = alt.X('CAT_CRASH_TIME:N', axis=alt.Axis(title='Time of the Day')),
    y = alt.Y('count(COLLISION_ID):Q', axis=alt.Axis(title='Number of collisions')),
    color =  alt.Color('SUMMER:N', scale=alt.Scale(scheme='accent'))
).properties(
    title='Evolution of the number of collisions along the day'
)

points = alt.Chart(summer_collisions).mark_circle().encode(
    x = 'CAT_CRASH_TIME:N',
    y = 'count(COLLISION_ID):Q',
    opacity = alt.value(1.0),
    size=alt.value(100),
    color =  alt.Color('SUMMER:N', scale=alt.Scale(scheme='accent'), title='Year')
)

chart3 = (lines+points).properties(width=200,height=200)

from PIL import Image

# Load PNG image
image_path = "map.png"
image = Image.open(image_path)

# Merge datasets on 'crash_date' and 'date' columns

summer_collisions_2018['CRASH DATE'] = pd.to_datetime(summer_collisions_2018['CRASH DATE'])
summer_weather_2018['DATE'] = pd.to_datetime(summer_weather_2018['DATE'])

merged_data_2018 = pd.merge(summer_collisions_2018, summer_weather_2018, left_on='CRASH DATE', right_on='DATE', how='left')

summer_collisions_2020['CRASH DATE'] = pd.to_datetime(summer_collisions_2020['CRASH DATE'])
summer_weather_2020['DATE'] = pd.to_datetime(summer_weather_2020['DATE'])

merged_data_2020 = pd.merge(summer_collisions_2020, summer_weather_2020, left_on='CRASH DATE', right_on='DATE', how='left')

# Removal of unused columns
columns_to_remove = ['STATION', 'NAME', 'LATITUDE_y', 'LONGITUDE_y', 'ELEVATION', 'DATE']
merged_data_2018 = merged_data_2018.drop(columns=columns_to_remove)
merged_data_2020 = merged_data_2020.drop(columns=columns_to_remove)
merged_data = pd.concat([merged_data_2018,merged_data_2020])

# Normalize values
columns_to_normalize = ['AWND', 'PRCP', 'SNOW', 'SNWD', 'TAVG', 'TMAX', 'TMIN', 'WDF2', 'WDF5', 'WSF2', 'WSF5']
scaler = MinMaxScaler()

merged_data_2018[columns_to_normalize] = scaler.fit_transform(merged_data_2018[columns_to_normalize])
merged_data_2020[columns_to_normalize] = scaler.fit_transform(merged_data_2020[columns_to_normalize])
merged_data[columns_to_normalize] = scaler.fit_transform(merged_data[columns_to_normalize])

columns_of_interest = ['MONTH_YEAR', 'AWND', 'PRCP', 'SNOW', 'TAVG', 'TMAX', 'TMIN', 'WDF2', 'WDF5', 'WSF2', 'WSF5']
extracted_data = merged_data[columns_of_interest]

averages_by_month = extracted_data.groupby('MONTH_YEAR').mean()
averages_by_month.reset_index(inplace=True)
restr = pd.melt(averages_by_month, id_vars=['MONTH_YEAR'], var_name='CONDITIONS', value_name='VALUES',
               ignore_index=True)

chart5 = alt.Chart(restr).mark_rect().encode(
    x=alt.X('MONTH_YEAR:O', title='Month', sort=['June 2018', 'July 2018', 'August 2018', 'June 2020', 'July 2020', 'August 2020']),
    y=alt.Y('CONDITIONS:N', title='Weather Condition', sort='-x'),
    color=alt.Color('VALUES:Q', scale=alt.Scale(scheme='goldorange'), title='Normalized Avg'),
).properties(
    width=500,
    height=300,
    title='Correlation of weather conditions with the number of collisions per month'
)

# Streamlit App
st.title("Summer Collisions and Weather Visualizations")

# Display the Altair charts.
st.altair_chart(chart1, use_container_width=True)
st.altair_chart(chart2, use_container_width=True)
st.altair_chart(chart3, use_container_width=True)
st.image(image, caption='Your Image Caption', use_column_width=True)
st.altair_chart(chart5, use_container_width=True)