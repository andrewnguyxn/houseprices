import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pydeck as pdk
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("housing.csv")

df = df.dropna()

#Transform variables to reduce skewedness
df['SRoom'] = np.sqrt(df['total_rooms'])
df['SBed'] = np.sqrt(df['total_bedrooms'])
df['SPop'] = np.sqrt(df['population'])
df['SHouse'] = np.sqrt(df['households'])
df['LIncome'] = np.log(df['median_income'])
df['SValue'] = np.sqrt(df['median_house_value'])

df_sample = df.sample(n=2000, random_state=42)

# Define the train-test split ratio
train_ratio = 0.7
test_ratio = 0.3

# Shuffle the DataFrame
df_sample = df_sample.sample(frac=1, random_state=42).reset_index(drop=True)

# Calculate the number of training samples
train_size = int(len(df_sample) * train_ratio)

# Split the DataFrame
train = df_sample[:train_size]
test = df_sample[train_size:]

house = pd.DataFrame(train)

# List of numeric columns
numeric_columns = [
    "longitude", "latitude", "housing_median_age", 
    "total_rooms", "total_bedrooms", "population", 
    "households", "median_income", "median_house_value"
]

# Title of the app
st.title('Histogram Slideshow')

# Slider for navigation
column = st.selectbox('Select column to display histogram', numeric_columns)

# Display the selected histogram based on the selected column
if column:
    fig, ax = plt.subplots()
    ax.hist(house[column], bins=30, color='turquoise', edgecolor='black')
    ax.set_title(f'Histogram of {column}')
    ax.set_xlabel(column)
    ax.set_ylabel('Frequency')

    # Display the plot in Streamlit
    st.pyplot(fig)

    # Optionally, display additional information
    st.write(f"Displaying histogram of {column}")

# Optionally, display the raw data table
if st.checkbox('Show raw data'):
    st.write(house)

lon_lat_df = house[['longitude', 'latitude']]


# Normalize the median_house_value data to the range [0, 1]
norm = mcolors.Normalize(vmin=df['median_house_value'].min(), vmax=df['median_house_value'].max())

# Apply a colormap (e.g., 'plasma') to the normalized data to get colors
colormap = cm.get_cmap('plasma')
df['color'] = df['median_house_value'].apply(lambda x: colormap(norm(x)))

# Convert RGBA colors to a format compatible with pydeck
df['color'] = df['color'].apply(lambda rgba: [int(255 * c) for c in rgba[:3]])

price_range = st.slider(
    'Select a price range',
    min_value=int(df['median_house_value'].min()),
    max_value=int(df['median_house_value'].max()),
    value=(int(df['median_house_value'].min()), int(df['median_house_value'].max()))
)

filtered_data = df[(df['median_house_value'] >= price_range[0]) & (df['median_house_value'] <= price_range[1])]

# Select longitude, latitude, and color columns
lon_lat_df = filtered_data[['longitude', 'latitude', 'color']]

# Title of the app
st.title('Heatmap Visualization Of Housing Prices')
st.markdown('More red indicates more expensive housing prices')

# Define a layer to display on a map
layer = pdk.Layer(
    'ScatterplotLayer',
    data=lon_lat_df,
    get_position='[longitude, latitude]',
    get_color='color',
    get_radius=10000,
)

# Set the viewport location
view_state = pdk.ViewState(
    longitude=lon_lat_df['longitude'].mean(),
    latitude=lon_lat_df['latitude'].mean(),
    zoom=5,
    pitch=0,
)

# Render the deck.gl map
r = pdk.Deck(layers=[layer], initial_view_state=view_state)
st.pydeck_chart(r)

x = train[['total_rooms', 'total_bedrooms', 'population', 'households', 'median_income']]
y = train['median_house_value']

model = LinearRegression()

st.header('Enter the details for a property price estimate:')
total_rooms = st.number_input('Total Rooms In Area', min_value=0)
bed = st.number_input('Total Bedrooms', min_value=0)
pop = st.number_input('Area Population', min_value=0)
household = st.number_input('Number Of Households', min_value=0)
income = st.number_input('Median Income', min_value=0)

if st.button('Predict'):
    input_data = pd.DataFrame({'total_rooms': [total_rooms], 'total_bedrooms': [bed],'population' : [pop], 'households' : [household, 'median_income': [income]})
    prediction = model.predict(input_data)
    st.subheader(f'Predicted Median House Value: ${prediction[0]:,.2f}')





