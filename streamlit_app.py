import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pydeck as pdk
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from sklearn.linear_model import LinearRegression

df = pd.read_csv("housing.csv")

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
train['color'] = train['median_house_value'].apply(lambda x: colormap(norm(x)))

# Convert RGBA colors to a format compatible with pydeck
train['color'] = train['color'].apply(lambda rgba: [int(255 * c) for c in rgba[:3]])

# Select longitude, latitude, and color columns
lon_lat_df = train[['longitude', 'latitude', 'color']]

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

x = train[['SRoom', 'SBed', 'SPop', 'SHouse', 'LIncome']]
y = train['SValue']

model = LinearRegression()

model.fit(x,y)

mse = mean_squared_error(y_test, y_pred)

# Calculate R-squared
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')



