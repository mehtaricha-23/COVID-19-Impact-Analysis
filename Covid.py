#Data Handling
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

 
# Machine Learning
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans


df = pd.read_csv(r"C:\Users\sar\OneDrive\Desktop\New folder\Covid-Data\owid-covid-data.csv")

# Keep only important columns
column_needed=[    'location', 'date', 'total_cases', 'new_cases',
    'total_deaths', 'new_deaths', 'total_tests', 
    'people_vaccinated', 'people_fully_vaccinated', 'population'
]
df=df[column_needed]


# Remove aggregate data (continents & world)
df = df[~df['location'].isin(['World', 'Asia', 'Africa', 'Europe', 'North America',
                              'South America', 'European Union', 'Oceania'])]

#convert date to datetime
df['date'] = pd.to_datetime(df['date'])

#Filling missing value with 0
print("Null Values Before Filling :")
print(df.isnull().sum())
df.fillna(0,inplace=True)

#Focus on india
df_india = df[df['location']== 'India']
df_india.set_index('date',inplace=True)


plt.figure(figsize=(10,6))
plt.plot(df_india.index,df_india['new_cases'],color='orange')
plt.title('Daily New COVID-19 Cases  In  India')
plt.xlabel('Date')
plt.ylabel('New Cases')
plt.grid(True)
plt.tight_layout()
plt.show()  


#Assuming 'top_10' DataFrame already created as:




# Convert columns
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['total_cases'] = pd.to_numeric(df['total_cases'], errors='coerce')

# Filter rows with total_cases > 0
df_valid = df.dropna(subset=['total_cases'])
df_valid = df_valid[df_valid['total_cases'] > 0]

# Get latest valid date
latest_date = df_valid['date'].max()
print("Latest valid date with data:", latest_date)

# Get data for that date
latest_df = df_valid[df_valid['date'] == latest_date]

# Optional: Remove regions
excluded_keywords = ['income', 'countries', 'World', 'Union']
latest_df = latest_df[~latest_df['location'].str.contains('|'.join(excluded_keywords), case=False)]

# Top 10 countries
top_10 = latest_df.sort_values(by='total_cases', ascending=False).head(10)
print(top_10[['location', 'total_cases']])

# Plot
plt.figure(figsize=(12, 6))
sns.barplot(data=top_10, x='location', y='total_cases', palette='viridis')
plt.title('Top 10 Countries by Total COVID-19 Cases', fontsize=16)
plt.xlabel('Country')
plt.ylabel('Total Cases')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.grid(True, axis='y')
plt.show()



ml_df = df[['people_vaccinated', 'total_cases']].dropna()
ml_df['people_vaccinated'] = pd.to_numeric(ml_df['people_vaccinated'], errors='coerce')
ml_df = ml_df[ml_df['people_vaccinated'] > 0]


X = ml_df[['people_vaccinated']]
y = ml_df['total_cases']

model = LinearRegression()
model.fit(X, y)

# Prediction line
predictions = model.predict(X)

plt.figure(figsize=(8, 5))
plt.scatter(X, y, color='blue', label='Actual')
plt.plot(X, predictions, color='red', label='Prediction')
plt.xlabel('People Vaccinated',fontsize =14)
plt.ylabel('Total Cases',fontsize = 14)
plt.title('Linear Regression: Vaccination vs Total Cases',fontsize = 14)
plt.legend()
plt.grid(True)
plt.show()

# Sample DataFrame 
cluster_df = df[['location', 'total_cases', 'total_deaths']]

# Filter rows with valid values
cluster_df = cluster_df[(cluster_df['total_cases'] > 0) & (cluster_df['total_deaths'] > 0)]

# Prepare data for clustering
X = cluster_df[['total_cases', 'total_deaths']]

# KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
cluster_df['cluster'] = kmeans.fit_predict(X)

# Plot using seaborn
plt.figure(figsize=(10, 6))
sns.scatterplot(data=cluster_df, x='total_cases', y='total_deaths', hue='cluster', palette='Set1')

plt.title('Clustering Countries by Cases and Deaths', fontsize=14)
plt.xlabel('Total Cases',fontsize = 14)
plt.ylabel('Total Deaths',fontsize =14)
plt.grid(True)
plt.tight_layout()
plt.show()

