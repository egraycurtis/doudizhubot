import psycopg2
import pandas as pd
import matplotlib.pyplot as plt


# Connect to your database
conn = psycopg2.connect("postgresql://postgres:password@localhost:5432")

# SQL query to fetch the data
sql = "SELECT created_at, results FROM competitions ORDER BY id DESC limit 700;"

# Load data into a DataFrame
df = pd.read_sql(sql, conn)

# Close the database connection
conn.close()

# Parse the JSON results into separate columns
df['deep'] = df['results'].apply(lambda x: x['deep'])
df['transformer'] = df['results'].apply(lambda x: x['transformer'])

# Convert created_at to datetime
df['created_at'] = pd.to_datetime(df['created_at'])

# Set created_at as the index
df.set_index('created_at', inplace=True)

# Sort by datetime index
df.sort_index(inplace=True)

# Calculate the trailing averages (you can adjust the window size)
window_size = 50  # Adjust this to change the smoothing level
df['deep_avg'] = df['deep'].rolling(window=window_size).mean()
df['transformer_avg'] = df['transformer'].rolling(window=window_size).mean()

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(df['deep_avg'], label='Deep Avg', color='blue')
plt.plot(df['transformer_avg'], label='Transformer Avg', color='red')
plt.title('Trailing Average Scores Over Time')
plt.xlabel('Date')
plt.ylabel('Average Score')
plt.legend()
plt.grid(True)
plt.show()
