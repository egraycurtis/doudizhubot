import psycopg2
import pandas as pd
import matplotlib.pyplot as plt

conn = psycopg2.connect("postgresql://postgres:password@localhost:5432")
sql = "SELECT created_at, results FROM competitions where id > 520 ORDER BY id DESC;"
df = pd.read_sql(sql, conn)
conn.close()

df['deep'] = df['results'].apply(lambda x: x['deep'])
df['transformer'] = df['results'].apply(lambda x: x['transformer'])
df['created_at'] = pd.to_datetime(df['created_at'])
df.set_index('created_at', inplace=True)
df.sort_index(inplace=True)

df['difference'] = df['transformer'] - df['deep']
window_size = 200
df['difference_avg'] = df['difference'].rolling(window=window_size).mean()

plt.figure(figsize=(10, 6))
plt.plot(df['difference_avg'], label='Transformer - Deep Avg', color='green')
plt.title('Trailing Average Difference (Transformer - Deep) Over Time')
plt.xlabel('Date')
plt.ylabel('Average Difference')
plt.legend()
plt.grid(True)
plt.show()
