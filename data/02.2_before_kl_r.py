import pandas as pd



df =pd.read_pickle("/Users/kang/CMEM/data/01.1_raw/A.pkl")
resampled_df = df[['date','timeHMs','qty']]

# Step 2: Pivot the DataFrame to get the desired format
pivot_df = resampled_df.pivot_table(index=resampled_df.index.time, columns=resampled_df.index.date, values='qty')
# Step 3: Create a new DataFrame with formatted column names
new_columns = [col.strftime('%Y-%m-%d') for col in pivot_df.columns]
pivot_df.columns = new_columns

# Step 4: Add the 'AM' or 'PM' label to the index
pivot_df.index = [f'{t.strftime("%I:%M %p")}' for t in pivot_df.index]

# Step 5: Save the DataFrame to a CSV file
pivot_df.to_csv("data_for_r.csv")

# Optionally, display the DataFrame
# print(pivot_df)

'/Users/kang/CMEM/data_for_r.csv'
