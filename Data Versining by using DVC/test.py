import pandas as pd
import os

# Correctly structured data
data = {
    "Name": ['mir', 'sadab', 'ali'],
    "Age": [18, 20, 21],
    "City": ['Kendrapara', 'Angul', 'Cuttack']
}

# Create a DataFrame
df = pd.DataFrame(data)

# Define the directory path
data_dir = r'C:\Users\alisa\OneDrive\Desktop\MLOPs Note\Data Versining by using DVC'

# Create the directory if it doesn't exist
os.makedirs(data_dir, exist_ok=True)  # Corrected function name

# Define the file path
file_path = os.path.join(data_dir, 'sample_data.csv')

# Save the DataFrame to a CSV file
df.to_csv(file_path, index=False)

print(f"Data saved to {file_path}")