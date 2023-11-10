import numpy as np
import pandas as pd
import os

# Get the current directory
current_directory = os.path.dirname(os.path.abspath(__file__))

# Navigate to the main folder of the repository
main_folder = os.path.abspath(os.path.join(current_directory, "../../"))

# Load the file using the relative path from the main folder
data_folder = os.path.join(main_folder, "data")

if __name__ == '__main__':
    df = pd.read_excel(os.path.join(data_folder,  "AirQualityUCI.xlsx")) #replace with your own path
    
    groups = df.groupby([df['Date'].dt.year, df['Date'].dt.month])
    data_chunks = {}

    # Iterate over the groups and save them in the dictionary
    for (year, month), group in groups:
        key = f"{month:02d}_{year}"
        data_chunks[key] = group

    for key, value in data_chunks.items():
        value.to_csv(os.path.join(data_folder, f'data_splits/{key}.csv'), index=False)
        
    print(data_chunks.keys())
