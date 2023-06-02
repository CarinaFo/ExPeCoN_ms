# Make the necessary imports
import pandas as pd
from ydata_profiling import ProfileReport

# save the preprocessing data
behavpath = 'D:\\expecon_ms\\data\\behav\\behav_df\\'

# Load the data
df = pd.read_csv(f"{behavpath}prepro_behav_data.csv", na_values='?')
df.drop(df.columns[0], axis=1, inplace=True)

# Generate the report
profile = ProfileReport(df,title="ExpecoN")

# Save the report to .html
profile.to_file(f"{behavpath}ExpecoN.html")

