# anonymize behavioral data
import random
import string
from pathlib import Path

import pandas as pd

# load data
data = pd.read_csv(Path("E:/expecon_ms/data/behav/prepro_behav_data_expecon1_2.csv"))


# create a random id consisting of 5 digits for
# each participant in the data
def random_id():
    return "".join(random.choice(string.digits) for _ in range(5))


# create a list of random ids


random_ids = [random_id() for _ in range(len(pd.unique(data.ID)))]

# now replace the initial IDs with the random IDs for each participant
data["ID"] = data["ID"].replace(pd.unique(data.ID), random_ids)

# drop the age column and the unnamed column
data = data.drop(["age", "Unnamed: 0"], axis=1)

# now we shuffle the participants data based on the randoms IDs but
# making sure that the trial data is still in the correct order
data = data.sort_values(["ID", "trial"])

# save the anonymized data
data.to_csv(Path("E:/expecon_ms/data/behav/prepro_behav_data_expecon1_2_anon.csv"), index=False)
