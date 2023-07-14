import pandas as pd
import os
import random

# provide a list with the participants with MetaboAge that should only
# take part in the testing set
participants_with_metabo = []

# Import the dataset
df = pd.read_spss(
    "path/to/dataset",
    convert_categoricals=False,
)

# Filter the participants that should be excluded
filter = (df["B1_VD1_2.6.10"]!=1) & (df["CVA_Rose"]!=1 & df["VISIT3_DATE"].notnull()) & (df["VISIT1_DATE"].notnull())
df_subset = df[filter]

# Rename the necessary columns
mapping = {
    "Age": "age",
    "deelnemer_id": "id",
    "SEX": "sex",
    "CVA_Rose": "stroke",
    "N_GTS_WHO": "diabetes",
    "1_VD1_2.6.10": "parkinson",
    "VISIT3_DATE": "visit3",
    "VISIT1_DATE": "visit1",
    "MRI_lagtime": "lagtime"
}
df_subset = df_subset.rename(columns=mapping)

ids = df_subset['id'].values
# Calculate the age at  the MRI visit
age = (df_subset["age"] + ((df_subset["VISIT3_DATE"] - df_subset["VISIT1_DATE"]).dt.days)/365.25 + df_subset["MRI_lagtime"].fillna(0)).values
sex = df_subset['sex'].values
diabetes = df_subset['diabetes'].values

# To avoid going through the imaging folder everytime, provide a list with the ids that 
# have imaging data
idx = []
ids_with_imaging_data = [int(code) for code in list(df_subset.loc[idx]["deelnemer_id"].values)]
idx_included = []
is_training_data = []
for index, id in enumerate(ids):
    # In alternative, it's possible to check if the necessary imaging file exists
    # if os.path.isfile(f"/path/to/scans/{str(int(id))}_aseg_GM_to_template_GM_mod.nii.gz"):
    if id in ids_with_imaging_data:
        idx_included.append(index)
        if id in participants_with_metabo:
                is_training_data.append(0)
        else:
                is_training_data.append(1)

print("Number of participants included for training/validation and testing:")
print(is_training_data.count(0))
print(is_training_data.count(1))

result = pd.DataFrame({
    "id": ids[idx_included],
    "clinical_id": ids[idx_included],
    "imaging_id": ids[idx_included],
    "age": age[idx_included].round(1),
    "sex": sex[idx_included]-1,
    "is_training_data": is_training_data,
})

result.to_csv("dataset.csv")
