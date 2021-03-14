#Importing libraries and packages
import pandas as pd
import numpy as np
import zipfile
import re



# Loading dataset and its information document

nlsy_data_raw = pd.read_csv("original_data/child_ineq_data.csv")
var_info_raw = pd.read_excel("original_data/var_info.xlsx")

# Checking for duplicates in nlsy_name

if len(var_info_raw) == len(var_info_raw["nlsy_name"].value_counts()):
    print("nlsy_name is unique")
else:
    print("nlsy_name is not unique")

# Combining readable_name with survey_year

var_info = var_info_raw
var_info = var_info.astype({'survey_year':'str'})
var_info["readable_name_year"] = var_info["readable_name"] + "_" + var_info["survey_year"]

# Changing the time-invariant readable variable names

var_info["readable_name_year"] = var_info["readable_name_year"].str.replace("_invariant","")

# Defining a function to detect and replace the missing data, which is given as negative values

def negative_as_missing(value):
    """Returns pandas missing for negative values
    
    Args:
    value (string/float/integer): any value
    
    Returns:
    (string/float): pandas missing if value is negative; else value itself
    
    """
    
    if type(value) != str:
        if value < 0:
            out = np.nan
        else:
            out = value
    else:
        out = value
    return out

# Applying missing value function to the dataset
nlsy_df = nlsy_data_raw.applymap(negative_as_missing)

# Renaming all variables with readable names
name_dict = dict(zip(var_info["nlsy_name"], var_info["readable_name_year"]))
nlsy_df.rename(columns=(name_dict), inplace=True)

# Keeping only those variables that are present in NLSY variable information file
droplist = [i for i in nlsy_df.columns if i not in set(var_info["readable_name_year"])]
nlsy_df.drop(droplist,axis=1,inplace=True)

# Changing individual_id into integer (is index variable)
nlsy_df["individual_id"] = nlsy_df["individual_id"].astype(int)

# Changing the format of the dataset from wide to long 
varnames_long = set(list(map(lambda x: re.sub("\_[0-9]{4}$","",x), list(nlsy_df.columns)))[3::])
nlsy_df_long = pd.wide_to_long(df=nlsy_df,stubnames = varnames_long,i=["individual_id"],j="year",sep = "_")

# Changing the values in the column for labor market participatiion variable such that it turns out to be 1 if the individual participate, 0 if not
# In the dataset, 1 - working, 2 - with job not at work, 3 - unemployed, 4 - keeping house, 5 - going to school, 6 - unable to work, 7 - other, 8 - in active forces
nlsy_df_long["participation"] = nlsy_df_long["participation"].replace(list([4,5,6,7]),0)
nlsy_df_long["participation"] = nlsy_df_long["participation"].replace(list([2,3,8]),1)

# Loading child data including the year of the first born child

child_data_raw = pd.read_csv("original_data/child_data.csv")
child_var_info = pd.read_excel("original_data/child_var_info.xlsx")

# Changing the names of variables
name_dict = dict(zip(child_var_info["nlsy_name"], child_var_info["readable_name"]))

# Adapting the names in the data file accordingly
child_data_raw.rename(columns=(name_dict), inplace=True)

# Changing data types
child_data_raw = child_data_raw.astype({"individual_id":"int", "first_child":"int"})
# Setting individual_id as the index
child_data_raw.set_index(["individual_id"], inplace=True)

# Applying missing value function to the dataset and dropping any NaN
child_data = child_data_raw.applymap(negative_as_missing)
child_data = child_data.dropna()

# Merging two dataframe, using individual_id and year as indexes
df = pd.merge(nlsy_df_long, child_data, how="right", on=["individual_id"], right_index = True)

# Creating variables related to the birth of the first child
df = df.reset_index()
df["first_child_birth"] = np.where(df["first_child"] == df["year"], 1, 0)
df["event_time"] = df["year"] - df["first_child"]
df["any_children"] = np.where(df["event_time"] < 0, 0, 1)
df = df.drop(["first_child"], axis=1)

# Filling missing data in age 
s = df.groupby("individual_id").age.cumcount()
s1 = (df.age - s).groupby(df.individual_id).transform("first")
df["age"] = s1 + s

# Setting indices again
df.set_index(["individual_id","year"], inplace=True)

# Changing datatypes
df = df.astype({"earnings":"Int64", "age":"Int64","gender":"Int64",
                "any_children":"Int64","participation":"Int64",
                "weeks_worked":"Int64","hours_worked":"Int64","event_time":"Int64"})

# Restricting the dataset 5 years before and 10 years after the birth of first child
indexNames = df[(df["event_time"] > 10) | (df["event_time"] < -5)].index
indexNames
df.drop(indexNames , inplace=True)

# Generating the event_time dummies (where t=-1 is omitted, implying that the event time coefficients measure
# The impact of children relative to the year just before the first child birth
dummies = pd.get_dummies(df["event_time"])
dummies.columns = [ "event_time_" + str(dummies) for dummies in range(1,17) ]
dummies = dummies.drop(["event_time_5"], axis=1)
df = pd.concat([df, dummies], axis=1)

# Creating two datasets for female and male
df_female = df[df["gender"]==2]
df_male = df[df["gender"]==1]
