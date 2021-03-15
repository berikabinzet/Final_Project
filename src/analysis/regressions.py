import pandas as pd
import numpy as np
#pip install linearmodels
from linearmodels import PanelOLS
import statsmodels.formula.api as smf
import sys
sys.path.append("src/data_management")
from data_management import df
from data_management import df_female
from data_management import df_male

# Creating lists for dependent variable, and results for each gender 
dependent = ["hours_worked","participation","earnings"] 
result_male = pd.DataFrame(data = {'dependent': ["hours_worked","participation","earnings"]})
result_female = pd.DataFrame(data = {'dependent': ["hours_worked","participation","earnings"]})

# Defining a function to create balanced datasets
def gen_balance_df(data, value):
    """A function to create balanced datasets, which is a dataset that has no missing data for the specific variable
    
    Args:
    data (dataset): df_female for female balanced datasets; df_male for male balanced datasets
    value (string): "hours_worked","participation","earnings"
    
    Returns:
    (dataset): returns a dataset which has no missing data for the specific value
    
    """
    df_1 = data.groupby(level = 0)[[value]].apply(lambda x: x.isna().sum())
    df_balance = df.loc[df_1.loc[df_1[value]== 0].index]
    df_balance = df_balance.astype(float)
    return df_balance

# Resetting the index to be able to use year as fixed effect in the regression
df = df.reset_index()

# Defining a regression function and result dataframe for each event_time variable
def get_reg_result (data, dependent, result_df ):
    """A function for running regressions for separate dependent variables
    
    Args:
    data (dataset): df_female for female results; df_male for male results
    value (string): "hours_worked","participation","earnings"
    result_df (dataset) : the coefficients for each dummy variable for event_time
    
    Returns:
    result_df (dataset) : the coefficients for each dummy variable for event_time
    
    """
    form = dependent + "~  C(age) + C(year) + event_time_1 + event_time_2 + event_time_3 + event_time_4 + event_time_6 + event_time_7 + event_time_8 +event_time_9 + event_time_10 + event_time_11 + event_time_12 + event_time_13 + event_time_14 + event_time_15 + event_time_16"
    mod = smf.ols(formula=form, data = data)
    res = mod.fit()

    result_df.loc[ (result_df["dependent"] == dependent)   , "β1"] = res.params["event_time_1"].round(3)
    result_df.loc[ (result_df["dependent"] == dependent)   , "β2"] = res.params["event_time_2"].round(3)
    result_df.loc[ (result_df["dependent"] == dependent)   , "β3"] = res.params["event_time_3"].round(3)
    result_df.loc[ (result_df["dependent"] == dependent)  , "β4"] = res.params["event_time_4"].round(3)
    result_df.loc[ (result_df["dependent"] == dependent)   , "β6"] = res.params["event_time_6"].round(3)
    result_df.loc[ (result_df["dependent"] == dependent)  , "β7"] = res.params["event_time_7"].round(3)
    result_df.loc[ (result_df["dependent"] == dependent)  , "β8"] = res.params["event_time_8"].round(3)
    result_df.loc[ (result_df["dependent"] == dependent)  , "β9"] = res.params["event_time_9"].round(3)
    result_df.loc[ (result_df["dependent"] == dependent)   , "β10"] = res.params["event_time_10"].round(3)
    result_df.loc[ (result_df["dependent"] == dependent)  , "β11"] = res.params["event_time_11"].round(3)
    result_df.loc[ (result_df["dependent"] == dependent)   , "β12"] = res.params["event_time_12"].round(3)
    result_df.loc[ (result_df["dependent"] == dependent)   , "β13"] = res.params["event_time_13"].round(3)
    result_df.loc[ (result_df["dependent"] == dependent)  , "β14"] = res.params["event_time_14"].round(3)
    result_df.loc[ (result_df["dependent"] == dependent)  , "β15"] = res.params["event_time_15"].round(3)
    result_df.loc[ (result_df["dependent"] == dependent)  , "β16"] = res.params["event_time_16"].round(3)

# Creating balanced datasets and running regressions for female balanced datasets 
for i in dependent:
        df_balance = gen_balance_df (df_female , i)
        get_reg_result(df_balance, i,  result_female)

# Creating balanced datasets and running regressions for male balanced datasets 
for i in dependent:
        df_balance = gen_balance_df (df_male , i)
        get_reg_result(df_balance, i,  result_male)

# Renaming the results properly to be able to distinguish between male and female
result_male["dependent"].replace({"hours_worked": "hours_worked_male",
                                "participation": "participation_male","earnings": "earnings_male" }, inplace=True)
result_female["dependent"].replace({"hours_worked": "hours_worked_female",
                                "participation": "participation_female","earnings": "earnings_female" }, inplace=True)

# Appending male and female results in one dataset
result = result_male.append(result_female, ignore_index = True)

# Changing dataset to make it easier to plot
result_tr = result.transpose()
result_tr.columns = result_tr.iloc[0]
result_tr = result_tr.drop('dependent',axis=0)

# Adding back event_time_5 which was dropped before
β5 = pd.DataFrame({"hours_worked_male":0, "participation_male": 0, "earnings_male": 0,"hours_worked_female":0, "participation_female": 0, "earnings_female": 0 }, index=["β5"])
result_tr = pd.concat([result_tr.iloc[:4],β5,result_tr.iloc[4:]] )
result_tr["event_time"] = [-5, -4 , -3, -2, -1 ,0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

