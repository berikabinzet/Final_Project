import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.insert(1, 'data_management')
from data_management import df
from regressions import result_tr

### Summary statistics
print(df[["earnings", "weeks_worked", "age", "hours_worked","gender"]].describe())

### Heatmap for correlation between variables 
df2 = df.dropna()
df2['female'] = np.where(df2["gender"] == 2, 1, 0)
subscales_selection = ["female","earnings","participation","any_children","hours_worked","weeks_worked"]
subscale_items_columns = []

for subscale in subscales_selection:
    subscale_items = list(filter(lambda x: x.startswith(subscale), df2.columns))
    subscale_items_columns = subscale_items_columns + subscale_items

subscale_items_columns

# Calculating correlation matrix 

corr_matrix = df2[subscale_items_columns].corr()

# Creating the figure

mask = np.zeros_like(corr_matrix)
mask[np.triu_indices_from(mask)] = True
fig_heatmap, axs = plt.subplots(figsize=(15, 15))
axs = sns.heatmap(
    corr_matrix, annot=True, fmt=".2f", vmin=-1, vmax=1, mask=mask, square=True, center=0
)
plt.show()
fig_heatmap.savefig("../bld/Corr_Heatmap.png")

### Regression Plots

# Plotting participation variable
# gca stands for 'get current axis'
ax = plt.gca()
result_tr.plot(kind="line",x="event_time",y="participation_male",ax=ax)
result_tr.plot(kind="line",x="event_time",y="participation_female", color="red", ax=ax )
plt.xticks(np.arange(-5, 11, 1.0))
plt.axvline(x=-0.5, ymin = -1000, ymax = 100, color="black")
plt.show()
plt.savefig("../bld/participation_plot.png")


# Plotting hours_worked variable
ax = plt.gca()
result_tr.plot(kind="line",x="event_time",y="hours_worked_male",ax=ax)
result_tr.plot(kind="line",x="event_time",y="hours_worked_female", color="red", ax=ax )
plt.xticks(np.arange(-5, 11, 1.0))
plt.axvline(x=-0.5, ymin = -1000, ymax = 100, color="black")
plt.show()
plt.savefig("../bld/hours_worked_plot.png")

# Plotting earnings variable
ax = plt.gca()
result_tr.plot(kind="line",x="event_time",y="earnings_male",ax=ax)
result_tr.plot(kind="line",x="event_time",y="earnings_female", color="red", ax=ax )
plt.xticks(np.arange(-5, 11, 1.0))
plt.axvline(x=-0.5, ymin = -1000, ymax = 100, color="black")
plt.show()
plt.savefig("../bld/earnings_plot.png")