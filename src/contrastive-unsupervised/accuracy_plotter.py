#%%
import pandas as pd
import seaborn as sns
sns.set_style("darkgrid")
# %%

data = pd.DataFrame({"Time Skip Range": [0, 10, 25, 50, 75, 100], r"Stream51$\rightarrow$Stream51": [44.16, 47.41, 52.44, 52.56, 51.88, 51.42], r"Stream51$\rightarrow$CIFAR10": [61.55, 64.44, 68.39, 69., 69.06, 69.1]})
data = pd.melt(data, id_vars="Time Skip Range", value_vars=[r"Stream51$\rightarrow$Stream51", r"Stream51$\rightarrow$CIFAR10"], value_name="Accuracy", var_name="Dataset")
data

#%%

fig = sns.lineplot(data=data, x="Time Skip Range", y="Accuracy", hue="Dataset", legend=False, style="Dataset")
sns.scatterplot(data=data, ax=fig, x="Time Skip Range", y="Accuracy", hue="Dataset", style="Dataset")
fig.set_title("SimSiam Test Accuracy")
# %%

# %%

data = pd.DataFrame({"Time Skip Range": [0, 10, 25, 50, 75, 100], r"UCF101$\rightarrow$UCF101": [17.4, 37.8, 37.5, 27.75, 35.8, 33.68], r"UCF101$\rightarrow$CIFAR10": [35.61, 47.84, 43.32, 33.35, 41.94, 41.09]})
data = pd.melt(data, id_vars="Time Skip Range", value_vars=[r"UCF101$\rightarrow$UCF101", r"UCF101$\rightarrow$CIFAR10"], value_name="Accuracy", var_name="Dataset")
data
# %%

fig = sns.lineplot(data=data, x="Time Skip Range", y="Accuracy", hue="Dataset", legend=False, style="Dataset")
sns.scatterplot(data=data, ax=fig, x="Time Skip Range", y="Accuracy", hue="Dataset", style="Dataset")
fig.set_title("SimSiam Test Accuracy")
# %%
