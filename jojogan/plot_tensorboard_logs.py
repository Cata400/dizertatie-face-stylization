import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import tensorboard as tb


log_name = 'arnold+arcane_jinx2_ALPHA=1_PC=False_20230605-221755'

experiment_id = 'asnmwrcaT72OhB612EugqQ'
experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
df = experiment.get_scalars()

print(df)

interest_df = df[df['run'].str.contains("arnold")]

print(interest_df.columns)
for col in interest_df.columns[:-1]:
    print(col, interest_df[col].unique())
    

loss_df = interest_df[interest_df['tag'].str.contains('Loss')]

labels = ['Arnold + Arcane Jinx, Alpha = 0.25', 'Arnold + Arcane Jinx, Alpha = 0.5', 'Arnold + Arcane Jinx, Alpha = 0.75', 'Arnold + Arcane Jinx, Alpha = 0', 'Arnold + Arcane Jinx, Alpha = 1']

plt.figure(figsize=(16, 6))
ax_loss = sns.lineplot(loss_df, x='step', y='value', hue='run')
ax_loss.set_title('Loss')
ax_loss.set(xlabel='Epoch', ylabel='Value')
handles, _ = ax_loss.get_legend_handles_labels()
ax_loss.legend(handles=handles, labels=labels, fontsize='large')
# plt.grid()

plt.show()
