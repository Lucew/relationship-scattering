import os

import seaborn as sns
import pandas as pd
import sklearn.manifold as skm
import matplotlib.pyplot as plt
import matplotlib

SAVE = False

# use pgf backed
if SAVE:
    matplotlib.use("pgf")
    matplotlib.rcParams.update({
        "pgf.texsystem": "pdflatex",
        'font.family': 'serif',
        'text.usetex': True,
        'pgf.rcfonts': False,
    })

# load the power envelope correlation
plant = "ALBUR"
df = pd.read_csv(os.path.join(r"PLANT_ESST_1d_cp_correlation_window_1.csv"), index_col=0)
sensors = list(df.columns)

# make the tsne embedding
tsne = skm.TSNE(metric='precomputed', init='random', perplexity=25, random_state=42)
transf = tsne.fit_transform(1-df.to_numpy())

# put into a dataframe
df = pd.DataFrame(transf, columns=["x", "y"])
df["Turbine"] = [ele[2:3] for ele in sensors]
df["Block"] = [ele[1:2] for ele in sensors]

# make the plot
plt1 = sns.scatterplot(df, x="x", y="y", hue="Turbine", style="Block", s=80)
# Put a legend below current axis
plt1.legend(loc='upper center', bbox_to_anchor=(1, 1.15), framealpha=1)
plt1.axes.get_xaxis().set_visible(False)
plt1.axes.get_yaxis().set_visible(False)
plt.tight_layout()
if SAVE:
    plt.savefig('scattering.pgf')
else:
    plt.show()

