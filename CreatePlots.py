import os

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_dimensionality_reduction(save_figure=False):

    # load the results into memory
    results = pd.read_parquet("Result_1742297640.7004316.parquet")
    # print(results.columns)

    # drop non-metric MDS
    results = results[results["Method"] != "Non-Met. MDS"]
    results.loc[results["Method"] == "Met. MDS", "Method"] = "MDS"

    # get the trustworthiness for six neighbors
    results["Trustworthiness"] = results["Trustworthiness"].apply(lambda x: x[1][1])

    # get the dataset name from the filename
    results["Dataset"] = results["File"].apply(lambda x: os.path.split(x)[-2].split("_")[-1])

    # check whether we want to save
    if save_figure:
        # use pgf backed
        matplotlib.use("pgf")
        matplotlib.rcParams.update({
            "pgf.texsystem": "pdflatex",
            'font.family': 'serif',
            'text.usetex': True,
            'pgf.rcfonts': False,
        })

    # make a seaborn bar plot
    order = ["keti", "soda", "rotary", "plant1", "plant2", "plant3", "plant4"]
    plt.figure()
    plt1 = sns.boxplot(results, x="Method", y="Trustworthiness", hue="Dataset", hue_order=order)
    plt1.set_ylim([0.3, 1])
    plt1.legend(ncol=len(order)//2+1, framealpha=1, loc="lower left")
    [plt1.axvline(x + .5, color='k',linestyle="--") for x in plt1.get_xticks()]
    plt.grid(axis="y")
    if save_figure:
        plt.savefig('trustworthiness.pgf')

    plt.figure()
    plt2 = sns.boxplot(results, x="Method", y="Triplet Accuracy", hue="Dataset", hue_order=order)
    plt2.set_ylim([0.3, 1])
    [plt2.axvline(x + .5, color='k', linestyle="--") for x in plt2.get_xticks()]
    plt2.legend(ncol=len(order)//2+1, framealpha=1, loc="lower left")
    plt.grid(axis="y")
    if save_figure:
        plt.savefig('triplet_accuracy.pgf')
    else:
        plt.show()


def main():
    plot_dimensionality_reduction(False)


if __name__ == "__main__":
    main()
