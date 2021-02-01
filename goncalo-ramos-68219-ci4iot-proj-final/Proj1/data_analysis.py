import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scikitplot as skplt


def raw_data(df):

    print("DF Raw data:\n", df)
    print("DF info:\n", df.info())
    print("DF types:\n", df.dtypes)


def correlation_graph(df):

    sns.set(style="ticks")
    sns.pairplot(df, hue="Falha")
    plt.show()


def scatterplot_graph(df):

    sns.relplot(x='Requests', y='Load', hue='Falha', data=df)


def multi_plot(df, val):
    groups = [0, 1, 2]
    i = 1
    #plot each column
    plt.figure()
    for group in groups:
        plt.subplot(len(groups), 1, i)
        plt.plot(val[:, group])
        plt.title(df.columns[group], y=0.5, loc='right')
        i += 1
    plt.show()


def data_analysis():

    print("Starting IoTGatewayCrash..\n")

    dataset = pd.read_csv('lab6-7-8_IoTGatewayCrash.csv')
    values = dataset.values

    raw_data(dataset)
    correlation_graph(dataset)
    scatterplot_graph(dataset)
    multi_plot(dataset, values)


data_analysis()

