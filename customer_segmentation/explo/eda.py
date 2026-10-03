import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd

    return (pd,)


@app.cell
def _(pd):
    # From : https://www.kaggle.com/datasets/abisheksudarshan/customer-segmentation
    train = pd.read_csv("../data/train.csv")
    return (train,)


@app.cell
def _(train):
    train.columns
    train.describe(include="all")
    return


@app.cell
def _():
    def get_discrete_bins(col):
        return int(col.max() - col.min()) + 1

    def barplot_hist(col):
        nbins = get_discrete_bins(col)
        return col.hist(bins=nbins)
    

    return (barplot_hist,)


@app.cell
def _(barplot_hist, train):
    barplot_hist(train["Age"])
    return


@app.cell
def _(barplot_hist, train):
    barplot_hist(train["Work_Experience"])

    return


@app.cell
def _(barplot_hist, train):
    barplot_hist(train["Family_Size"])
    return


@app.cell
def _(pd):
    test = pd.read_csv("../data/test.csv")
    return (test,)


@app.cell
def _(train):
    def display_freqs(col):
        vc = col.value_counts()
        print(vc)
        print(vc / len(col))

    display_freqs(train["Gender"])
    display_freqs(train["Ever_Married"])
    display_freqs(train["Graduated"])
    display_freqs(train["Profession"])
    display_freqs(train["Spending_Score"])
    display_freqs(train["Var_1"])
    display_freqs(train["Segmentation"])

    return (display_freqs,)


@app.cell
def _():
    # Look for new categorical values so our OHE stratefy makes sense
    return


@app.cell
def _(test):
    test.describe(include="all")
    return


@app.cell
def _(display_freqs, test):
    display_freqs(test["Gender"])
    display_freqs(test["Ever_Married"])
    display_freqs(test["Graduated"])
    display_freqs(test["Profession"])
    display_freqs(test["Spending_Score"])
    display_freqs(test["Var_1"])
    return


@app.cell
def _(train):
    train.select_dtypes(include=['float64', 'Int64']).columns
    return


@app.cell
def _(train):
    train.select_dtypes(include=['object']).columns
    return


if __name__ == "__main__":
    app.run()
