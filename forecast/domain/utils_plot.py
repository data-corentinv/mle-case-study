import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

def plot_features_importance(data, importances, n_feat):
    """Plot the features importance barplot.

    Parameters
    ----------
    data : pd.DataFrame
        data containing colnames used in the model.

    importances : np.ndarray
        list of feature importances

    n_feat : int
        number of features to plot

    """
    indices = np.argsort(importances)[::-1]
    features = data.columns

    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances[indices][:n_feat],
                y=features[indices][:n_feat], palette='Blues_r')
    plt.title("Top {} Features Importance".format(n_feat))
    return plt.show()
