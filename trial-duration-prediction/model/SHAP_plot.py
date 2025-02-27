import shap
import pickle
import pandas as pd
import matplotlib.pyplot as plt


def shap_plot(explainer_value):
    plt.figure(figsize=(12, 8))
    plt.subplots_adjust(left=0.3, right=1, top=0.9, bottom=0.1)
    shap.plots.waterfall(explainer_value, 14)
    shap.plots.force(explainer_value, matplotlib=True)
    plt.show()


def shap_save(shap_values):
    data_df = shap_values.data
    for i in range(len(data_df)):
        values = shap_values.values[i]
        for j in range(len(data_df.columns)):
            tmp = 'shap: ' + str(values[j]) + '\n'
            data_df.iloc[i, j] = tmp + str(data_df.iloc[i, j])
    data_df.to_csv(file_road + "shap_values.csv", index=True)


if __name__ == '__main__':
    # with open('../data/shap_values.pkl', 'rb') as shap_file:
    #     shap_values = pickle.load(shap_file)
    # file_road = "E:\\task\\大四\\谈\\毕设\\checkpoint_store\\09\\"
    file_road = "../data/"
    with open(file_road + "shap_values.pkl", 'rb') as shap_file:
        shap_values = pickle.load(shap_file)
    shap.initjs()

    explainer_values = []
    for i in range(len(shap_values)):
        explainer_value = shap.Explanation(values=shap_values.values[i], base_values=shap_values.base_values[i],
                                           feature_names=shap_values.feature_names)
        explainer_values.append(explainer_value)
    shap_plot(explainer_values[0])
    # shap_save(shap_values)
    plt.subplots_adjust(left=0.3, right=1, top=0.9, bottom=0.1)
    shap.plots.beeswarm(shap_values, max_display=14)
