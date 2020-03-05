import numpy as np
import matplotlib.pyplot as plt
import json


def model_comparison_barplot_labels(MSE_811, MSE_2630, std_811, std_2630, labels, name='model_comparison_access'):
    x = np.arange(len(labels))
    width = 0.35

    with plt.style.context('seaborn-dark'):
        fig = plt.figure(figsize=(7, 4))
        ax = plt.gca()
        ax.bar(x - width/2, MSE_811, width, alpha=0.8, ecolor='black', capsize=10, label="811 MHz", yerr=std_811)
        ax.bar(x + width/2, MSE_2630, width, alpha=0.8, ecolor='black', capsize=10, label="2630 MHz", yerr=std_2630)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=10)
        ax.set_ylabel('RMSE [dB]')
        plt.legend()
        plt.tight_layout()
        plt.grid()
        plt.savefig('results/{}.eps'.format(name))
        plt.savefig('results/{}.png'.format(name))
        plt.show()


if __name__ == "__main__":

    access_model_name = '63b17fa0-89e0-440b-952c-aa7f2e63e49c'
    with open("results/evaluations/"+access_model_name+"_results.json") as file:
        access_model = json.load(file)


    model_name = '20ca9ba4-64ef-45a8-8a42-0f1308d9cdce'
    with open("results/evaluations/"+model_name+"_results.json") as file:
        results_model = json.load(file)

    MSE_811 = results_model['RMSE_811']
    MSE_2630 = results_model['RMSE_2630']

    MSE_811_access = access_model['RMSE_811']
    MSE_2630_access = access_model['RMSE_2630']

    model_comparison_barplot_labels([MSE_811, MSE_811_access], [MSE_2630, MSE_2630_access], [0, 0], [0, 0], ['OSM picture', 'Satellite images'])

