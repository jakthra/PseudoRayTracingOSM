import numpy as np
import matplotlib.pyplot as plt



def model_comparison_barplot_labels(MSE_811, MSE_2630, std_811, std_2630, labels, name='model_comparison_bar_group_std_split'):
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
        plt.show()