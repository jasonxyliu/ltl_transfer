import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

rc = {'axes.labelsize': 24, 'axes.titlesize': 32, 'legend.fontsize': 18, 'legend.title_fontsize': 18, 'xtick.labelsize': 20, 'ytick.labelsize': 20}


if __name__ == '__main__':
    slip = {
        0: [0.928, 0.898, 0.908, 0.878],
        0.1: [0.928, 0.888, 0.908, 0.908],
        0.2: [0.918, 0.898, 0.888, 0.918],
        0.3: [0.908, 0.877, 0.867, 0.908],
        0.4: [0.908, 0.847, 0.857, 0.857]
    }
    data, idx =  {}, 0

    for slip_prob, success_rates in slip.items():
        for success_rate in success_rates:
            data[idx] = {
                'Slip Probability': slip_prob,
                'Success Rate': success_rate
            }
            idx += 1

    data = pd.DataFrame.from_dict(data, orient='index')

    with sns.plotting_context('poster',rc=rc):
        plt.figure(figsize=[10*0.85, 8*0.85])
        sns.lineplot(data=data, x='Slip Probability', y='Success Rate', marker='o')
        plt.xticks([0,0.1, 0.2, 0.3, 0.4,])
        plt.savefig('stochastic_success.jpg', dpi=400, bbox_inches='tight')
