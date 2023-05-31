import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np


# A pie chart for showing each class percentage
def plot_pie_chart(df, file_name):
    plt.figure(figsize=(14, 7))
    pie_chart = plt.Circle((0, 0), 0.68, color='white')
    values = df[187].value_counts()
    labels = values.index.map({
        0: 'Normal beat',
        1: 'Unclassifiable beat',
        2: 'Premature ventricular contraction',
        3: 'Supraventricular premature beat',
        4: 'Fusion of ventricular and normal beat'
    })
    plt.pie(values, labels=labels, autopct='%1.1f%%')
    plot = plt.gcf()
    plot.gca().add_artist(pie_chart)
    plt.savefig(os.path.join(dir_path, 'reports', 'figures', f'pie_chart_{file_name}.png'))  # Save the image
    #plt.show()



# Visualization of ECG signals for each type
class EcgSignalPlotter:
    def __init__(self, train_data, sample_size=10):
        self.train_data = train_data
        self.sample_size = sample_size
        self.label_dict = {
            0: "Normal",
            1: "Artial Premature",
            2: "Premature ventricular contraction",
            3: "Fusion of ventricular and normal",
            4: "Fusion of paced and normal"
        }
        self.fig, self.axs = plt.subplots(5 ,1, figsize=(13, 12))

    def get_samples(self):
        self.samples = [self.train_data.loc[self.train_data[187] == cls].sample(self.sample_size) for cls in range(5)]
        self.titles = [self.label_dict[cls] for cls in range(5)]

    def plot_ecg_signals(self):
        with plt.style.context("seaborn-white"):
            for i in range(5):
                ax = self.axs.flat[i]
                ax.plot(self.samples[i].values[:, :-2].transpose())
                ax.set_title(self.titles[i])
            plt.tight_layout()
            plt.suptitle("Signals", fontsize=20, y=1.05, weight="bold")
            plt.savefig(os.path.join(dir_path, 'reports', 'figures', 'ecg_signals.png'))  # Save the figure



# plot the confusion matrix
def plot_confusion_matrix(y_true, y_pred, figure_name):
    cm = confusion_matrix(y_true, y_pred)
    num_classes = len(np.unique(y_true))

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    ax.xaxis.set_ticklabels(range(num_classes))
    ax.yaxis.set_ticklabels(range(num_classes))
    plt.savefig(os.path.join(dir_path, 'reports', 'figures', f'confusion_matrix_{figure_name}.png'))  # Save the figure
    #plt.show()


# Set the directory path as an environment variable
os.environ['DIR_PATH'] = '/Users/behdad/sickkids_interview/ECG Heartbeat Categorization'
dir_path = os.getenv('DIR_PATH')
df_train_after = pd.read_csv(os.path.join(dir_path, 'data/processed/processed_train.csv'), header=None)
df_train_before = pd.read_csv(os.path.join(dir_path, 'data/interim/inter_train.csv'), header=None)


## to draw a pie chart after and before balancing
plot_pie_chart(df_train_before, 'before_balancing')
plot_pie_chart(df_train_after, 'after_balancing')


# To draw a ECG signals for each class
ecg_plotter = EcgSignalPlotter(df_train_after)
ecg_plotter.get_samples()
ecg_plotter.plot_ecg_signals()



