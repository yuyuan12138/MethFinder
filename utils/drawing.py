import matplotlib.pyplot as plt
from config import config
import os
import pickle
import umap
from .save_pickles import delete_all_previous_folder_files

def draw_acc_loss_line(train_acc_table, test_acc_table, train_loss_table, test_loss_table):
    # Draw and save accuracy and loss line plots
    plt.figure()
    plt.plot(train_acc_table, 'ro-', label='Train accuracy')
    plt.plot(test_acc_table, 'bs-', label='Val accuracy')
    plt.legend()
    plt.savefig("/".join(["acc_loss_plot", config.data_name + '_accuracy.png']))

    plt.figure()
    plt.plot(train_loss_table, 'ro-', label='Train loss')
    plt.plot(test_loss_table, 'bs-', label='Val loss')
    plt.legend()
    plt.savefig("/".join(["acc_loss_plot", config.data_name + '_loss.png']))

def draw_umap(folder_path):
    files = [file for file in os.listdir(f"{folder_path}/pickles") if os.path.isfile(os.path.join(f"{folder_path}/pickles", file))]
    print("<===== Start drawing Umap =====>")
    for path in files:
        print(f"<===== Start Drawing {path} =====>")
        with open(f'{folder_path}/pickles/{path}', 'rb') as f:
            data = pickle.load(f)

            length = len(data) // 2
            reducer = umap.UMAP()
            
            embedding = reducer.fit_transform(data)

            plt.scatter(
                embedding[length:, 0],
                embedding[length:, 1],
                color='lightblue',
                alpha=0.5,
                s=10,
                label='positive samples'
            )
            plt.scatter(
                embedding[:length, 0],
                embedding[:length, 1],
                color='lightcoral',
                alpha=0.5,
                s=10,
                label='negative samples'
            )

            plt.legend()
            plt.savefig(f"{folder_path}/figures/{path}.eps", dpi=600, format='eps')
            plt.clf()
            print(f"<===== Finish Drawing {path} =====>")

    print("<===== Finish drawing Umap =====>")

    delete_all_previous_folder_files(f"{folder_path}/pickles")