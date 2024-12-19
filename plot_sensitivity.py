import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# fontsize = 13
# plt.rcParams.update(
#     {
#         "font.size": fontsize,
#         "axes.labelsize": fontsize,
#         "xtick.labelsize": 10,
#         "ytick.labelsize": 10,
#         "legend.fontsize": 12,
#         "axes.titlesize": fontsize,
#     }
# )


def plot_saliency():
    # Load data
    celeba_sens = np.load("celeba_sensitivity.npy")
    qnli_sens = np.load("qnli_sensitivity.npy")

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_figwidth(10)
    fig.set_figheight(5)

    ax1.set_title("CelebA Saliency")
    ax1.set_ylabel("Gradient")
    ax1.bar(np.arange(celeba_sens.shape[0]), celeba_sens)

    ax2.set_title("QNLI Saliency")
    ax2.set_ylabel("Gradient")
    ax2.bar(np.arange(qnli_sens.shape[0]), qnli_sens)

    # Save the plot
    plt.tight_layout()
    plt.savefig("sensitivity.png")

    # Display the plot
    plt.show()


def plot_disparity():

    # Load data
    qnli_disparity = np.load("qnli_scores.npy")
    # qnli_sst_disparity = np.load("qnli_sst2_scores.npy")

    sns.heatmap(qnli_disparity, cmap="YlGnBu")
    plt.title("QNLI Disparity")

    plt.tight_layout()
    plt.savefig("disparity.png")


plot_disparity()
