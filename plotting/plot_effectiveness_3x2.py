import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import patches as mpatches
import numpy as np
import pdb

fontsize = 14
plt.rcParams.update(
    {
        "font.size": fontsize,
        "axes.labelsize": fontsize,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "axes.titlesize": fontsize,
    }
)

# fig, ax1 = plt.subplots()
fig, ax = plt.subplots(2, 3)
fig.set_figwidth(15)
fig.set_figheight(6)
##############################################

xs = np.arange(1, 6)
cr_binary = [0.0, 0.0541, 0.1516, 0.0541, 0.2238]
cr_halving = [0.0, 0.0288, 0.1769, 0.1516, 0.2238]
cr_binary = [cr * 100 for cr in cr_binary]
cr_halving = [cr * 100 for cr in cr_halving]
cr_mistique20 = [0.0, 18.37, 100, 100, 100]
cr_dedup20 = [0.0, 18.37, 18.37, 18.37, 100]

ax[0, 0].set_xlabel("Epsilon")
ax[0, 0].set_ylabel("C.R.")  # we already handled the x-label with ax1
ax[0, 0].bar(xs - 0.2, cr_mistique20, 0.1, color="C0", label="Mistique20")
ax[0, 0].bar(xs - 0.1, cr_dedup20, 0.1, color="C1", label="Dedup20")
ax[0, 0].bar(xs, cr_binary, 0.1, color="C2", label="DRD")
ax[0, 0].bar(xs + 0.1, cr_halving, 0.1, color="C3", label="DRED")
# ax2.bar(xs, cr_halving)
ax[0, 0].set_ylim([0, 100])
ax[0, 0].tick_params(axis="y")
# ax[0,0].legend(loc="lower left")

ax1_twin = ax[0, 0].twinx()  # instantiate a second Axes that shares the same x-axis
original_accs = [0.7836, 0.7927, 0.7997, 0.8049, 0.8148]
bs_dedup_accs = [0.7836, 0.7882, 0.7913, 0.7975, 0.8034]
sh_dedup_accs = [0.7836, 0.7935, 0.7966, 0.7983, 0.8034]
retrain_accs = [0.7836, 0.7997, 0.8012, 0.8148, 0.818]
mistique20_accs = [0.7836, 0.7975, 0.7992, 0.8047, 0.8146]
dedup20_accs = [0.7836, 0.7975, 0.7996, 0.8074, 0.8146]

ax1_twin.set_ylabel("Accuracy")
ax1_twin.plot(xs, mistique20_accs, "C0s-", label="Mistique20")
ax1_twin.plot(xs, dedup20_accs, "C1v-", label="Dedup20")
ax1_twin.plot(xs, bs_dedup_accs, "C2o-", label="DRD")
ax1_twin.plot(xs, sh_dedup_accs, "C3*-", label="DRED")
ax1_twin.plot(xs, original_accs, "C4^--", label="Original")
ax1_twin.plot(xs, retrain_accs, "C5--", label="Retrain")
ax1_twin.set_ylim([0.72, 0.83])
ax1_twin.tick_params(axis="y")

ax[0, 0].set_xticklabels([0.0, 1.0, 2.0, 4.0, 6.0, 7.0])
# ax1_twin.legend(loc="lower left")
ax[0, 0].title.set_text("A1. Roberta QNLI")

##############################################

xs = np.arange(1, 6)
cr_binary = [0.101, 0.1516, 0.343, 0.1516, 0.2491]
cr_halving = [0.2599, 0.1335, 0.343, 0.1335, 0.2599]
cr_binary = [cr * 100 for cr in cr_binary]
cr_halving = [cr * 100 for cr in cr_halving]
cr_mistique20 = [100, 100, 100, 100, 100]
cr_dedup20 = [85.55, 100, 92.78, 100, 100]

ax[0, 1].set_xlabel("Epsilon")
ax[0, 1].set_ylabel("C.R.")  # we already handled the x-label with ax1
ax[0, 1].bar(xs - 0.2, cr_mistique20, 0.1, color="C0", label="Mistique20")
ax[0, 1].bar(xs - 0.1, cr_dedup20, 0.1, color="C1", label="Dedup20")
ax[0, 1].bar(xs, cr_binary, 0.1, color="C2", label="DRD")
ax[0, 1].bar(xs + 0.1, cr_halving, 0.1, color="C3", label="DRED")
# ax3.bar(xs, cr_halving)
ax[0, 1].set_ylim([0, 100])
ax[0, 1].tick_params(axis="y")
# ax[0,1].legend(loc="lower left")

ax3_twin = ax[0, 1].twinx()  # instantiate a second Axes that shares the same x-axis
original_accs = [0.8750, 0.9002, 0.9025, 0.9037, 0.9071]
bs_dedup_accs = [0.8693, 0.8922, 0.8956, 0.9037, 0.906]
sh_dedup_accs = [0.8635, 0.8888, 0.8956, 0.8968, 0.906]
retrain_accs = [0.8750, 0.9002, 0.9025, 0.9037, 0.9071]
mistique20_accs = [0.8750, 0.9002, 0.9025, 0.9037, 0.9071]
dedup20_accs = [0.8647, 0.9002, 0.9083, 0.9037, 0.9071]

ax3_twin.set_ylabel("Accuracy")
ax3_twin.plot(xs, mistique20_accs, "C0s-", label="Mistique20")
ax3_twin.plot(xs, dedup20_accs, "C1v-", label="Dedup20")
ax3_twin.plot(xs, bs_dedup_accs, "C2o-", label="DRD")
ax3_twin.plot(xs, sh_dedup_accs, "C3*-", label="DRED")
ax3_twin.plot(xs, original_accs, "C4^--", label="Original")
ax3_twin.plot(xs, retrain_accs, "C5--", label="Retrain")
ax3_twin.set_ylim([0.80, 0.91])
ax3_twin.tick_params(axis="y")

ax[0, 1].set_xticklabels([0.0, 0.3, 0.4, 0.6, 0.8, 1.0])
# ax3_twin.legend(loc="lower left")
ax[0, 1].title.set_text("A2. Roberta MNLI-SST2")

##############################################


xs = np.arange(1, 6)
cr_binary = [0.0, 0.0541, 0.0541, 0.0541, 0.0288]
cr_halving = [0.0, 0.0541, 0.0541, 0.0541, 0.0541]
cr_binary = [cr * 100 for cr in cr_binary]
cr_halving = [cr * 100 for cr in cr_halving]
cr_mistique20 = [0.0, 42.57, 42.57, 100, 100]
cr_dedup20 = [0.0, 18.37, 18.37, 51.96, 100]

ax[0, 2].set_xlabel("Epsilon")
ax[0, 2].set_ylabel("C.R.")  # we already handled the x-label with ax1
ax[0, 2].bar(xs - 0.2, cr_mistique20, 0.1, color="C0", label="Mistique20")
ax[0, 2].bar(xs - 0.1, cr_dedup20, 0.1, color="C1", label="Dedup20")
ax[0, 2].bar(xs, cr_binary, 0.1, color="C2", label="DRD")
ax[0, 2].bar(xs + 0.1, cr_halving, 0.1, color="C3", label="DRED")
ax[0, 2].set_ylim([0, 100])
ax[0, 2].tick_params(axis="y")
# ax[1,0].legend(loc="lower left")

ax4_twin = ax[0, 2].twinx()  # instantiate a second Axes that shares the same x-axis
original_accs = [0.6885, 0.7294, 0.7435, 0.7490, 0.7175]
bs_dedup_accs = [0.6885, 0.7175, 0.7328, 0.7352, 0.7203]
sh_dedup_accs = [0.6885, 0.7184, 0.7328, 0.7352, 0.7349]
retrain_accs = [0.6885, 0.7294, 0.7435, 0.7490, 0.7175]
mistique20_accs = [0.6885, 0.7132, 0.7267, 0.7487, 0.7187]
dedup20_accs = [0.6885, 0.7151, 0.7288, 0.7282, 0.7187]

ax4_twin.set_ylabel("Accuracy")
ax4_twin.plot(xs, mistique20_accs, "C0s-", label="Mistique20")
ax4_twin.plot(xs, dedup20_accs, "C1v-", label="Dedup20")
ax4_twin.plot(xs, bs_dedup_accs, "C2o-", label="DRD")
ax4_twin.plot(xs, sh_dedup_accs, "C3*-", label="DRED")
ax4_twin.plot(xs, original_accs, "C4^--", label="Original")
ax4_twin.plot(xs, retrain_accs, "C5--", label="Retrain")
ax4_twin.set_ylim([0.60, 0.76])
ax4_twin.tick_params(axis="y")

ax[0, 2].set_xticklabels([0.0, 0.2, 0.4, 0.8, 1.6, 2.0])
# ax4_twin.legend(loc="lower left")
ax[0, 2].title.set_text("A3. Roberta MNLI")

##############################################

xs = np.arange(1, 6)
cr_binary = [0.0, 0.0669, 0.2224, 0.1913, 0.2224]
cr_halving = [0.0, 0.0358, 0.2224, 0.1913, 0.2224]
cr_binary = [cr * 100 for cr in cr_binary]
cr_halving = [cr * 100 for cr in cr_halving]
cr_mistique20 = [0.0, 8.76, 8.76, 100, 100]
cr_dedup20 = [0.0, 8.76, 8.76, 21.55, 21.55]

ax[1, 0].set_xlabel("Epsilon")
ax[1, 0].set_ylabel("C.R.")  # we already handled the x-label with ax1
ax[1, 0].bar(xs - 0.2, cr_mistique20, 0.1, color="C0", label="Mistique20")
ax[1, 0].bar(xs - 0.1, cr_dedup20, 0.1, color="C1", label="Dedup20")
ax[1, 0].bar(xs, cr_binary, 0.1, color="C2", label="DRD")
ax[1, 0].bar(xs + 0.1, cr_halving, 0.1, color="C3", label="DRED")
# ax1.bar(xs, cr_halving)
ax[1, 0].set_ylim([0, 100])
ax[1, 0].tick_params(axis="y")
# ax[1,1].legend(loc="lower left")

ax0_twin = ax[1, 0].twinx()  # instantiate a second Axes that shares the same x-axis
original_accs = [0.8101, 0.8331, 0.8558, 0.8757, 0.8964]
bs_dedup_accs = [0.8101, 0.8208, 0.8375, 0.857, 0.878]
sh_dedup_accs = [0.8101, 0.8198, 0.8375, 0.857, 0.878]
retrain_accs = [0.8101, 0.8783, 0.883, 0.89, 0.8996]
mistique20_accs = [0.8101, 0.824, 0.84, 0.8569, 0.8793]
dedup20_accs = [0.8101, 0.8232, 0.8383, 0.8758, 0.8963]

ax0_twin.set_ylabel("Accuracy")
ax0_twin.plot(xs, mistique20_accs, "C0s-", label="Mistique20")
ax0_twin.plot(xs, dedup20_accs, "C1v-", label="Dedup20")
ax0_twin.plot(xs, bs_dedup_accs, "C2o-", label="DRD")
ax0_twin.plot(xs, sh_dedup_accs, "C3*-", label="DRED")
ax0_twin.plot(xs, original_accs, "C4^--", label="Original")
ax0_twin.plot(xs, retrain_accs, "C5--", label="Retrain")
ax0_twin.set_ylim([0.7, 0.9])
ax0_twin.tick_params(axis="y")

ax[1, 0].set_xticklabels([0.0, 0.5, 0.6, 0.75, 1.0, 2.0])
# ax0_twin.legend(loc="lower left")
ax[1, 0].title.set_text("A4. ViT CIFAR100")

##############################################


xs = np.arange(1, 6)
cr_binary = [0.0, 0.1176, 0.1176, 0.1851, 0.3562]
cr_halving = [0.0, 0.0861, 0.0861, 0.1311, 0.4057]
cr_binary = [cr * 100 for cr in cr_binary]
cr_halving = [cr * 100 for cr in cr_halving]
cr_mistique20 = [0.0, 84.24, 100, 100, 100]
cr_dedup20 = [0.0, 84.24, 84.24, 84.24, 84.24]

ax[1, 1].set_xlabel("Epsilon")
ax[1, 1].set_ylabel("C.R.")  # we already handled the x-label with ax1
ax[1, 1].bar(xs - 0.2, cr_mistique20, 0.1, color="C0", label="Mistique20")
ax[1, 1].bar(xs - 0.1, cr_dedup20, 0.1, color="C1", label="Dedup20")
ax[1, 1].bar(xs, cr_binary, 0.1, color="C2", label="DRD")
ax[1, 1].bar(xs + 0.1, cr_halving, 0.1, color="C3", label="DRED")
# ax3.bar(xs, cr_halving)
ax[1, 1].set_ylim([0, 100])
ax[1, 1].tick_params(axis="y")
# ax[2,0].legend(loc="lower left")

ax2_twin = ax[1, 1].twinx()  # instantiate a second Axes that shares the same x-axis
original_accs = [0.8037, 0.8192, 0.8258, 0.8307, 0.8368]
bs_dedup_accs = [0.8037, 0.8116, 0.8143, 0.8148, 0.8186]
sh_dedup_accs = [0.8037, 0.8111, 0.8123, 0.8125, 0.8237]
retrain_accs = [0.8037, 0.8307, 0.8337, 0.8347, 0.8375]
mistique20_accs = [0.8037, 0.8159, 0.8260, 0.8311, 0.8369]
dedup20_accs = [0.8037, 0.8159, 0.8235, 0.8239, 0.8272]

ax2_twin.set_ylabel("Accuracy")
ax2_twin.plot(xs, mistique20_accs, "C0s-", label="Mistique20")
ax2_twin.plot(xs, dedup20_accs, "C1v-", label="Dedup20")
ax2_twin.plot(xs, bs_dedup_accs, "C2o-", label="DRD")
ax2_twin.plot(xs, sh_dedup_accs, "C3*-", label="DRED")
ax2_twin.plot(xs, original_accs, "C4^--", label="Original")
ax2_twin.plot(xs, retrain_accs, "C5--", label="Retrain")
ax2_twin.set_ylim([0.75, 0.85])
ax2_twin.tick_params(axis="y")

ax[1, 1].set_xticklabels([0.0, 0.4, 0.6, 0.8, 1.0, 2.0])
# ax2_twin.legend(loc="lower left")
ax[1, 1].title.set_text("A5. ResNet CelebA")

##############################################


# Clear bottom-right ax
bottom_right_ax = ax[-1, -1]
bottom_right_ax.clear()  # clears the random data I plotted previously
bottom_right_ax.set_axis_off()  # removes the XY axes


line1 = Line2D(
    [0],
    [0],
    label="Accuracy of the original model (Original)",
    marker="^",
    color="C4",
    linestyle="--",
)
line2 = Line2D(
    [0],
    [0],
    label="Accuracy of a model finetuned using\nthe dedup. model's epsilon (Retrain)",
    color="C5",
    linestyle="--",
)
line3 = line3 = Line2D(
    [0], [0], label="Accuracy for Mistique-20", marker="s", color="C0", linestyle="-"
)
line4 = Line2D(
    [0], [0], label="Accuracy for Dedup-20", marker="v", color="C1", linestyle="-"
)
line5 = Line2D(
    [0], [0], label="Accuracy for DRD", marker="o", color="C2", linestyle="-"
)
line6 = Line2D(
    [0], [0], label="Accuracy for DRED", marker="*", color="C3", linestyle="-"
)
patch1 = mpatches.Patch(color="C0", label="C.R. (in %) for Mistique-20")
patch2 = mpatches.Patch(color="C1", label="C.R. (in %) for Dedup-20")
patch3 = mpatches.Patch(color="C2", label="C.R. (in %) for DRD")
patch4 = mpatches.Patch(color="C3", label="C.R. (in %) for DRED")
handles = [line1, line2, line3, line4, line5, line6, patch1, patch2, patch3, patch4]
bottom_right_ax.legend(handles=handles, loc="center")


##############################################

fig.tight_layout()  # otherwise the right y-label is slightly clipped
# plt.show()
plt.savefig("effectiveness_3x2.pdf")
