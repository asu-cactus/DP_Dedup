import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib import patches as mpatches

import matplotlib.pyplot as plt

fontsize = 10
plt.rcParams.update(
    {
        "font.size": fontsize,
        "axes.labelsize": fontsize,
        "xtick.labelsize": fontsize,
        "ytick.labelsize": fontsize,
        "legend.fontsize": 11,
        # "axes.titlesize": 12,
    }
)

fig, ax = plt.subplots(2, 3)
fig.set_figwidth(12)
fig.set_figheight(4)

##############################################

eps = np.arange(1, 4)
cr_drd = [0.4175, 0.4175, 0.4478]
cr_dred = [0.4175, 0.4175, 0.4174]
cr_drd = [cr * 100 for cr in cr_drd]
cr_dred = [cr * 100 for cr in cr_dred]
eps_inc_sum = [1.2, 1.2, 1.5]


# ax[0, 0].set_xlabel("Base models")
ax[0, 0].set_ylabel("Overall C.R.")  # we already handled the x-label with ax1
ax[0, 0].bar(eps - 0.2, cr_drd, 0.4, color="C0", label="DRD")
ax[0, 0].bar(eps + 0.2, cr_dred, 0.4, color="C1", label="DRED")
# ax2.bar(eps, cr_dred)
ax[0, 0].set_ylim([0, 52])
ax[0, 0].tick_params(axis="y")
# ax[1,1].legend(loc="upper right")

ax1_twin = ax[0, 0].twinx()  # instantiate a second Axes that shares the same x-axis

ax1_twin.set_ylabel("Sum of Eps. Increase")
ax1_twin.plot(eps, eps_inc_sum, "o-", label="Sum of Eps. Increase", color="C3")
ax1_twin.set_ylim([1.0, 1.6])
ax1_twin.tick_params(axis="y")

ax[0, 0].set_xticks(np.arange(0, 4))
ax[0, 0].set_xticklabels(["", "Ours", "Multi", "Cross"])

# ax4_twin.legend(loc="upper right")
ax[0, 0].title.set_text("A2. Roberta-SST2")

##############################################

eps = np.arange(1, 4)
cr_drd = [0.2440, 0.4175, 0.4182]
cr_dred = [0.2390, 0.4175, 0.4182]
cr_drd = [cr * 100 for cr in cr_drd]
cr_dred = [cr * 100 for cr in cr_dred]
eps_inc_sum = [0.8, 1.2, 1.4]


# ax[0, 1].set_xlabel("Base models")
ax[0, 1].set_ylabel("Overall C.R.")  # we already handled the x-label with ax1
ax[0, 1].bar(eps - 0.2, cr_drd, 0.4, color="C0", label="DRD")
ax[0, 1].bar(eps + 0.2, cr_dred, 0.4, color="C1", label="DRED")
# ax2.bar(eps, cr_dred)
ax[0, 1].set_ylim([0, 50])
ax[0, 1].tick_params(axis="y")
# ax[1,1].legend(loc="upper right")

ax2_twin = ax[0, 1].twinx()  # instantiate a second Axes that shares the same x-axis

ax2_twin.set_ylabel("Sum of Eps. Increase")
ax2_twin.plot(eps, eps_inc_sum, "o-", label="Sum of Eps. Increase", color="C3")
ax2_twin.set_ylim([0.5, 1.5])
ax2_twin.tick_params(axis="y")

ax[0, 1].set_xticks(np.arange(0, 4))
ax[0, 1].set_xticklabels(["", "Ours", "Multi", "Cross"])

ax[0, 1].title.set_text("A3. Roberta-MNLI")

##############################################

eps = np.arange(1, 4)
cr_drd = [0.3462, 0.3546, 0.4571]
cr_dred = [0.3437, 0.3593, 0.4571]
cr_drd = [cr * 100 for cr in cr_drd]
cr_dred = [cr * 100 for cr in cr_dred]
eps_inc_sum = [9, 29, 35]


# ax[0, 2].set_xlabel("Base models")
ax[0, 2].set_ylabel("Overall C.R.")  # we already handled the x-label with ax1
ax[0, 2].bar(eps - 0.2, cr_drd, 0.4, color="C0", label="DRD")
ax[0, 2].bar(eps + 0.2, cr_dred, 0.4, color="C1", label="DRED")
# ax2.bar(eps, cr_dred)
ax[0, 2].set_ylim([0, 50])
ax[0, 2].tick_params(axis="y")
# ax[1,1].legend(loc="upper right")

ax3_twin = ax[0, 2].twinx()  # instantiate a second Axes that shares the same x-axis

ax3_twin.set_ylabel("Sum of Eps. Increase")
ax3_twin.plot(eps, eps_inc_sum, "o-", label="Sum of Eps. Increase", color="C3")
ax3_twin.set_ylim([0, 36])
ax3_twin.tick_params(axis="y")

ax[0, 2].set_xticks(np.arange(0, 4))
ax[0, 2].set_xticklabels(["", "Ours", "Multi", "Cross"])

# ax4_twin.legend(loc="upper right")
ax[0, 2].title.set_text("B1. Roberta-QNLI")

##############################################

eps = np.arange(1, 4)
cr_drd = [0.2612, 0.3236, 0.4571]
cr_dred = [0.2585, 0.3236, 0.4661]
cr_drd = [cr * 100 for cr in cr_drd]
cr_dred = [cr * 100 for cr in cr_dred]
eps_inc_sum = [4.5, 4.6, 5.3]


# ax[1, 0].set_xlabel("Base models")
ax[1, 0].set_ylabel("Overall C.R.")  # we already handled the x-label with ax1
ax[1, 0].bar(eps - 0.2, cr_drd, 0.4, color="C0", label="DRD")
ax[1, 0].bar(eps + 0.2, cr_dred, 0.4, color="C1", label="DRED")
# ax2.bar(eps, cr_dred)
ax[1, 0].set_ylim([0, 50])
ax[1, 0].tick_params(axis="y")
# ax[1,1].legend(loc="upper right")

ax4_twin = ax[1, 0].twinx()  # instantiate a second Axes that shares the same x-axis

ax4_twin.set_ylabel("Sum of Eps. Increase")
ax4_twin.plot(eps, eps_inc_sum, "o-", label="Sum of Eps. Increase", color="C3")
ax4_twin.set_ylim([4, 5.5])
ax4_twin.tick_params(axis="y")

ax[1, 0].set_xticks(np.arange(0, 4))
ax[1, 0].set_xticklabels(["", "Ours", "Multi", "Cross"])

# ax4_twin.legend(loc="upper right")
ax[1, 0].title.set_text("B2. ViT-CIFAR100")


##############################################

eps = np.arange(1, 4)
cr_drd = [0.3764, 0.3817, 0.3942]
cr_dred = [0.3767, 0.3817, 0.3942]
cr_drd = [cr * 100 for cr in cr_drd]
cr_dred = [cr * 100 for cr in cr_dred]
eps_inc_sum = [3.8, 13.4, 16.4]


# ax[1, 1].set_xlabel("Base models")
ax[1, 1].set_ylabel("Overall C.R.")  # we already handled the x-label with ax1
ax[1, 1].bar(eps - 0.2, cr_drd, 0.4, color="C0", label="DRD")
ax[1, 1].bar(eps + 0.2, cr_dred, 0.4, color="C1", label="DRED")
# ax2.bar(eps, cr_dred)
ax[1, 1].set_ylim([0, 50])
ax[1, 1].tick_params(axis="y")
# ax[1,1].legend(loc="upper right")

ax5_twin = ax[1, 1].twinx()  # instantiate a second Axes that shares the same x-axis

ax5_twin.set_ylabel("Sum of Eps. Increase")
ax5_twin.plot(eps, eps_inc_sum, "o-", label="Sum of Eps. Increase", color="C3")
ax5_twin.set_ylim([0, 18])
ax5_twin.tick_params(axis="y")

ax[1, 1].set_xticks(np.arange(0, 4))
ax[1, 1].set_xticklabels(["", "Ours", "Multi", "Cross"])

# ax4_twin.legend(loc="upper right")
ax[1, 1].title.set_text("B3. ResNet-CelebA")

##############################################
# line1 = Line2D([0], [0], label='Sum of Eps. Increase for DRD', marker='o', color='C0')
# line2 = Line2D([0], [0], label='Sum of Eps. Increase for DRED', marker='*', color='C1')
# patch1 = mpatches.Patch(color='C0', label='C.R. for DRD')
# patch2 = mpatches.Patch(color='C1', label='C.R. for DRED')
# handles = [line1, line2, patch1, patch2]

# # Put a legend below current axis
# plt.legend(handles=handles, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.65))

# fig.tight_layout()  # otherwise the right y-label is slightly clipped

# # Shrink current axis's height by 10% on the bottom
# plt.subplots_adjust(top=0.95, bottom=0.20, wspace=0.3, hspace=0.5)

# Clear bottom-right ax
bottom_right_ax = ax[-1, -1]
bottom_right_ax.clear()  # clears the random data I plotted previously
bottom_right_ax.set_axis_off()  # removes the XY axes


patch1 = mpatches.Patch(color="C0", label="C.R. (in %) for DRD")
patch2 = mpatches.Patch(color="C1", label="C.R. (in %) for DRED")
line1 = Line2D(
    [0], [0], label="Sum of epsilon increase\nin a cluster", marker="o", color="C3"
)

handles = [patch1, patch2, line1]
bottom_right_ax.legend(handles=handles, loc="center")

##############################################

fig.tight_layout()  # otherwise the right y-label is slightly clipped

# plt.show()

plt.savefig("base_model_selection.pdf")
