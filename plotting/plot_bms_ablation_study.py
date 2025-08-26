import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib import patches as mpatches

import matplotlib.pyplot as plt

fontsize = 13
plt.rcParams.update(
    {
        "font.size": fontsize,
        "axes.labelsize": fontsize,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": fontsize,
        "axes.titlesize": fontsize,
    }
)

fig, ax = plt.subplots(2, 3)
fig.set_figwidth(15)
fig.set_figheight(5)
a1 = ax[0, 0]
a2 = ax[0, 1]
a3 = ax[0, 2]
a4 = ax[1, 0]
a5 = ax[1, 1]


##############################################

eps = np.arange(1, 6)
cr_drd = [0.4008, 0.3773, 0.3764, 0.3773, 0.3764]
cr_dred = [0.4008, 0.3764, 0.3764, 0.3701, 0.3701]
cr_drd = [cr * 100 for cr in cr_drd]
cr_dred = [cr * 100 for cr in cr_dred]


a1.set_xlabel("Epsilon of the Base Model")
a1.set_ylabel("C.R.")  # we already handled the x-label with ax1
a1.bar(eps - 0.2, cr_drd, 0.4, color="C0", label="DRD")
a1.bar(eps + 0.2, cr_dred, 0.4, color="C1", label="DRED")

a1.set_ylim([0, 45])
a1.tick_params(axis="y")

a1.set_xticklabels([0.0, 1.0, 2.0, 3.0, 4.0, "5.0(Ours)"])
a1.title.set_text("Different single base model (C1,eps*=5.0)")

##############################################

eps = np.arange(1, 6)
cr_drd = [0.3627, 0.4422, 0.4028, 0.3627, 0.3406]
cr_dred = [0.5224, 0.4429, 0.418, 0.3599, 0.3344]
cr_drd = [cr * 100 for cr in cr_drd]
cr_dred = [cr * 100 for cr in cr_dred]

a2.set_xlabel("Epsilon of the Base Model")
a2.set_ylabel("C.R.")  # we already handled the x-label with ax1
a2.bar(eps - 0.2, cr_drd, 0.4, color="C0", label="DRD")
a2.bar(eps + 0.2, cr_dred, 0.4, color="C1", label="DRED")

a2.set_ylim([0, 55])
a2.tick_params(axis="y")


a2.set_xticklabels([0.0, 0.30, 0.35, 0.40, 0.45, "0.5(Ours)"])
# ax1_twin.legend(loc="upper right")
a2.title.set_text("Different single base model (C2)")

##############################################


eps = np.arange(1, 6)
cr_drd = [0.4525, 0.439, 0.439, 0.448, 0.3945]
cr_dred = [0.4597, 0.4156, 0.382, 0.4282, 0.3641]
cr_drd = [cr * 100 for cr in cr_drd]
cr_dred = [cr * 100 for cr in cr_dred]

a3.set_xlabel("Epsilon of the Base Model")
a3.set_ylabel("C.R.")  # we already handled the x-label with ax1
a3.bar(eps - 0.2, cr_drd, 0.4, color="C0", label="DRD")
a3.bar(eps + 0.2, cr_dred, 0.4, color="C1", label="DRED")
# ax2.bar(eps, cr_dred)
a3.set_ylim([0, 55])
a3.tick_params(axis="y")

a3.set_xticklabels([0.0, 0.3, 0.4, 0.5, 0.6, "0.7(Ours)"])
# ax3_twin.legend(loc="upper right")
a3.title.set_text("Different single base model (C3)")

##############################################

eps = np.arange(1, 4)
cr_drd = [0.3367, 0.3132, 0.0541]
cr_dred = [0.3304, 0.3249, 0.0477]
cr_drd = [cr * 100 for cr in cr_drd]
cr_dred = [cr * 100 for cr in cr_dred]
nov_binary = [21, 20, 20]
nov_halving = [18, 22, 16]

a4.set_xlabel("Base model - Epsilon")
a4.set_ylabel("C.R.")  # we already handled the x-label with ax1
a4.bar(eps - 0.2, cr_drd, 0.4, color="C0", label="DRD")
a4.bar(eps + 0.2, cr_dred, 0.4, color="C1", label="DRED")
# ax2.bar(eps, cr_dred)
a4.set_ylim([0, 50])
a4.tick_params(axis="y")
# ax[1,1].legend(loc="upper right")

ax4_twin = a4.twinx()  # instantiate a second Axes that shares the same x-axis

ax4_twin.set_ylabel("Num. of Validations")
ax4_twin.plot(eps, nov_binary, "o-", label="DRD")
ax4_twin.plot(eps, nov_halving, "*-", label="DRED")
ax4_twin.set_ylim([6, 23])
ax4_twin.tick_params(axis="y")

a4.set_xticks(np.arange(0, 4))
a4.set_xticklabels(["", "QNLI-0.2", "SST2-0.3", "MNLI-0.2(Ours)"])

# ax4_twin.legend(loc="upper right")
a4.title.set_text("Intra-data vs Inter-data (C4)")

##############################################

eps = np.arange(1, 5)
cr_drd = [0.1389, 0.1335, 0.1696, 0.1751]
cr_dred = [0.1507, 0.1507, 0.1696, 0.1741]
cr_drd = [cr * 100 for cr in cr_drd]
cr_dred = [cr * 100 for cr in cr_dred]
nov_binary = [26, 28, 26, 26]
nov_halving = [24, 24, 28, 29]

a5.set_xlabel("Base Model(s)")
a5.set_ylabel("C.R.")  # we already handled the x-label with ax1
a5.bar(eps - 0.2, cr_drd, 0.4, color="C0", label="DRD")
a5.bar(eps + 0.2, cr_dred, 0.4, color="C1", label="DRED")
# ax2.bar(eps, cr_dred)
a5.set_ylim([0, 25])
a5.tick_params(axis="y")
# ax[1,0].legend(loc="upper right")

ax5_twin = a5.twinx()  # instantiate a second Axes that shares the same x-axis

ax5_twin.set_ylabel("Num. of Validations")
ax5_twin.plot(eps, nov_binary, "o-", label="DRD")
ax5_twin.plot(eps, nov_halving, "*-", label="DRED")
ax5_twin.set_ylim([6, 30])
ax5_twin.tick_params(axis="y")

a5.set_xticks(np.arange(0, 5))
a5.set_xticklabels(["", "eps:4(Ours)", "eps:3,4", "eps:2,3,4", "eps:1,2,3,4"])

a5.title.set_text("Multiple base models (C1,eps*=10.0)")

##############################################
# line1 = Line2D([0], [0], label='Num. of Validations for DRD', marker='o', color='C0')
# line2 = Line2D([0], [0], label='Num. of Validations for DRED', marker='*', color='C1')
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


line1 = Line2D([0], [0], label="Number of validations for DRD", marker="o", color="C0")
line2 = Line2D([0], [0], label="Number of validations for DRED", marker="*", color="C1")
patch1 = mpatches.Patch(color="C0", label="C.R. (in %) for DRD")
patch2 = mpatches.Patch(color="C1", label="C.R. (in %) for DRED")
handles = [line1, line2, patch1, patch2]
bottom_right_ax.legend(handles=handles, loc="center")

##############################################

fig.tight_layout()  # otherwise the right y-label is slightly clipped

# plt.show()

plt.savefig("base_model_selection_ablation.pdf")
