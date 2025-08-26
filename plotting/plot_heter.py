import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import patches as mpatches
import numpy as np
import pdb

fontsize = 13
plt.rcParams.update(
    {
        "font.size": fontsize,
        "axes.labelsize": fontsize,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.titlesize": fontsize,
    }
)

# fig, ax1 = plt.subplots()
fig, ax = plt.subplots(2, 1)
fig.set_figwidth(4)
fig.set_figheight(5)

#### Color and Style


# OB_B = 'C5' # Optimized baseline bar char color
# OB_S = 'C5X' # Optimized baseline line style
# OB_AL = 0.6 # Optimized baseline transparency
AL = 0.6
##############################################

# A2. CIFAR100 - SST2

C0 = "#50ad9f"  # Dedup bar char color
# OB_S = 'C5X-' # Dedup line style
# OB_AL = 1.0 # Dedup transparency

C1 = "#e9c716"
# OB_N = "C6p-"

C2 = "#bc272d"

xs = np.arange(2, 5)

cr_greedy1_bb = [0.52, 0.6031, 0.5597]
cr_binary_bb = [0.7042, 0.7511, 0.7259]
cr_halving_bb = [0.7042, 0.7728, 0.7511]
cr_greedy1_sb = [0.4739, 0.4781, 0.4655]
cr_binary_sb = [0.675, 0.6804, 0.6708]
cr_halving_sb = [0.6876, 0.6789, 0.6792]


cr_greedy1_bb = [cr * 100 for cr in cr_greedy1_bb]
cr_binary_bb = [cr * 100 for cr in cr_binary_bb]
cr_halving_bb = [cr * 100 for cr in cr_halving_bb]
cr_greedy1_sb = [cr * 100 for cr in cr_greedy1_sb]
cr_binary_sb = [cr * 100 for cr in cr_binary_sb]
cr_halving_sb = [cr * 100 for cr in cr_halving_sb]

ax[0].set_xlabel("Epsilon")
ax[0].set_ylabel("C.R.")  # we already handled the x-label with ax1


ax[0].bar(xs - 0.3, cr_greedy1_sb, 0.1, color=C0, label="Greedy-1-sb")
ax[0].bar(xs - 0.2, cr_binary_sb, 0.1, color=C1, label="DRD-sb")
ax[0].bar(xs - 0.1, cr_halving_sb, 0.1, color=C2, label="DRED-sb")
ax[0].bar(xs, cr_greedy1_bb, 0.1, color=C0, label="Greedy-1-bb", alpha=AL)
ax[0].bar(xs + 0.1, cr_binary_bb, 0.1, color=C1, label="DRD-bb", alpha=AL)
ax[0].bar(xs + 0.2, cr_halving_bb, 0.1, color=C2, label="DRED-bb", alpha=AL)


# ax1.bar(xs, cr_halving)
ax[0].set_ylim([40, 80])
ax[0].tick_params(axis="y")
# ax[0].legend(loc="lower left")

ax0_twin = ax[0].twinx()  # instantiate a second Axes that shares the same x-axis

# bs_dedup_accs_bb = [0.8876, 0.8956, 0.8979]
# sh_dedup_accs_bb = [0.8956, 0.9036, 0.9048]
# g1_dedup_accs_bb = [0.8899, 0.8922, 0.8979]
g1_dedup_accs_sb = [0.8876, 0.8887, 0.8922]
bs_dedup_accs_sb = [0.8899, 0.8922, 0.8933]
sh_dedup_accs_sb = [0.8887, 0.8899, 0.8922]

# ==============================================================================

ax0_twin.set_ylabel("Accuracy")
ax0_twin.plot(
    xs,
    g1_dedup_accs_sb,
    marker="^",
    color=C0,
    linestyle="--",
    label="Greedy-1-sb",
)
ax0_twin.plot(xs, bs_dedup_accs_sb, marker="o", color=C1, linestyle="-", label="DRD-sb")
ax0_twin.plot(
    xs, sh_dedup_accs_sb, marker="*", color=C2, linestyle="-", label="DRED-sb"
)

ax0_twin.set_ylim([0.75, 0.91])
ax0_twin.tick_params(axis="y")

ax[0].set_xticklabels([0.5, 0.6, 0.8, 1.0])
# ax0_twin.legend(loc="lower left")
ax[0].title.set_text("CIFAR100 - SST2")


##############################################

# Clear bottom-right ax
bottom_right_ax = ax[1]
bottom_right_ax.clear()  # clears the random data I plotted previously
bottom_right_ax.set_axis_off()  # removes the XY axes

line1 = Line2D(
    [0],
    [0],
    label="Accuracy for Greedy-1 with block size 49152",
    marker="^",
    color=C0,
    linestyle="--",
)
line2 = Line2D(
    [0],
    [0],
    label="Accuracy for DRD with block size 49152",
    marker="o",
    color=C1,
    linestyle="-",
)
line3 = Line2D(
    [0],
    [0],
    label="Accuarcy for DRED with block size 49152",
    marker="*",
    color=C2,
    linestyle="-",
)


patch1 = mpatches.Patch(
    color=C0, label="C.R. (in %) for Greedy-1 with block size 49152"
)
patch2 = mpatches.Patch(color=C1, label="C.R. (in %) for DRD with block size 49152")
patch3 = mpatches.Patch(color=C2, label="C.R. (in %) for DRED with block size 49152")
patch4 = mpatches.Patch(
    color=C0, alpha=AL, label="C.R. (in %) for Greedy-1 with block size 589824"
)
patch5 = mpatches.Patch(
    color=C1, alpha=AL, label="C.R. (in %) for DRD with block size 589824"
)
patch6 = mpatches.Patch(
    color=C2, alpha=AL, label="C.R. (in %) for DRED with block size 589824"
)

handles = [line1, line2, line3, patch1, patch2, patch3, patch4, patch5, patch6]
bottom_right_ax.legend(handles=handles, loc="center")


##############################################

fig.tight_layout()  # otherwise the right y-label is slightly clipped
# plt.show()
plt.savefig("hetero_models.pdf")
