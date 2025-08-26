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
        "legend.fontsize": 12,
        "axes.titlesize": fontsize,
    }
)


fig, ax = plt.subplots(1, 2)
fig.set_figwidth(15)
fig.set_figheight(3.5)

##############################################

xs = np.arange(1, 11)
cr_drd = [0.0, 0.0358, 0.0358, 0.0358, 0.0669, 0.2224, 0.1602, 0.1291, 0.5023, 0.2846]
cr_drd_svt = [
    0.0,
    0.0669,
    0.1291,
    0.1291,
    0.098,
    0.2535,
    0.2224,
    0.1291,
    0.5023,
    0.3468,
]
cr_dred = [0.0, 0.0358, 0.0358, 0.0358, 0.0669, 0.2224, 0.1464, 0.1291, 0.5023, 0.2846]
cr_dred_svt = [
    0.0,
    0.0669,
    0.1291,
    0.0669,
    0.1291,
    0.2535,
    0.1913,
    0.1913,
    0.5023,
    0.3779,
]

acc_drd = [
    0.8101,
    0.8156,
    0.8198,
    0.8242,
    0.8301,
    0.8375,
    0.8424,
    0.8468,
    0.852,
    0.8518,
]
acc_drd_svt = [
    0.8101,
    0.8168,
    0.8223,
    0.8279,
    0.8324,
    0.8405,
    0.845,
    0.8464,
    0.852,
    0.8553,
]
acc_dred = [
    0.8101,
    0.8156,
    0.8198,
    0.8242,
    0.8301,
    0.8375,
    0.842,
    0.8468,
    0.852,
    0.8518,
]
acc_dred_svt = [
    0.8101,
    0.8169,
    0.8231,
    0.8264,
    0.8344,
    0.8405,
    0.8438,
    0.8529,
    0.852,
    0.8563,
]

cr_drd = [cr * 100 for cr in cr_drd]
cr_drd_svt = [cr * 100 for cr in cr_drd_svt]
cr_dred = [cr * 100 for cr in cr_dred]
cr_dred_svt = [cr * 100 for cr in cr_dred_svt]

ax[0].set_xlabel("Epsilon")
ax[0].set_ylabel("C.R.")  # we already handled the x-label with ax1
ax[0].bar(xs - 0.3, cr_drd, 0.2, label="DRD")
ax[0].bar(xs - 0.1, cr_drd_svt, 0.2, label="DRD-SVT")
ax[0].bar(xs + 0.1, cr_dred, 0.2, label="DRED")
ax[0].bar(xs + 0.3, cr_dred_svt, 0.2, label="DRED-SVT")

# ax1.bar(xs, cr_halving)
ax[0].set_ylim([0, 60])
ax[0].tick_params(axis="y")
# ax[0].legend(loc="lower left")

ax0_twin = ax[0].twinx()  # instantiate a second Axes that shares the same x-axis
ax0_twin.set_ylabel("Accuracy")
ax0_twin.plot(xs, acc_drd, "-", label="DRD")
ax0_twin.plot(xs, acc_drd_svt, "--", label="DRD-SVT")
ax0_twin.plot(xs, acc_dred, "-", label="DRED")
ax0_twin.plot(xs, acc_dred_svt, "--", label="DRED-SVT")
ax0_twin.set_ylim([0.80, 0.86])
ax0_twin.tick_params(axis="y")

ax[0].set_xticks(np.arange(0, 11))
ax[0].set_xticklabels([0.0, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
# ax0_twin.legend(loc="upper left")
ax[0].title.set_text("B2. ViT CIFAR100")


##############################################

xs = np.arange(1, 11)
cr_drd = [0.0, 0.0861, 0.0861, 0.1896, 0.3202, 0.4552, 0.3877, 0.3877, 0.5858, 0.4552]
cr_drd_svt = [
    0.0,
    0.0861,
    0.1221,
    0.1896,
    0.3562,
    0.4237,
    0.4237,
    0.4552,
    0.5858,
    0.4913,
]
cr_dred = [0.0, 0.0861, 0.0861, 0.2526, 0.3202, 0.3562, 0.4552, 0.5228, 0.8559, 0.8559]
cr_dred_svt = [
    0.0,
    0.0861,
    0.1041,
    0.2526,
    0.4552,
    0.5228,
    0.5228,
    0.5858,
    0.8559,
    0.8559,
]

acc_drd = [
    0.7992,
    0.8016606680007535,
    0.8018297383283854,
    0.8077960745292918,
    0.8129972073396668,
    0.8141318626411452,
    0.8153905042987132,
    0.8163385544113712,
    0.8235209515381166,
    0.8236173849141188,
]
acc_drd_svt = [
    0.7992,
    0.8016606680007535,
    0.8022229851110344,
    0.8077960745292918,
    0.8142545951080399,
    0.8166666779510006,
    0.8188833900377515,
    0.8215860255602283,
    0.8235209515381166,
    0.8264565295368573,
]
acc_dred = [
    0.7992,
    0.8016606680007535,
    0.8018297383283854,
    0.8110973464500306,
    0.8129972073396668,
    0.8142971757919943,
    0.8238515796074706,
    0.8279806750891574,
    0.8333195684047157,
    0.8364292277015172,
]
acc_dred_svt = [
    0.7992,
    0.8016606680007535,
    0.8014440052304594,
    0.8110973464500306,
    0.8184625902033535,
    0.8273444666841945,
    0.8291704371629766,
    0.830931282320549,
    0.8333195684047157,
    0.8364292277015172,
]

cr_drd = [cr * 100 for cr in cr_drd]
cr_drd_svt = [cr * 100 for cr in cr_drd_svt]
cr_dred = [cr * 100 for cr in cr_dred]
cr_dred_svt = [cr * 100 for cr in cr_dred_svt]

ax[1].set_xlabel("Epsilon")
ax[1].set_ylabel("C.R.")  # we already handled the x-label with ax1
ax[1].bar(xs - 0.3, cr_drd, 0.2, label="DRD")
ax[1].bar(xs - 0.1, cr_drd_svt, 0.2, label="DRD-SVT")
ax[1].bar(xs + 0.1, cr_dred, 0.2, label="DRED")
ax[1].bar(xs + 0.3, cr_dred_svt, 0.2, label="DRED-SVT")

# ax1.bar(xs, cr_halving)
ax[1].set_ylim([0, 100])
ax[1].tick_params(axis="y")
# ax[1].legend(loc="lower left")

ax1_twin = ax[1].twinx()  # instantiate a second Axes that shares the same x-axis
ax1_twin.set_ylabel("Accuracy")
ax1_twin.plot(xs, acc_drd, "-", label="DRD")
ax1_twin.plot(xs, acc_drd_svt, "--", label="DRD-SVT")
ax1_twin.plot(xs, acc_dred, "-", label="DRED")
ax1_twin.plot(xs, acc_dred_svt, "--", label="DRED-SVT")
ax1_twin.set_ylim([0.78, 0.84])
ax1_twin.tick_params(axis="y")

ax[1].set_xticks(np.arange(0, 11))
ax[1].set_xticklabels([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
# ax1_twin.legend(loc="upper left")
ax[1].title.set_text("B3. ResNet CelebA")


##############################################
line1 = Line2D([0], [0], label="DRD Acc.", linestyle="-", color="C0")
line2 = Line2D([0], [0], label="DRD-SVT Acc.", linestyle="--", color="C1")
line3 = Line2D([0], [0], label="DRED Acc.", linestyle="-", color="C2")
line4 = Line2D([0], [0], label="DRED-SVT Acc.", linestyle="--", color="C3")
patch1 = mpatches.Patch(color="C0", label="DRD C.R.")
patch2 = mpatches.Patch(color="C1", label="DRD-SVT C.R.")
patch3 = mpatches.Patch(color="C2", label="DRED C.R.")
patch4 = mpatches.Patch(color="C3", label="DRED-SVT C.R.")
handles = [line1, patch1, line2, patch2, line3, patch3, line4, patch4]

# Use fig.legend instead of plt.legend to have better control of the position
fig.legend(
    handles=handles,
    loc="upper left",
    ncol=8,
    bbox_to_anchor=(0.02, 0.13),
    bbox_transform=fig.transFigure,
)

fig.tight_layout()  # otherwise the right y-label is slightly clipped

# Adjust subplot spacing to make room for the legend at the bottom
plt.subplots_adjust(bottom=0.25, left=0.1, right=0.9, hspace=0.2)


# plt.show()

plt.savefig("svt_2x1.pdf")
