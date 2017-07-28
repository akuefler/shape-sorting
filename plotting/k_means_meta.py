import numpy as np
import h5py

with h5py.File("k_means_meta.h5","r") as hf:
    spread = hf["spread"][...]
    acc = hf["acc"][...]

"""
Both the spread and the accuracy increase when the cursor appears
"""
print("Average Spread, no cursor :: Val: {}; Adv: {}".format(spread[0][-2].mean(),
    spread[0][-1].mean()))
print("Average Spread, w cursor :: Val: {}; Adv: {}".format(spread[1][-2].mean(),
    spread[1][-1].mean()))

print("Average Accuracy, no cursor :: Val: {}; Adv: {}".format(acc[0][-2].mean(), acc[0][-1].mean()))
print("Average Accuracy, w cursor :: Val: {}; Adv: {}".format(acc[1][-2].mean(), acc[1][-1].mean()))

for m_name, metric in [("Spread", spread),("Acc", acc)]:
    print("Metric: {}".format(m_name))
    for c_name, condition in zip(["no grab", "grab"],metric):
        print("Condition: {}".format(c_name))
        for l, l_name in enumerate(["C1","C2","C3","VL","AV"]):
            print("{}: {}; {}".format(l_name, condition[l],
                np.mean(condition[l])))
