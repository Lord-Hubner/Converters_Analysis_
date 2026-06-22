import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

data = pd.read_csv("KIRC_features.csv", index_col=1)

# Example data
relevant_data = data[["log2FC", "padj"]]
positiveLog2 = relevant_data[relevant_data["log2FC"] >= 0]
negativeLog2 = relevant_data[relevant_data["log2FC"] <= 0]

maxUp = []
currentMin = (math.inf, math.inf)
update = True
i=0
for name, (log2, padj) in positiveLog2.iterrows():
    product = log2*-np.log10(padj)
    if len(maxUp) < 10:
        maxUp.append((i, name, product))
        i+=1
        continue
    if update:
        for list_idx, (_, _, value) in enumerate(maxUp):
            if value<currentMin[1]:
                currentMin = (list_idx, value) 
    update = False
    if product > currentMin[1]:
        maxUp[currentMin[0]] = (i, name, product)
        update=True
        currentMin = (math.inf, math.inf)
    i+=1

maxDown = []
currentMin = (math.inf, math.inf)
update = True
i=0
for name, (log2, padj) in negativeLog2.iterrows():
    product = (-log2)*-np.log10(padj)
    if len(maxDown) < 10:
        maxDown.append((i, name, product))
        i+=1
        continue
    if update:
        for list_idx, (_, _, value) in enumerate(maxDown):
            if value<currentMin[1]:
                currentMin = (list_idx, value) 
    update = False
    if product > currentMin[1]:
        maxDown[currentMin[0]] = (i, name, product)
        update=True
        currentMin= (math.inf, math.inf)
    i+=1


for a in maxUp:
    print(a, sep=" ", end="")
print("")
for a in maxDown:
    print(a, sep=" ", end="")


plt.scatter(relevant_data["log2FC"], -np.log10(relevant_data["padj"]), color="black", alpha=0.6)
plt.axhline(-np.log10(0.05), color="red", linestyle="--", label="p=0.05")
plt.axvline(-1, color="gray", linestyle="--")
plt.axvline(1, color="gray", linestyle="--")

for i, name, product in maxUp:
    plt.text(relevant_data.loc[name]["log2FC"], -np.log10(relevant_data.loc[name]["padj"]), name, fontsize=8, color="blue")

for i, name, product in maxDown:
    plt.text(relevant_data.loc[name]["log2FC"], -np.log10(relevant_data.loc[name]["padj"]), name, fontsize=8, color="blue")

plt.xlabel("Log2 Fold Change")
plt.ylabel("-log10(p-value)")
plt.title("Volcano Plot")

plt.legend()
plt.legend()
plt.show()