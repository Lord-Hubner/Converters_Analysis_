import pandas as pd

#base = pd.read_csv("metadata.csv")

# map = {
#     2: "NoDifference",
#     0: "Upregulated",
#     1: "Downregulated"
# }

# rs = []
# for a in base["label"]:
#     rs.append(map[a])

# paths = []
# filenames = []
# for i, a in enumerate(base["image_path"]):
#     filenames.append(a.split("/", -1)[-1])
#     data = base.iloc[i]
#     paths.append(f"HACNet_PRAD_IMGS/fold_{data["fold"]}/{map[data["label"]]}"+f"/sample_{data["sample_idx"]:06d}.png")


# output = pd.DataFrame({
#     "fold": base["fold"],
#     "sample_idx": base["sample_idx"],
#     "filename": filenames,
#     "label": base["label"],
#     "class": rs,
#     "path": paths

# }).to_csv("labels.csv", index=False)

labels = pd.read_csv("labels.csv")

labels_1 = labels[labels["fold"] == 1]
labels_2 = labels[labels["fold"] == 2]
labels_3 = labels[labels["fold"] == 3]
labels_4 = labels[labels["fold"] == 4]
labels_5 = labels[labels["fold"] == 5]

labels_1.to_csv("labels_1.csv", index=False)
labels_2.to_csv("labels_2.csv", index=False)
labels_3.to_csv("labels_3.csv", index=False)
labels_4.to_csv("labels_4.csv", index=False)
labels_5.to_csv("labels_5.csv", index=False)