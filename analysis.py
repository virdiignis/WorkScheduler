import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# sns.set_theme()

# DATA ORDER
# random instances, generation instances, mutation variant, pairing variant, breeding variant, fuzzy variant, i, mutation time, ranking time, crossing time, time passed, best result, worst result

names = ["random instances", "generation instances", "mutation variant", "pairing variant", "breeding variant",
         "fuzzy variant", "i", "mutation time", "ranking time", "crossing time", "ranking time2", "time passed",
         "best result", "worst result"]
types = dict(zip(names,
                 ["uint16", "uint16", "uint8", "uint8", "uint8", "uint8", "uint32", "float64", "float64", "float64",
                  "float64", "float64", "int32", "int32"]))

data = pd.read_csv("/tmp/all.csv", header=None, names=names, dtype=types)
data = data.drop(columns=["i", "mutation time", "ranking time", "crossing time", "ranking time2"])

# print(data)
# dt = data.groupby(["random instances", "generation instances", "mutation variant", "pairing variant",
#                    "breeding variant", "fuzzy variant"])

# data = data[data["best result"] == data["best result"].max()]
# data = data[(data["random instances"] == 1000) & (data["generation instances"] == 10)]
maxr = data["best result"].max() * 0.89

# data = data.drop(columns=["random instances", "generation instances"])
data.set_index(["time passed"], inplace=True)
group = data.groupby(["random instances", "generation instances", "mutation variant", "pairing variant", "breeding variant", "fuzzy variant"])
group = group.filter(lambda x: x["best result"].max() >= maxr)
group = group.groupby(["random instances", "generation instances", "mutation variant", "pairing variant", "breeding variant", "fuzzy variant"])

# print(group)
# exit()
ax = plt.gca()
group["best result"].plot(ax=ax, legend=True)
ax.legend()
# group.plot(y=["best result", "worst result"], ax=ax, legend=True)

plt.savefig("figs/top10.png")
