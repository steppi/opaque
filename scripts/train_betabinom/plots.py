import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from io import StringIO
from scipy.optimize._zeros_py import _differentiate

from opaque.stats import prevalence_cdf


with open("validation_run1_results.json", "rb") as f:
    results = json.load(f)


for key, data in results:
    data["test_df"] = pd.read_json(StringIO(data["test_df"]))

agg = []
for _, data in results:
    df = data["test_df"]
    agg.append(df)
df = pd.concat(agg)


df['sens'] = df.K_inlier / df.N_inlier
df['spec'] = df.K_outlier / df.N_outlier

df["90% HDI covers"] = df["HDI_90_covers"]

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
sns.scatterplot(
    data=df,
    x="spec",
    y="sens",
    ax=ax,
    clip_on=False,
    hue="90% HDI covers",
    hue_order=[True, False],
    palette="colorblind",
)

ax.grid(which="major", color="black", linestyle="-", alpha=0.4)
ax.grid(which="minor", color="gray", linestyle="--", alpha=0.4)
ax.set_xlabel("Specificity")
ax.set_ylabel("Sensitivity")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xticks(np.linspace(0, 1, 11))
ax.set_yticks(np.linspace(0, 1, 11))
plt.minorticks_on()
plt.tight_layout()
plt.savefig("results_scatterplot.png")
plt.close()


example = df.iloc[2]
prevalence = example.prevalence
HDI_left, HDI_right = example.HDI_90
N = example.N_inlier + example.N_outlier
t = example.K_outlier + example.N_inlier - example.K_inlier
sens_alpha, sens_beta = example.sens_alpha, example.sens_beta
spec_alpha, spec_beta = example.spec_alpha, example.spec_beta

theta = np.linspace(0, 1, 2001)
y = _differentiate(
    lambda x: prevalence_cdf(x, N, t, sens_alpha, sens_beta, spec_alpha, spec_beta), theta,
    rtol=1e-4,
).df

theta = theta[10:-10]
y = y[10:-10]


fig, ax = plt.subplots(1, 1)
prev_x = [prevalence, prevalence]
prev_y = [
    0,
    _differentiate(
        lambda x: prevalence_cdf(x, N, t, sens_alpha, sens_beta, spec_alpha, spec_beta), [prevalence],
        rtol=1e-4,
    ).df.take(0)
]

ax.plot(prev_x, prev_y, color="black", linestyle="--")
ax.plot(theta, y)
ax.set_xlim(0, 1)
ax.set_ylim(0)
ax.grid(which="major", color="black", linestyle="-", alpha=0.4)
ax.grid(which="minor", color="gray", linestyle="--", alpha=0.4)
ax.fill_between(
    theta[(HDI_left <= theta) & (theta <= HDI_right)],
    y[(HDI_left <= theta) & (theta <= HDI_right)],
    color="blue",
    alpha=0.5,
)
plt.minorticks_on()
plt.tight_layout()
plt.savefig("example_interval.png")
plt.close()
