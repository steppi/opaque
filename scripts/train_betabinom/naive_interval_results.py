import argparse
import pickle

from opaque.results import OpaqueResultsManager
from opaque.stats import simple_prevalence_interval

run_name = "validate_run1"

results = list(OpaqueResultsManager.iterrows("validate_run1"))
for key, data in results:
    test_df = data["test_df"]

    test_df["SI_90"] = test_df.apply(
        lambda row: simple_prevalence_interval(
            row.N_inlier + row.N_outlier,
            row.K_outlier + row.N_inlier - row.K_inlier,
            row.sens_neg_set,
            row.mean_spec,
            alpha=0.1,
        ),
        axis=1,
    )

    test_df["SI_95"] = test_df.apply(
        lambda row: simple_prevalence_interval(
            row.N_inlier + row.N_outlier,
            row.K_outlier + row.N_inlier - row.K_inlier,
            row.sens_neg_set,
            row.mean_spec,
            alpha=0.05,
        ),
        axis=1,
    )

    test_df["SI_99"] = test_df.apply(
        lambda row: simple_prevalence_interval(
            row.N_inlier + row.N_outlier,
            row.K_outlier + row.N_inlier - row.K_inlier,
            row.sens_neg_set,
            row.mean_spec,
            alpha=0.01,
        ),
        axis=1,
    )

    test_df["prevalence"] = test_df.apply(
        lambda row: row.N_outlier / (row.N_outlier + row.N_inlier),
        axis=1,
    )

    test_df["SI_90_covers"] = test_df.apply(
        lambda row: row.SI_90[0] <= row.prevalence <= row.SI_90[1],
        axis=1,
    )
    test_df["SI_95_covers"] = test_df.apply(
        lambda row: row.SI_95[0] <= row.prevalence <= row.SI_95[1],
        axis=1,
    )
    test_df["SI_99_covers"] = test_df.apply(
        lambda row: row.SI_99[0] <= row.prevalence <= row.SI_99[1],
        axis=1,
    )

    test_df["length_90"] = test_df.SI_90.apply(lambda x: x[1] - x[0])
    test_df["length_95"] = test_df.SI_95.apply(lambda x: x[1] - x[0])
    test_df["length_99"] = test_df.SI_99.apply(lambda x: x[1] - x[0])


    data["coverage_90"] = test_df.SI_90_covers.sum() / len(test_df)
    data["coverage_95"] = test_df.SI_95_covers.sum() / len(test_df)
    data["coverage_99"] = test_df.SI_99_covers.sum() / len(test_df)

    data["length_90_mean"] = test_df.length_90.mean()
    data["length_90_std"] = test_df.length_90.std()
    data["length_95_mean"] = test_df.length_95.mean()
    data["length_95_std"] = test_df.length_95.std()
    data["length_99_mean"] = test_df.length_99.mean()
    data["length_99_std"] = test_df.length_99.std()
