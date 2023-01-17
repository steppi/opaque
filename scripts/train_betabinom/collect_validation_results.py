import argparse

from opaque.results import OpaqueResultsManager
from opaque.stats import highest_density_interval


run_name = "validate_run1"

results = list(OpaqueResultsManager.iterrows("validate_run1"))


for key, data in results:
    test_df = data["test_df"]
    test_df["HDI_90"] = test_df.apply(
        lambda row: highest_density_interval(
            row.N_inlier + row.N_outlier,
            row.K_outlier + row.N_inlier - row.K_inlier,
            row.sens_alpha,
            row.sens_beta,
            row.spec_alpha,
            row.spec_beta,
            alpha=0.1,
        ),
        axis=1,
    )

    test_df["HDI_95"] = test_df.apply(
        lambda row: highest_density_interval(
            row.N_inlier + row.N_outlier,
            row.K_outlier + row.N_inlier - row.K_inlier,
            row.sens_alpha,
            row.sens_beta,
            row.spec_alpha,
            row.spec_beta,
            alpha=0.05,
        ),
        axis=1,
    )

    test_df["HDI_99"] = test_df.apply(
        lambda row: highest_density_interval(
            row.N_inlier + row.N_outlier,
            row.K_outlier + row.N_inlier - row.K_inlier,
            row.sens_alpha,
            row.sens_beta,
            row.spec_alpha,
            row.spec_beta,
            alpha=0.01,
        ),
        axis=1,
    )

    test_df["prevalence"] = test_df.apply(
        lambda row: row.N_outlier / (row.N_outlier + row.N_inlier),
        axis=1,
    )
    test_df["HDI_90_covers"] = test_df.apply(
        lambda row: row.HDI_90[0] <= row.prevalence <= row.HDI_90[1],
        axis=1,
    )
    test_df["HDI_95_covers"] = test_df.apply(
        lambda row: row.HDI_90[0] <= row.prevalence <= row.HDI_95[1],
        axis=1,
    )
    test_df["HDI_99_covers"] = test_df.apply(
        lambda row: row.HDI_90[0] <= row.prevalence <= row.HDI_99[1],
        axis=1,
    )
    coverage_90 = test_df.HDI_90_covers.sum() / len(test_df)
    coverage_95 = test_df.HDI_95_covers.sum() / len(test_df)
    coverage_99 = test_df.HDI_99_covers.sum() / len(test_df)

    data["coverage_90"] = coverage_90
    data["coverage_95"] = coverage_95
    data["coverage_99"] = coverage_99

