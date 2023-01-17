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
        lambda row: row.HDI_95[0] <= row.prevalence <= row.HDI_95[1],
        axis=1,
    )
    test_df["HDI_99_covers"] = test_df.apply(
        lambda row: row.HDI_99[0] <= row.prevalence <= row.HDI_99[1],
        axis=1,
    )

    coverage_90 = test_df.HDI_90_covers.sum() / len(test_df)
    coverage_95 = test_df.HDI_95_covers.sum() / len(test_df)
    coverage_99 = test_df.HDI_99_covers.sum() / len(test_df)

    data["coverage_90"] = coverage_90
    data["coverage_95"] = coverage_95
    data["coverage_99"] = coverage_99

    test_df["HDI_90_pos"] = test_df.apply(
        lambda row: highest_density_interval(
            row.N_inlier + row.N_outlier,
            row.K_outlier + row.N_inlier - row.K_inlier,
            row.sens_alpha,
            row.sens_beta,
            row.spec_alpha,
            row.spec_beta,
            alpha=0.1,
            mode="positive",
        ),
        axis=1,
    )

    test_df["HDI_95_pos"] = test_df.apply(
        lambda row: highest_density_interval(
            row.N_inlier + row.N_outlier,
            row.K_outlier + row.N_inlier - row.K_inlier,
            row.sens_alpha,
            row.sens_beta,
            row.spec_alpha,
            row.spec_beta,
            alpha=0.05,
            mode="positive",
        ),
        axis=1,
    )

    test_df["HDI_99_pos"] = test_df.apply(
        lambda row: highest_density_interval(
            row.N_inlier + row.N_outlier,
            row.K_outlier + row.N_inlier - row.K_inlier,
            row.sens_alpha,
            row.sens_beta,
            row.spec_alpha,
            row.spec_beta,
            alpha=0.01,
            mode="positive",
        ),
        axis=1,
    )

    test_df["precision"] = test_df.apply(
        lambda row: row.K_outlier / (row.K_outlier + row.N_inlier - row.K_inlier),
        axis=1,
    )


    test_df["HDI_90_pos_covers"] = test_df.apply(
        lambda row: row.HDI_90_pos[0] <= row.precision <= row.HDI_90_pos[1],
        axis=1,
    )
    test_df["HDI_95_pos_covers"] = test_df.apply(
        lambda row: row.HDI_95_pos[0] <= row.precision <= row.HDI_95_pos[1],
        axis=1,
    )
    test_df["HDI_99_pos_covers"] = test_df.apply(
        lambda row: row.HDI_99_pos[0] <= row.precision <= row.HDI_99_pos[1],
        axis=1,
    )

    coverage_90_pos = test_df.HDI_90_pos_covers.sum() / len(test_df)
    coverage_95_pos = test_df.HDI_95_pos_covers.sum() / len(test_df)
    coverage_99_pos = test_df.HDI_99_pos_covers.sum() / len(test_df)

    data["coverage_90_pos"] = coverage_90_pos
    data["coverage_95_pos"] = coverage_95_pos
    data["coverage_99_pos"] = coverage_99_pos

    test_df["HDI_90_neg"] = test_df.apply(
        lambda row: highest_density_interval(
            row.N_inlier + row.N_outlier,
            row.K_outlier + row.N_inlier - row.K_inlier,
            row.sens_alpha,
            row.sens_beta,
            row.spec_alpha,
            row.spec_beta,
            alpha=0.1,
            mode="negative",
        ),
        axis=1,
    )

    test_df["HDI_95_neg"] = test_df.apply(
        lambda row: highest_density_interval(
            row.N_inlier + row.N_outlier,
            row.K_outlier + row.N_inlier - row.K_inlier,
            row.sens_alpha,
            row.sens_beta,
            row.spec_alpha,
            row.spec_beta,
            alpha=0.05,
            mode="negative",
        ),
        axis=1,
    )

    test_df["HDI_99_neg"] = test_df.apply(
        lambda row: highest_density_interval(
            row.N_inlier + row.N_outlier,
            row.K_outlier + row.N_inlier - row.K_inlier,
            row.sens_alpha,
            row.sens_beta,
            row.spec_alpha,
            row.spec_beta,
            alpha=0.01,
            mode="negative",
        ),
        axis=1,
    )

    test_df["FOR"] = test_df.apply(
        lambda row: 0 if row.N_outlier - row.K_outlier == 0
        else (row.N_outlier - row.K_outlier)
        / (row.K_inlier + row.N_outlier - row.K_outlier),
        axis=1,
    )


    test_df["HDI_90_neg_covers"] = test_df.apply(
        lambda row: row.HDI_90_neg[0] <= row.FOR <= row.HDI_90_neg[1],
        axis=1,
    )
    test_df["HDI_95_neg_covers"] = test_df.apply(
        lambda row: row.HDI_95_neg[0] <= row.FOR <= row.HDI_95_neg[1],
        axis=1,
    )
    test_df["HDI_99_neg_covers"] = test_df.apply(
        lambda row: row.HDI_99_neg[0] <= row.FOR <= row.HDI_99_neg[1],
        axis=1,
    )

    coverage_90_neg = test_df.HDI_90_neg_covers.sum() / len(test_df)
    coverage_95_neg = test_df.HDI_95_neg_covers.sum() / len(test_df)
    coverage_99_neg = test_df.HDI_99_neg_covers.sum() / len(test_df)

    data["coverage_90_neg"] = coverage_90_neg
    data["coverage_95_neg"] = coverage_95_neg
    data["coverage_99_neg"] = coverage_99_neg

