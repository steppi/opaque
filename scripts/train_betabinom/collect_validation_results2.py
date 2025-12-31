import argparse

from multiprocessing import Pool
from opaque.results import OpaqueResultsManager
from opaque.stats import equal_tailed_interval, highest_density_interval


def process_job(
        key, N_inlier, N_outlier, K_inlier, K_outlier,
        sens_alpha, sens_beta, spec_alpha, spec_beta
):
    results = {"key": key}

    results["ETI_90"] = equal_tailed_interval(
        N_inlier + N_outlier,
        K_outlier + N_inlier - K_inlier,
        sens_alpha, sens_beta, spec_alpha, spec_beta,
        alpha=0.1,
    )
    results["ETI_95"] = equal_tailed_interval(
        N_inlier + N_outlier,
        K_outlier + N_inlier - K_inlier,
        sens_alpha, sens_beta, spec_alpha, spec_beta,
        alpha=0.05,
    )
    results["ETI_99"] = equal_tailed_interval(
        N_inlier + N_outlier,
        K_outlier + N_inlier - K_inlier,
        sens_alpha, sens_beta, spec_alpha, spec_beta,
        alpha=0.01,
    )

    results["ETI_90_pos"] = equal_tailed_interval(
        N_inlier + N_outlier,
        K_outlier + N_inlier - K_inlier,
        sens_alpha, sens_beta, spec_alpha, spec_beta,
        alpha=0.1,
        mode="positive",
    )
    results["ETI_95_pos"] = equal_tailed_interval(
        N_inlier + N_outlier,
        K_outlier + N_inlier - K_inlier,
        sens_alpha, sens_beta, spec_alpha, spec_beta,
        alpha=0.05,
        mode="positive",
    )
    results["ETI_99_pos"] = equal_tailed_interval(
        N_inlier + N_outlier,
        K_outlier + N_inlier - K_inlier,
        sens_alpha, sens_beta, spec_alpha, spec_beta,
        alpha=0.01,
        mode="positive",
    )

    results["ETI_90_neg"] = equal_tailed_interval(
        N_inlier + N_outlier,
        K_outlier + N_inlier - K_inlier,
        sens_alpha, sens_beta, spec_alpha, spec_beta,
        alpha=0.1,
        mode="negative",
    )
    results["ETI_95_neg"] = equal_tailed_interval(
        N_inlier + N_outlier,
        K_outlier + N_inlier - K_inlier,
        sens_alpha, sens_beta, spec_alpha, spec_beta,
        alpha=0.05,
        mode="negative",
    )
    results["ETI_99_neg"] = equal_tailed_interval(
        N_inlier + N_outlier,
        K_outlier + N_inlier - K_inlier,
        sens_alpha, sens_beta, spec_alpha, spec_beta,
        alpha=0.01,
        mode="negative",
    )

    results["HDI_90"] = highest_density_interval(
        N_inlier + N_outlier,
        K_outlier + N_inlier - K_inlier,
        sens_alpha, sens_beta, spec_alpha, spec_beta,
        alpha=0.1,
    )
    results["HDI_95"] = highest_density_interval(
        N_inlier + N_outlier,
        K_outlier + N_inlier - K_inlier,
        sens_alpha, sens_beta, spec_alpha, spec_beta,
        alpha=0.05,
    )
    results["HDI_99"] = highest_density_interval(
        N_inlier + N_outlier,
        K_outlier + N_inlier - K_inlier,
        sens_alpha, sens_beta, spec_alpha, spec_beta,
        alpha=0.01,
    )

    results["HDI_90_pos"] = highest_density_interval(
        N_inlier + N_outlier,
        K_outlier + N_inlier - K_inlier,
        sens_alpha, sens_beta, spec_alpha, spec_beta,
        alpha=0.1,
        mode="positive",
    )
    results["HDI_95_pos"] = highest_density_interval(
        N_inlier + N_outlier,
        K_outlier + N_inlier - K_inlier,
        sens_alpha, sens_beta, spec_alpha, spec_beta,
        alpha=0.05,
        mode="positive",
    )
    results["HDI_99_pos"] = highest_density_interval(
        N_inlier + N_outlier,
        K_outlier + N_inlier - K_inlier,
        sens_alpha, sens_beta, spec_alpha, spec_beta,
        alpha=0.01,
        mode="positive",
    )

    results["HDI_90_neg"] = highest_density_interval(
        N_inlier + N_outlier,
        K_outlier + N_inlier - K_inlier,
        sens_alpha, sens_beta, spec_alpha, spec_beta,
        alpha=0.1,
        mode="negative",
    )
    results["HDI_95_neg"] = highest_density_interval(
        N_inlier + N_outlier,
        K_outlier + N_inlier - K_inlier,
        sens_alpha, sens_beta, spec_alpha, spec_beta,
        alpha=0.05,
        mode="negative",
    )
    results["HDI_99_neg"] = highest_density_interval(
        N_inlier + N_outlier,
        K_outlier + N_inlier - K_inlier,
        sens_alpha, sens_beta, spec_alpha, spec_beta,
        alpha=0.01,
        mode="negative",
    )

    results["prevalence"] = N_outlier / (N_outlier + N_inlier)
    results["precision"] = K_outlier / (K_outlier + N_inlier - K_inlier)
    false_omission_rate = N_outlier - K_outlier
    if false_omission_rate != 0:
        false_omission_rate /= (K_inlier + N_outlier - K_outlier)
    results["false_omission_rate"] = false_omission_rate

    return results


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("run_name")
    parser.add_argument("outpath")
    args = parser.parse_args()

    results = OpaqueResultsManager.iterrows(args.run_name)

    cases = []
    for key, outer_row in results:
        test_df = outer_row["test_df"]
        for _, row  in test_df.iterrows():
            cases.append(
                [
                    key,
                    row.N_inlier,
                    row.N_outlier,
                    row.K_inlier,
                    row.K_outlier,
                    row.sens_alpha,
                    row.sens_beta,
                    row.spec_alpha,
                    row.spec_beta
                ]
            )

    with Pool(16) as pool:
        results = pool.starmap(process_job, cases)
