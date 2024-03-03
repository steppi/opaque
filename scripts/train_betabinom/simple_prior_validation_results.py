import argparse
import ast
import dask.dataframe as dd
import json
import multiprocessing
import pandas as pd


from opaque.results import OpaqueResultsManager
from opaque.stats import highest_density_interval


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("run_name")
    parser.add_argument("outpath")
    args = parser.parse_args()

    results = list(OpaqueResultsManager.iterrows(args.run_name))


    cpu_count = multiprocessing.cpu_count()

    for key, data in results:

        test_df = data["test_df"]
        # This block of lines is use to calculate simple priors which take into account
        # only the anomaly detection models performance on its own validation
        # data, without incorporating info from validation of all other anomaly
        # detection models.
        test_df["N_spec"] = (test_df.num_entrez + test_df.num_mesh) / 5
        test_df["mean_spec_smooth"] = (
            (test_df.N_spec * test_df.mean_spec + 1) / (test_df.N_spec + 2)
        )
        test_df["std_spec_smooth"] = (
            (test_df.N_spec)/(test_df.N_spec + 2) * test_df.std_spec
        )
        
        test_df["simple_spec_nu"] = (
            test_df.mean_spec_smooth*(1 - test_df.mean_spec_smooth) /
            test_df.std_spec_smooth**2 - 1
        )
        test_df["simple_spec_alpha"] = (
            test_df.mean_spec_smooth * test_df.simple_spec_nu
        )
        test_df["simple_spec_beta"] = (
            (1-test_df.mean_spec_smooth)*test_df.simple_spec_nu
        )
        
        test_df[["simple_spec_alpha", "simple_spec_beta"]] = test_df.apply(
            lambda row: pd.Series([1, 1]) if row.std_spec_smooth == 0 else
            pd.Series([row.simple_spec_alpha, row.simple_spec_beta]), axis=1
        )
        
        test_df["sens_neg_set_smooth"] = (50000 * test_df.sens_neg_set + 1) / 50002
        test_df["simple_sens_nu"] = 50002
        test_df["simple_sens_alpha"] = (
            test_df.sens_neg_set_smooth * test_df.simple_sens_nu
        )
        test_df["simple_sens_beta"] = (
            (1 - test_df.sens_neg_set) * test_df.simple_sens_nu
        )
        # --------------------------------------------------------------

    
        len_test_df = len(data["test_df"])
        test_df = dd.from_pandas(data["test_df"], npartitions=cpu_count)

        test_df["HDI_90"] = test_df.apply(
            lambda row: highest_density_interval(
                row.N_inlier + row.N_outlier,
                row.K_outlier + row.N_inlier - row.K_inlier,
                row.simple_sens_alpha,
                row.simple_sens_beta,
                row.simple_spec_alpha,
                row.simple_spec_beta,
                alpha=0.1,
            ),
            axis=1,
            meta=("HDI_90", "object"),
        ).compute()

        test_df["HDI_95"] = test_df.apply(
            lambda row: highest_density_interval(
                row.N_inlier + row.N_outlier,
                row.K_outlier + row.N_inlier - row.K_inlier,
                row.simple_sens_alpha,
                row.simple_sens_beta,
                row.simple_spec_alpha,
                row.simple_spec_beta,
                alpha=0.05,
            ),
            axis=1,
            meta=("HDI_95", "object"),
        ).compute()

        test_df["HDI_99"] = test_df.apply(
            lambda row: highest_density_interval(
                row.N_inlier + row.N_outlier,
                row.K_outlier + row.N_inlier - row.K_inlier,
                row.simple_sens_alpha,
                row.simple_sens_beta,
                row.simple_spec_alpha,
                row.simple_spec_beta,
                alpha=0.01,
            ),
            axis=1,
            meta=("HDI_99", "object"),
        ).compute()


        test_df["HDI_90_pos"] = test_df.apply(
            lambda row: highest_density_interval(
                row.N_inlier + row.N_outlier,
                row.K_outlier + row.N_inlier - row.K_inlier,
                row.simple_sens_alpha,
                row.simple_sens_beta,
                row.simple_spec_alpha,
                row.simple_spec_beta,
                alpha=0.1,
                mode="positive",
            ),
            axis=1,
            meta=("HDI_90_pos", "object"),
        ).compute()
        test_df["HDI_95_pos"] = test_df.apply(
            lambda row: highest_density_interval(
                row.N_inlier + row.N_outlier,
                row.K_outlier + row.N_inlier - row.K_inlier,
                row.simple_sens_alpha,
                row.simple_sens_beta,
                row.simple_spec_alpha,
                row.simple_spec_beta,
                alpha=0.05,
                mode="positive",
            ),
            axis=1,
            meta=("HDI_95_pos", "object"),
        ).compute()
        test_df["HDI_99_pos"] = test_df.apply(
            lambda row: highest_density_interval(
                row.N_inlier + row.N_outlier,
                row.K_outlier + row.N_inlier - row.K_inlier,
                row.simple_sens_alpha,
                row.simple_sens_beta,
                row.simple_spec_alpha,
                row.simple_spec_beta,
                alpha=0.01,
                mode="positive",
            ),
            axis=1,
            meta=("HDI_99_pos", "object"),
        ).compute()

        test_df["HDI_90_neg"] = test_df.apply(
            lambda row: highest_density_interval(
                row.N_inlier + row.N_outlier,
                row.K_outlier + row.N_inlier - row.K_inlier,
                row.simple_sens_alpha,
                row.simple_sens_beta,
                row.simple_spec_alpha,
                row.simple_spec_beta,
                alpha=0.1,
                mode="negative",
            ),
            axis=1,
            meta=("HDI_90_neg", "object"),
        ).compute()

        test_df["HDI_95_neg"] = test_df.apply(
            lambda row: highest_density_interval(
                row.N_inlier + row.N_outlier,
                row.K_outlier + row.N_inlier - row.K_inlier,
                row.simple_sens_alpha,
                row.simple_sens_beta,
                row.simple_spec_alpha,
                row.simple_spec_beta,
                alpha=0.05,
                mode="negative",
            ),
            axis=1,
            meta=("HDI_95_neg", "object"),
        ).compute()

        test_df["HDI_99_neg"] = test_df.apply(
            lambda row: highest_density_interval(
                row.N_inlier + row.N_outlier,
                row.K_outlier + row.N_inlier - row.K_inlier,
                row.simple_sens_alpha,
                row.simple_sens_beta,
                row.simple_spec_alpha,
                row.simple_spec_beta,
                alpha=0.01,
                mode="negative",
            ),
            axis=1,
            meta=("HDI_99_neg", "object"),
        ).compute()


        test_df = data["test_df"] = test_df.compute()

        for key in (
                "HDI_90", "HDI_95", "HDI_99", "HDI_90_pos", "HDI_95_pos", "HDI_99_pos",
                "HDI_90_neg", "HDI_95_neg", "HDI_99_neg"
        ):
            test_df[key] = test_df[key].apply(ast.literal_eval)

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

        coverage_90 = test_df.HDI_90_covers.sum() / len_test_df
        coverage_95 = test_df.HDI_95_covers.sum() / len_test_df
        coverage_99 = test_df.HDI_99_covers.sum() / len_test_df

        data["coverage_90"] = coverage_90
        data["coverage_95"] = coverage_95
        data["coverage_99"] = coverage_99


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

        coverage_90_pos = test_df.HDI_90_pos_covers.sum() / len_test_df
        coverage_95_pos = test_df.HDI_95_pos_covers.sum() / len_test_df
        coverage_99_pos = test_df.HDI_99_pos_covers.sum() / len_test_df

        data["coverage_90_pos"] = coverage_90_pos
        data["coverage_95_pos"] = coverage_95_pos
        data["coverage_99_pos"] = coverage_99_pos


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

        coverage_90_neg = test_df.HDI_90_neg_covers.sum() / len_test_df
        coverage_95_neg = test_df.HDI_95_neg_covers.sum() / len_test_df
        coverage_99_neg = test_df.HDI_99_neg_covers.sum() / len_test_df


        data["coverage_90_neg"] = coverage_90_neg
        data["coverage_95_neg"] = coverage_95_neg
        data["coverage_99_neg"] = coverage_99_neg

        data["test_df"] = test_df.to_json()

    with open(args.outpath, "w") as f:
        json.dump(results, f, indent=True)
