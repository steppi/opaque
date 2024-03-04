import argparse
import pandas as pd

from opaque.results import OpaqueResultsManager


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_table_name")
    parser.add_argument("outpath")
    args = parser.parse_args()

    rows = OpaqueResultsManager.iterrows(args.results_table_name)
    results_rows = []
    for key, json_data in rows:
        results_rows.append(
            {
                "prior_type": json_data["hps"]["prior_type"],
                "coeff_scale": json_data["hps"]["coeff_scale"],
                "nkld_score": json_data["nkld_score"],
                "baseline_score": json_data["baseline_score"],
                "target_type": json_data["target_type"],
                "outer_split": json_data["outer_split"],
                "inner_split": json_data["inner_split"],
            }
        )
    df = pd.DataFrame(results_rows)
    grouped = df.groupby(
        ["target_type", "outer_split", "prior_type", "coeff_scale"],
        as_index=False)[["nkld_score", "baseline_score"]].mean()
    grouped = grouped.sort_values(["target_type", "outer_split", "nkld_score"])
    best_hps_df = grouped.groupby(
        ["target_type", "outer_split"], as_index=False
    ).first()
    best_hps_df.to_csv(args.outpath, sep=",", index=False)
