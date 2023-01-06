"""Helps split data in a way that avoids information leak

This adds columns to the csv data which can be used for using grouped
stratified cross validation. The main ideas are expressed in this
docstring. See the code below for specifics.

Groups
------
In each split, we should never have a pair of datapoints A, B from train and
test respectively for which

1) A and B have the same grounding

or

2) A and B have the same agent text

If A and B have the same grounding, then their respective anomaly detectors
will use the same exact training data. If A and B have the same agent text,
then there will be a large overlap in their validation data, since the
validation data will have come from the same adeft model.

Stratification
--------------
We also try to stratify crudely by the number of inlier and outlier samples in
the data.
"""

import argparse
import networkx as nx
import numpy as np
import pandas as pd

import opaque.locations as loc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--numpy_seed", type=int)
    args = parser.parse_args()

    df = pd.read_csv(loc.ADEFT_BETABINOM_DATASET_PATH, sep=',')

    # We're creating a graph structure on the datapoints in such a way that
    # two points are linked if they have either the same grounding or the
    # same agent text.
    G = nx.Graph()
    G.add_nodes_from(df.index)
    for i in range(len(df)):
        for j in range(i+1, len(df)):
            row1, row2 = df.iloc[i], df.iloc[j]
            if (
                    row1.shortform == row2.shortform or
                    row1.grounding == row2.grounding
            ):
                G.add_edge(i, j)
    # We find the connected components in this graph. Any two points in the
    # same connected component are considered to be in the same group. We can
    # then use group split methods to ensure that no train/test split has points
    # from the same group in both train and test.
    components = nx.connected_components(G)
    components = [frozenset(component) for component in components]

    # There's some bookkeeping involved.
    component_to_group = {
        component: i for i, component in enumerate(components)
    }
    index_to_group = {
        index: component_to_group[component]
        for component in components for index in component
    }

    df['group'] = df.index.map(
        lambda x: index_to_group[x]
    )

    def get_size_group(x):
        return max(2, int(np.log10(x + 1)))

    df['spec_strat_label'] = df.N_inlier.apply(get_size_group)
    df['sens_strat_label'] = df.N_outlier.apply(get_size_group)
    df['joint_strat_label'] = 3*df.spec_strat_label + df.sens_strat_label
    
    df.to_csv("adeft_betabinom_dataset_processed.csv", sep=",", index=False)
