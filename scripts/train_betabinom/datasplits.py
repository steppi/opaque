import networkx as nx
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

import opaque.locations as loc


if __name__ == "__main__":
    df = pd.read_csv(loc.ADEFT_BETABINOM_DATASET_PATH, sep=',')

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
    components = nx.connected_components(G)
    components = [frozenset(component) for component in components]
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

    splits = GroupShuffleSplit(
        test_size=0.2, n_splits=2, random_state=1729
    ).split(df, groups=df.group)
    train_idx, test_idx = next(splits)
    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]

    train_df.to_csv('adeft_betabinom_dataset_train.csv', sep=',', index=False)
    test_df.to_csv('adeft_betabinom_dataset_test.csv', sep=',', index=False)
