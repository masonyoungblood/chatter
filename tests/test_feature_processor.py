import pandas as pd
from chatter.features import FeatureProcessor


def test_feature_processor_birch_smoke(tiny_config):
    # Minimal df with required columns for clustering
    df = pd.DataFrame(
        {
            "pacmap_x": [0.0, 1.0, -0.5],
            "pacmap_y": [0.5, -1.0, 0.0],
            "h5_index": [0, 1, 2],
        }
    )
    fp = FeatureProcessor(df, tiny_config)
    fp.run_birch_clustering([2])
    assert "birch_2" in fp.df.columns
