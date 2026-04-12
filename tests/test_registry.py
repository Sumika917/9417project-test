from project9417.registry import DATASET_REGISTRY


def test_uci_source_urls_use_beta_archive():
    assert DATASET_REGISTRY["iris"].source_url == "https://archive-beta.ics.uci.edu/dataset/53/iris"
    assert (
        DATASET_REGISTRY["appendicitis"].source_url
        == "https://archive-beta.ics.uci.edu/dataset/938/regensburg+pediatric+appendicitis"
    )
    assert (
        DATASET_REGISTRY["parkinsons"].source_url
        == "https://archive-beta.ics.uci.edu/dataset/189/parkinsons+telemonitoring"
    )
