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


def test_uci_direct_download_urls_are_configured():
    assert DATASET_REGISTRY["iris"].source_download_url == "https://cdn.uci-ics-mlr-prod.aws.uci.edu/53/iris.zip"
    assert (
        DATASET_REGISTRY["appendicitis"].source_download_url
        == "https://zenodo.org/records/7669442/files/app_data.xlsx?download=1"
    )
    assert (
        DATASET_REGISTRY["parkinsons"].source_download_url
        == "https://cdn.uci-ics-mlr-prod.aws.uci.edu/189/parkinsons%2Btelemonitoring.zip"
    )
