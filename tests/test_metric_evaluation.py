from itertools import permutations
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from sklearn.metrics import accuracy_score, normalized_mutual_info_score
from yacs.config import CfgNode

from pydgc.metrics import DGCMetric


def make_metric(reference, predicted):
    truth = np.array(reference)
    labels = np.array(predicted)
    embeddings = torch.arange(2 * truth.size, dtype=torch.float32).reshape(-1, 2)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
    return DGCMetric(truth, labels, embeddings, edge_index)


def expected_metrics(truth, labels):
    # The fixtures use contiguous zero-based IDs. Enumerate label assignments
    # independently of the production Hungarian solver.
    cluster_count = max(np.unique(truth).size, np.unique(labels).size)
    acc = max(
        accuracy_score(truth, np.array(assignment)[labels])
        for assignment in permutations(range(cluster_count))
    )
    nmi = normalized_mutual_info_score(truth, labels)
    return {"ACC": round(acc, 4), "NMI": round(nmi, 4)}


def evaluation_config(*, acc, nmi, nested=False):
    settings = CfgNode(
        {
            "acc": acc,
            "nmi": nmi,
            "ari": False,
            "f1": False,
            "hom": False,
            "com": False,
            "pur": False,
            "sc": False,
            "gre": False,
        }
    )
    return CfgNode({"evaluate": settings}) if nested else settings


@pytest.mark.parametrize(
    ("reference", "predicted"),
    [
        pytest.param([0, 0, 1, 1], [0, 1, 0, 1], id="independent"),
        pytest.param([0, 0, 1, 1], [0, 0, 0, 0], id="single-cluster"),
        pytest.param([0, 0, 0, 1, 1], [0, 0, 1, 1, 1], id="partial-agreement"),
        pytest.param([0, 0, 0, 0, 1, 1, 2], [0, 0, 1, 0, 1, 1, 2], id="unequal-sizes"),
        pytest.param([0, 0, 1, 1], [0, 0, 1, 1], id="perfect"),
        pytest.param([0, 0, 1, 1], [1, 1, 0, 0], id="permuted"),
        pytest.param([0, 0, 1, 1, 2, 2], [2, 2, 1, 1, 0, 0], id="three-clusters"),
    ],
)
def test_default_evaluation_reports_acc_and_nmi(reference, predicted):
    metric = make_metric(reference, predicted)
    expected = expected_metrics(metric.ground_truth, metric.predicted_labels)
    logger = Mock()

    results = metric.evaluate_one_epoch(logger, cfg=None)

    logger.info.assert_called_once_with(results)
    assert list(results) == ["ACC", "NMI"]
    assert results == expected

    configured = evaluation_config(acc=True, nmi=True)
    assert metric.evaluate_one_epoch(Mock(), cfg=configured) == results


@pytest.mark.parametrize("nested", [False, True], ids=["direct", "nested"])
@pytest.mark.parametrize(
    ("acc", "nmi", "expected"),
    [
        (True, True, {"ACC": 0.5, "NMI": 0.0}),
        (True, False, {"ACC": 0.5}),
        (False, True, {"NMI": 0.0}),
    ],
)
def test_configured_evaluation_respects_metric_flags(nested, acc, nmi, expected):
    metric = make_metric([0, 0, 1, 1], [0, 1, 0, 1])
    cfg = evaluation_config(acc=acc, nmi=nmi, nested=nested)
    logger = Mock()
    results = metric.evaluate_one_epoch(logger, cfg=cfg)

    assert list(results) == list(expected)
    assert results == expected
    logger.info.assert_called_once_with(results)


def test_repeated_default_evaluation_preserves_inputs():
    metric = make_metric([0, 0, 0, 1, 1], [0, 0, 1, 1, 1])
    truth, labels = metric.ground_truth.copy(), metric.predicted_labels.copy()
    embeddings, edges = metric.embeddings.clone(), metric.edge_index.clone()

    first = metric.evaluate_one_epoch(Mock())
    second = metric.evaluate_one_epoch(Mock())

    assert second == first
    np.testing.assert_array_equal(metric.ground_truth, truth)
    np.testing.assert_array_equal(metric.predicted_labels, labels)
    torch.testing.assert_close(metric.embeddings, embeddings)
    torch.testing.assert_close(metric.edge_index, edges)
