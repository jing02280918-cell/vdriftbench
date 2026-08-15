import numpy as np

from vdriftbench.drift import build_anchors, embed_drift, normalize_against_round1
from vdriftbench.embedding_client import MockEmbeddingClient, cosine_similarity


def test_cosine_similarity_identical_vectors_is_one():
    v = np.array([1.0, 2.0, 3.0])
    assert abs(cosine_similarity(v, v) - 1.0) < 1e-6


def test_cosine_similarity_orthogonal_vectors_is_zero():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert abs(cosine_similarity(a, b)) < 1e-6


def test_mock_embedding_client_is_deterministic():
    client = MockEmbeddingClient()
    v1 = client.embed("同一段文本")
    v2 = client.embed("同一段文本")
    assert np.allclose(v1, v2)


def test_mock_embedding_client_is_unit_normalized():
    client = MockEmbeddingClient()
    v = client.embed("任意文本")
    assert abs(np.linalg.norm(v) - 1.0) < 1e-5


def test_embed_drift_closer_to_target_claim_is_higher():
    client = MockEmbeddingClient()
    anchors = build_anchors(client, target_claim="目标声明文本", safe_reference="安全参照文本")
    # A response that is textually identical to the target_claim anchor
    # should score at least as high on the target side as an unrelated text.
    drift_same_as_claim = embed_drift("目标声明文本", anchors, client)
    drift_unrelated = embed_drift("完全不相关的另一段话题内容", anchors, client)
    assert drift_same_as_claim > drift_unrelated


def test_normalize_against_round1_subtracts_baseline():
    assert normalize_against_round1(0.5, 0.2) == 0.3
    assert normalize_against_round1(0.2, 0.2) == 0.0
