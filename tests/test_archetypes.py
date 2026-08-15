from vdriftbench.archetypes import ResistArchetypeRegistry
from vdriftbench.embedding_client import MockEmbeddingClient


def test_first_observation_creates_a_new_cluster():
    registry = ResistArchetypeRegistry(tau_archetype=0.75, n_archetype_min=8)
    embed = MockEmbeddingClient()
    cluster = registry.observe("历史类", "免责声明式回避", embed)
    assert cluster.n_obs == 1
    assert not cluster.is_promoted()


def test_repeated_identical_label_grows_the_same_cluster():
    registry = ResistArchetypeRegistry(tau_archetype=0.75, n_archetype_min=8)
    embed = MockEmbeddingClient()
    first = registry.observe("历史类", "免责声明式回避", embed)
    second = registry.observe("历史类", "免责声明式回避", embed)
    assert first.cluster_id == second.cluster_id
    assert second.n_obs == 2


def test_dissimilar_labels_create_separate_clusters():
    registry = ResistArchetypeRegistry(tau_archetype=0.9, n_archetype_min=8)
    embed = MockEmbeddingClient()
    c1 = registry.observe("历史类", "免责声明式回避", embed)
    c2 = registry.observe("历史类", "反问确认式", embed)
    # MockEmbeddingClient produces near-orthogonal vectors for different
    # texts, so with a strict tau these should not merge.
    assert c1.cluster_id != c2.cluster_id


def test_clusters_are_independent_per_category_macro():
    registry = ResistArchetypeRegistry(tau_archetype=0.75, n_archetype_min=8)
    embed = MockEmbeddingClient()
    registry.observe("历史类", "免责声明式回避", embed)
    registry.observe("经济类", "免责声明式回避", embed)
    assert len(registry.clusters_for("历史类")) == 1
    assert len(registry.clusters_for("经济类")) == 1


def test_cluster_promotes_after_reaching_n_archetype_min():
    registry = ResistArchetypeRegistry(tau_archetype=0.75, n_archetype_min=3)
    embed = MockEmbeddingClient()
    cluster = None
    for _ in range(3):
        cluster = registry.observe("历史类", "免责声明式回避", embed)
    assert cluster.is_promoted()
    assert cluster.promoted_name == "历史类::免责声明式回避"


def test_cluster_not_promoted_below_threshold():
    registry = ResistArchetypeRegistry(tau_archetype=0.75, n_archetype_min=5)
    embed = MockEmbeddingClient()
    cluster = registry.observe("历史类", "免责声明式回避", embed)
    assert not cluster.is_promoted()


def test_snapshot_reports_all_categories_and_clusters():
    registry = ResistArchetypeRegistry(tau_archetype=0.75, n_archetype_min=8)
    embed = MockEmbeddingClient()
    registry.observe("历史类", "免责声明式回避", embed)
    snap = registry.snapshot()
    assert "历史类" in snap
    assert len(snap["历史类"]) == 1
    assert snap["历史类"][0]["n_obs"] == 1
