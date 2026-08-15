"""v3 6.5节：攻防联合自扩展 —— 抵抗原型分类体系
（Resistance-Archetype Taxonomy）。

Layer2 每次兜底判定为 `S_soft_resist` 时都会附带输出一句自由文本，描述这次
软抵抗具体是哪种表现形式（见 `state_cascade.layer2_llm_classify`）。这个模块
把这些自由文本标签用已有的 embedding client 做增量聚类：每个大类
(category_macro) 下独立维护一组簇；新标签与已有簇心比较余弦相似度，超过
`tau_archetype` 就归入已有簇（簇心增量更新为均值），否则新开一个簇。一个簇
的观测次数达到 `n_archetype_min` 后转正为正式命名的抵抗原型，`bandit.py` 的
上下文可以据此从粗粒度 `(大类, S_soft_resist)` 细化为 `(大类, 抵抗原型)`。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .embedding_client import EmbeddingClient, cosine_similarity


@dataclass
class ArchetypeCluster:
    cluster_id: str
    centroid: np.ndarray
    n_obs: int = 0
    sample_labels: List[str] = field(default_factory=list)
    promoted_name: Optional[str] = None  # 转正后的正式命名；未转正为 None

    def is_promoted(self) -> bool:
        return self.promoted_name is not None


@dataclass
class ResistArchetypeRegistry:
    """按大类(category_macro)独立维护的抵抗原型增量聚类registry。"""

    tau_archetype: float = 0.75
    n_archetype_min: int = 8
    _clusters: Dict[str, List[ArchetypeCluster]] = field(default_factory=dict)
    _next_id: int = 0

    def observe(self, category_macro: str, archetype_text: str, embed_client: EmbeddingClient) -> ArchetypeCluster:
        """摄入一条在 `category_macro` 下观察到的软抵抗自由文本描述。归入现有
        簇（若与某个簇心的余弦相似度 >= tau_archetype，取相似度最高的那个）或
        新建一个簇；当簇的观测次数达到 `n_archetype_min` 时就地转正。"""

        vec = np.asarray(embed_client.embed(archetype_text), dtype=np.float32)
        buckets = self._clusters.setdefault(category_macro, [])

        best_cluster, best_sim = None, -1.0
        for cluster in buckets:
            sim = cosine_similarity(vec, cluster.centroid)
            if sim > best_sim:
                best_cluster, best_sim = cluster, sim

        if best_cluster is not None and best_sim >= self.tau_archetype:
            n = best_cluster.n_obs
            best_cluster.centroid = (best_cluster.centroid * n + vec) / (n + 1)
            best_cluster.n_obs += 1
            best_cluster.sample_labels.append(archetype_text)
            cluster = best_cluster
        else:
            self._next_id += 1
            cluster = ArchetypeCluster(
                cluster_id=f"archetype#{self._next_id}", centroid=vec, n_obs=1, sample_labels=[archetype_text]
            )
            buckets.append(cluster)

        if not cluster.is_promoted() and cluster.n_obs >= self.n_archetype_min:
            cluster.promoted_name = self._name_cluster(category_macro, cluster)

        return cluster

    @staticmethod
    def _name_cluster(category_macro: str, cluster: ArchetypeCluster) -> str:
        # Use the most frequently observed free-text label in the cluster as
        # its formal name — simple, auditable, no extra LLM call needed.
        most_common = Counter(cluster.sample_labels).most_common(1)[0][0]
        return f"{category_macro}::{most_common}"

    def clusters_for(self, category_macro: str) -> List[ArchetypeCluster]:
        return list(self._clusters.get(category_macro, []))

    def snapshot(self) -> dict:
        return {
            cat: [
                {
                    "cluster_id": c.cluster_id,
                    "n_obs": c.n_obs,
                    "promoted_name": c.promoted_name,
                    "labels": c.sample_labels,
                }
                for c in clusters
            ]
            for cat, clusters in self._clusters.items()
        }
