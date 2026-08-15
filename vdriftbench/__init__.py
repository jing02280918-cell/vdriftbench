"""V-DriftBench: adaptive multi-turn value-drift elicitation method (APS-v2).

This package implements only the *method* described in the design docs:
principle library, geometry-language state cascade, cross-sample Thompson
Sampling scheduler, round execution loop, and judge scoring. It contains no
experiment/ablation harness (no Group A/B/C comparison, no baseline sweep).
"""

from .schema import JudgeScores, RoundRecord, Sample, SampleResult

__all__ = ["Sample", "RoundRecord", "JudgeScores", "SampleResult"]
