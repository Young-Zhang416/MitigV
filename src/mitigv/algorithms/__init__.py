"""Concrete hallucination mitigation algorithms.

Importing this package (or an individual module) registers each algorithm with
the global registry, after which :func:`mitigv.build_mitigator` can instantiate
it by name.
"""

from mitigv.algorithms.icd import ICD, ICDConfig
from mitigv.algorithms.vcd import VCD, VCDConfig

__all__ = ["VCD", "VCDConfig", "ICD", "ICDConfig"]
