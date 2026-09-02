"""Concrete hallucination mitigation algorithms.

Importing this package (or an individual module) registers each algorithm with
the global registry, after which :func:`mitigv.build_mitigator` can instantiate
it by name.
"""

from mitigv.algorithms.agla import AGLA, AGLAConfig
from mitigv.algorithms.icd import ICD, ICDConfig
from mitigv.algorithms.m3id import M3ID, M3IDConfig
from mitigv.algorithms.only import ONLY, ONLYConfig
from mitigv.algorithms.pai import PAI, PAIConfig
from mitigv.algorithms.probe_steer import LinearProbeSteer, LinearProbeSteerConfig
from mitigv.algorithms.vcd import VCD, VCDConfig
from mitigv.algorithms.vista import VISTA, VISTAConfig

__all__ = [
    "VCD", "VCDConfig",
    "ICD", "ICDConfig",
    "PAI", "PAIConfig",
    "M3ID", "M3IDConfig",
    "VISTA", "VISTAConfig",
    "LinearProbeSteer", "LinearProbeSteerConfig",
    "AGLA", "AGLAConfig",
    "ONLY", "ONLYConfig",
]
