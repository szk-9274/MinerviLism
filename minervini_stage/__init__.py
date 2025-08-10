"""Minervini stage classification and VCP detection package.
"""

from .indicators import compute_indicators
from .stage_rules import classify_stage, StageConfig
from .vcp import detect_vcp, VCPConfig
