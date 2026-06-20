# SPDX-License-Identifier: Apache-2.0
"""
models.samat_next
=================
Public API for SamatNext-v0.1.

Usage:
    from models.samat_next import SamatNextConfig, SamatNextForCausalLM

Architecture:
    - SamatNextConfig          — dataclass of all hyperparameters
    - SamatNextModel           — backbone (embeddings + blocks + final norm)
    - SamatNextForCausalLM     — backbone + LM head (+ optional verifier head)
    - SamatNextBlock           — single decoder block (DifferentialAttention OR LinearStateMixer + SwiGLU MLP)
    - DifferentialAttention    — dual-stream attention with learnable alpha mixing
    - DeltaNetInspiredLinearStateMixer — causal linear-state mixer (DeltaNet-inspired)
    - VerifierHead             — optional scalar verifier output on last token
    - RMSNorm                  — Root Mean Square Layer Normalization
"""

from .config import SamatNextConfig
from .model import SamatNextModel, SamatNextForCausalLM
from .layers import SamatNextBlock, RMSNorm, MLP
from .differential_attention import DifferentialAttention
from .linear_state_mixer import DeltaNetInspiredLinearStateMixer
from .verifier import VerifierHead

__all__ = [
    "SamatNextConfig",
    "SamatNextModel",
    "SamatNextForCausalLM",
    "SamatNextBlock",
    "RMSNorm",
    "MLP",
    "DifferentialAttention",
    "DeltaNetInspiredLinearStateMixer",
    "VerifierHead",
]
