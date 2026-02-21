"""
pipeline_orchestrator.py
========================
GeneForge Pipeline — Full Orchestration (DNA → Corrected DNA)

Overview
--------
This module is the top-level coordinator for the GeneForge in-silico protein
repair pipeline.  It does **no scientific computation itself** — it delegates
every stage to the specialist module responsible for that work, wires their
outputs together, manages per-stage metadata and error handling, and
returns a unified :class:`PipelineResult`.

Pipeline Stages
---------------
::

    ┌─────────────────────────────────────────────────────────────────┐
    │  Input: raw DNA string  +  reference PDB path                   │
    └───────────────────────────────┬─────────────────────────────────┘
                                    │
    Stage 1 ── DNA Preprocessing ───┘  dna_preprocessing.preprocess_sequence
                                    │     → DNARecord
    Stage 2 ── Protein Translation ─┘  protein_translation.translate_dna
                                    │     → ProteinRecord
    Stage 3 ── ESM-3 Instability ───┘  esm3_instability_detector.detect_instability
                                    │     → InstabilityReport
    Stage 4 ── RFdiffusion Healing ─┘  rfdiffusion_healer.heal_protein
                                    │     → HealingReport
    Stage 5 ── ProteinMPNN Inverse ─┘  proteinmpnn_inverse_fold.run_inverse_folding
                   Folding          │     → InverseFoldResult
    Stage 6 ── Codon Optimisation ──┘  codon_optimization.optimise_codons
                                    │     → CodonOptimizationResult
    Stage 7 ── Result Assembly ─────┘     → PipelineResult
                                    │
    ┌───────────────────────────────┴─────────────────────────────────┐
    │  Output: PipelineResult  (all stage artefacts + metadata)       │
    └─────────────────────────────────────────────────────────────────┘

Stage-Skip Semantics
--------------------
Stages 4 and 5 can be bypassed when no instability spikes are found or when
the input PDB path is not provided.  All skips are recorded in
:attr:`PipelineResult.stage_metadata` and :attr:`PipelineResult.skipped_stages`
so callers can distinguish "stage ran and produced no output" from "stage was
not run."

Error Strategy
--------------
Each stage is executed inside a ``try/except`` block.  Errors are classified
as *fatal* (pipeline stops immediately and returns a partial
:class:`PipelineResult` with :attr:`PipelineResult.success` = ``False``) or
*recoverable* (the stage is skipped with a warning, and subsequent stages that
can proceed do so).

+-----+----------------------------+------------------------------------------+
| #   | Stage                      | Error treatment                          |
+=====+============================+==========================================+
| 1   | DNA Preprocessing          | Fatal — invalid DNA cannot continue      |
+-----+----------------------------+------------------------------------------+
| 2   | Protein Translation        | Fatal — no protein, no structure         |
+-----+----------------------------+------------------------------------------+
| 3   | ESM-3 Instability          | Fatal — no instability map → Stage 4     |
|     |                            | has no targets                           |
+-----+----------------------------+------------------------------------------+
| 4   | RFdiffusion Healing        | Recoverable — if PDB absent or           |
|     |                            | NoRegionsToRepairError, skip to Stage 5  |
|     |                            | using original PDB                       |
+-----+----------------------------+------------------------------------------+
| 5   | ProteinMPNN Inverse Fold   | Recoverable — skip codon optimisation,   |
|     |                            | use reference sequence                   |
+-----+----------------------------+------------------------------------------+
| 6   | Codon Optimisation         | Recoverable — return un-optimised CDS    |
+-----+----------------------------+------------------------------------------+

Codon Optimisation Adapter
---------------------------
The module imports ``codon_optimization`` at call-time (lazy import) to stay
loosely coupled.  The expected public API is::

    codon_optimization.optimise_codons(
        protein_sequence: str,
        sequence_id: str,
        organism: str = "h_sapiens",
        seed: int | None = None,
    ) -> CodonOptimizationResult

:class:`CodonOptimizationResult` is defined in this module so the orchestrator
can function — and return a valid result — even before the codon optimisation
module is available (the adapter gracefully falls back to a naïve codon table
if the module is missing).

Reproducibility
---------------
A single ``seed`` integer is threaded through every stage that accepts one.
Per-stage seeds are derived deterministically from the root seed to prevent
cross-stage correlation::

    stage_seed(n) = seed + n * 1000    (n = 1-based stage number)

This scheme is stable across Python versions because it uses only arithmetic,
not hash randomness.

Async Readiness
---------------
``run_pipeline`` is a plain synchronous function.  It is deliberately
structured so that wrapping it with ``asyncio.to_thread`` or
``fastapi.BackgroundTasks`` requires zero changes to this file::

    # FastAPI usage (future):
    result = await asyncio.to_thread(run_pipeline, config)

Output Files
------------
Every stage writes its artefacts to a subdirectory of ``output_dir``::

    output_dir/
    ├── dna/
    │   ├── {seq_id}_preprocessed.fasta
    │   └── {seq_id}_mutations.json
    ├── protein/
    │   ├── {seq_id}.fasta
    │   └── {seq_id}_translation.json
    ├── instability/
    │   └── {seq_id}_instability.json
    ├── healing/
    │   ├── {seq_id}_healed.pdb
    │   └── {seq_id}_healing.json
    ├── inverse_fold/
    │   ├── {seq_id}_designed.fasta
    │   └── {seq_id}_designed.json
    ├── codons/
    │   ├── {seq_id}_optimised.fasta
    │   └── {seq_id}_codons.json
    └── {seq_id}_pipeline_result.json

Environment Variables
---------------------
Stage modules read their own checkpoint environment variables:

  RFDIFFUSION_WEIGHTS_PATH     — Stage 4
  PROTEINMPNN_WEIGHTS_PATH     — Stage 5
  ESM3_MODEL_PATH              — Stage 3 (if applicable)

Dependencies
------------
  Python   >= 3.12
  Standard library only at import time.
  Stage modules loaded lazily at call time.
"""

from __future__ import annotations

import json
import logging
import os
import time
import traceback
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Final

# ---------------------------------------------------------------------------
# Module logger
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pipeline version
# ---------------------------------------------------------------------------

PIPELINE_VERSION: Final[str] = "1.0.0"

# ---------------------------------------------------------------------------
# Stage identifiers (used as dict keys in metadata)
# ---------------------------------------------------------------------------


class Stage(str, Enum):
    """Canonical names for pipeline stages."""

    DNA_PREPROCESSING   = "dna_preprocessing"
    PROTEIN_TRANSLATION = "protein_translation"
    ESM3_INSTABILITY    = "esm3_instability"
    RFDIFFUSION_HEALING = "rfdiffusion_healing"
    INVERSE_FOLDING     = "inverse_folding"
    CODON_OPTIMIZATION  = "codon_optimization"


# ---------------------------------------------------------------------------
# Stage seed derivation
# ---------------------------------------------------------------------------

_STAGE_ORDER: Final[dict[Stage, int]] = {
    Stage.DNA_PREPROCESSING:   1,
    Stage.PROTEIN_TRANSLATION: 2,
    Stage.ESM3_INSTABILITY:    3,
    Stage.RFDIFFUSION_HEALING: 4,
    Stage.INVERSE_FOLDING:     5,
    Stage.CODON_OPTIMIZATION:  6,
}


def _stage_seed(root_seed: int | None, stage: Stage) -> int | None:
    """
    Derive a deterministic per-stage seed from the root seed.

    Parameters
    ----------
    root_seed : int | None
        ``None`` → non-deterministic; each stage also receives ``None``.
    stage : Stage

    Returns
    -------
    int | None
    """
    if root_seed is None:
        return None
    return root_seed + _STAGE_ORDER[stage] * 1000


# ---------------------------------------------------------------------------
# Codon optimisation data contract
# (defined here so the orchestrator works before the real module exists)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class CodonOptimizationResult:
    """
    Result of codon optimisation for a designed amino acid sequence.

    When the ``codon_optimization`` module is unavailable, the orchestrator
    falls back to a naïve per-residue most-frequent-codon lookup and sets
    :attr:`method` to ``"naive_fallback"``.

    Attributes
    ----------
    sequence_id : str
    protein_sequence : str
        The input amino acid sequence that was back-translated.
    optimised_dna : str
        The codon-optimised DNA coding sequence (5'→3', no stop codon
        unless the module appends one).
    cai : float
        Codon Adaptation Index [0, 1].  1.0 = every codon is the most
        frequent for the target organism.  -1.0 = not computed (fallback).
    organism : str
        Target organism for codon usage table (e.g. ``"h_sapiens"``).
    method : str
        Optimisation method used (e.g. ``"dna_chisel"``, ``"naive_fallback"``).
    warnings : list[str]
    """

    sequence_id: str
    protein_sequence: str
    optimised_dna: str
    cai: float
    organism: str
    method: str
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "sequence_id": self.sequence_id,
            "protein_sequence": self.protein_sequence,
            "optimised_dna": self.optimised_dna,
            "cai": round(self.cai, 4) if self.cai >= 0 else -1.0,
            "organism": self.organism,
            "method": self.method,
            "warnings": self.warnings,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# ---------------------------------------------------------------------------
# Stage metadata
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class StageMetadata:
    """
    Per-stage execution record stored in :attr:`PipelineResult.stage_metadata`.

    Attributes
    ----------
    stage : Stage
    status : str
        One of ``"success"``, ``"skipped"``, ``"failed"``.
    elapsed_s : float
        Wall-clock seconds for this stage.  0.0 if skipped.
    seed_used : int | None
        Seed passed to this stage (derived from the root seed).
    output_paths : list[str]
        Paths of files written by this stage.
    warnings : list[str]
        Warnings collected from the stage module.
    error : str | None
        Exception message if ``status == "failed"``, else ``None``.
    error_type : str | None
        Exception class name if ``status == "failed"``, else ``None``.
    """

    stage: Stage
    status: str
    elapsed_s: float
    seed_used: int | None = None
    output_paths: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    error_type: str | None = None

    def to_dict(self) -> dict:
        return {
            "stage": self.stage.value,
            "status": self.status,
            "elapsed_s": round(self.elapsed_s, 3),
            "seed_used": self.seed_used,
            "output_paths": self.output_paths,
            "warnings": self.warnings,
            "error": self.error,
            "error_type": self.error_type,
        }


# ---------------------------------------------------------------------------
# Pipeline configuration
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    """
    Complete configuration for a single GeneForge pipeline run.

    All paths and model parameters can be overridden here; none are
    hardcoded in the orchestrator.  Defaults are chosen to be sensible
    for a first run on a GPU-equipped workstation.

    Parameters
    ----------
    raw_dna : str
        The DNA sequence to repair (ACGT + ambiguous codes; case-insensitive).
    sequence_id : str
        Human-readable identifier.  Used as a prefix for all output files.
        Defaults to a UUID4 if not provided.
    output_dir : str | Path
        Root directory for all stage outputs.  Created if absent.
    reference_dna : str | None
        Optional reference DNA for mutation tracking in Stage 1.
    reference_pdb_path : str | Path | None
        Path to the wild-type or AlphaFold-predicted PDB for Stage 4
        (RFdiffusion input).  If ``None``, Stage 4 is skipped.
    device : str | None
        PyTorch device string (``"cuda"``, ``"cpu"``, ``"cuda:1"``, …).
        ``None`` = auto-detect for every stage that supports it.
    seed : int | None
        Root random seed.  ``None`` = non-deterministic.
    esm3_model_id : str
        ESM-3 model variant for Stage 3.
    esm3_z_threshold : float
        Z-score threshold for perplexity spike detection.
    esm3_ppl_abs_threshold : float
        Absolute perplexity threshold for spike detection.
    rfdiffusion_checkpoint : str | None
        Path to RFdiffusion checkpoint.  Falls back to
        ``$RFDIFFUSION_WEIGHTS_PATH/RFdiffusion.pt``.
    rfdiffusion_noise_scale : float
        Noise scale for partial diffusion (default 1.0).
    rfdiffusion_num_designs : int
        Number of backbone designs to generate per repair run.
    rfdiffusion_flank_residues : int
        Residues of structural context added around each repair region.
    rfdiffusion_merge_gap : int
        Gap (residues) below which adjacent repair regions are merged.
    proteinmpnn_checkpoint : str | None
        Path to ProteinMPNN checkpoint.  Falls back to
        ``$PROTEINMPNN_WEIGHTS_PATH/v_48_020.pt``.
    proteinmpnn_temperatures : list[float] | None
        Sampling temperatures for inverse folding.  ``None`` → ``[0.1]``.
    proteinmpnn_num_seqs : int
        Independent sequences to sample per temperature.
    proteinmpnn_chain_id : str
        PDB chain to design sequences for.
    codon_organism : str
        Target organism for codon usage table.
    translation_frame : int
        Reading frame (1, 2, or 3) for Stage 2.  ``0`` = auto-detect best ORF.
    overwrite : bool
        Whether to overwrite existing stage output files.
    stop_on_no_instability : bool
        If ``True`` and ESM-3 finds no spikes, stop after Stage 3.
        If ``False``, proceed to report clean protein and skip Stages 4–5.
    """

    raw_dna: str
    sequence_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    output_dir: str | Path = Path("geneforge_output")
    reference_dna: str | None = None
    reference_pdb_path: str | Path | None = None
    device: str | None = None
    seed: int | None = 42

    # Stage 3 — ESM-3
    esm3_model_id: str = "esm3_sm_open_v1"
    esm3_z_threshold: float = 2.0
    esm3_ppl_abs_threshold: float = 10.0

    # Stage 4 — RFdiffusion
    rfdiffusion_checkpoint: str | None = None
    rfdiffusion_noise_scale: float = 1.0
    rfdiffusion_num_designs: int = 1
    rfdiffusion_flank_residues: int = 5
    rfdiffusion_merge_gap: int = 3

    # Stage 5 — ProteinMPNN
    proteinmpnn_checkpoint: str | None = None
    proteinmpnn_temperatures: list[float] | None = None
    proteinmpnn_num_seqs: int = 8
    proteinmpnn_chain_id: str = "A"

    # Stage 6 — Codon optimisation
    codon_organism: str = "h_sapiens"

    # Translation
    translation_frame: int = 0  # 0 = auto

    # Behaviour flags
    overwrite: bool = False
    stop_on_no_instability: bool = False

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        if self.reference_pdb_path is not None:
            self.reference_pdb_path = Path(self.reference_pdb_path)

    def to_dict(self) -> dict:
        return {
            "sequence_id":              self.sequence_id,
            "output_dir":               str(self.output_dir),
            "device":                   self.device,
            "seed":                     self.seed,
            "esm3_model_id":            self.esm3_model_id,
            "esm3_z_threshold":         self.esm3_z_threshold,
            "esm3_ppl_abs_threshold":   self.esm3_ppl_abs_threshold,
            "rfdiffusion_checkpoint":   self.rfdiffusion_checkpoint,
            "rfdiffusion_noise_scale":  self.rfdiffusion_noise_scale,
            "rfdiffusion_num_designs":  self.rfdiffusion_num_designs,
            "rfdiffusion_flank_residues": self.rfdiffusion_flank_residues,
            "rfdiffusion_merge_gap":    self.rfdiffusion_merge_gap,
            "proteinmpnn_checkpoint":   self.proteinmpnn_checkpoint,
            "proteinmpnn_temperatures": self.proteinmpnn_temperatures,
            "proteinmpnn_num_seqs":     self.proteinmpnn_num_seqs,
            "proteinmpnn_chain_id":     self.proteinmpnn_chain_id,
            "codon_organism":           self.codon_organism,
            "translation_frame":        self.translation_frame,
            "overwrite":                self.overwrite,
            "stop_on_no_instability":   self.stop_on_no_instability,
            "reference_pdb_path":       str(self.reference_pdb_path) if self.reference_pdb_path else None,
        }


# ---------------------------------------------------------------------------
# Pipeline result
# ---------------------------------------------------------------------------


@dataclass
class PipelineResult:
    """
    Unified output object for a complete GeneForge pipeline run.

    All stage outputs are stored here.  Fields are ``None`` for stages
    that were skipped or failed.

    Attributes
    ----------
    run_id : str
        Unique identifier for this pipeline run (UUID4).
    sequence_id : str
        From :attr:`PipelineConfig.sequence_id`.
    success : bool
        ``True`` iff the pipeline completed all mandatory stages without a
        fatal error.
    final_corrected_dna : str | None
        The codon-optimised DNA sequence produced by Stage 6.  Falls back
        to the best ProteinMPNN sequence back-translated with naïve codons
        if Stage 6 is skipped.  ``None`` only if Stage 2 or earlier failed.
    dna_record : Any
        :class:`dna_preprocessing.DNARecord` from Stage 1.
    protein_record : Any
        :class:`protein_translation.ProteinRecord` from Stage 2.
    instability_report : Any
        :class:`esm3_instability_detector.InstabilityReport` from Stage 3.
    healing_report : Any
        :class:`rfdiffusion_healer.HealingReport` from Stage 4,
        or ``None`` if skipped.
    inverse_fold_result : Any
        :class:`proteinmpnn_inverse_fold.InverseFoldResult` from Stage 5,
        or ``None`` if skipped.
    codon_result : CodonOptimizationResult | None
        Codon optimisation output from Stage 6, or ``None`` if skipped.
    skipped_stages : list[Stage]
        Stages that were intentionally not run (not failed).
    failed_stages : list[Stage]
        Stages that raised exceptions.
    stage_metadata : dict[str, StageMetadata]
        Per-stage execution records keyed by :attr:`Stage.value`.
    total_elapsed_s : float
        Total wall-clock time for the pipeline.
    pipeline_version : str
    config : PipelineConfig
        The configuration used for this run.
    warnings : list[str]
        Cross-stage warnings collected by the orchestrator.
    """

    run_id: str
    sequence_id: str
    success: bool
    config: PipelineConfig
    total_elapsed_s: float

    # Stage outputs (None = skipped or failed)
    final_corrected_dna: str | None = None
    dna_record: Any = None
    protein_record: Any = None
    instability_report: Any = None
    healing_report: Any = None
    inverse_fold_result: Any = None
    codon_result: CodonOptimizationResult | None = None

    # Run metadata
    skipped_stages: list[Stage] = field(default_factory=list)
    failed_stages: list[Stage] = field(default_factory=list)
    stage_metadata: dict[str, StageMetadata] = field(default_factory=dict)
    pipeline_version: str = PIPELINE_VERSION
    warnings: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Convenience views
    # ------------------------------------------------------------------

    @property
    def num_stages_run(self) -> int:
        return sum(
            1 for m in self.stage_metadata.values()
            if m.status == "success"
        )

    @property
    def instability_detected(self) -> bool:
        if self.instability_report is None:
            return False
        return bool(getattr(self.instability_report, "spike_regions", []))

    @property
    def structural_repair_performed(self) -> bool:
        return self.healing_report is not None

    @property
    def best_designed_sequence(self) -> str | None:
        """Best ProteinMPNN sequence, or None if Stage 5 was skipped."""
        if self.inverse_fold_result is None:
            return None
        try:
            return self.inverse_fold_result.best_sequence.sequence
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """
        Return a JSON-serialisable dict of the pipeline result.

        Large stage objects (reports, records) are reduced to their
        ``.to_dict()`` representation.  ``None`` fields are preserved.
        """

        def _safe_to_dict(obj):
            if obj is None:
                return None
            if hasattr(obj, "to_dict"):
                try:
                    return obj.to_dict()
                except Exception:
                    return str(obj)
            return str(obj)

        return {
            "run_id":               self.run_id,
            "sequence_id":          self.sequence_id,
            "pipeline_version":     self.pipeline_version,
            "success":              self.success,
            "total_elapsed_s":      round(self.total_elapsed_s, 2),
            "num_stages_run":       self.num_stages_run,
            "instability_detected": self.instability_detected,
            "structural_repair_performed": self.structural_repair_performed,
            "skipped_stages":       [s.value for s in self.skipped_stages],
            "failed_stages":        [s.value for s in self.failed_stages],
            "final_corrected_dna":  self.final_corrected_dna,
            "warnings":             self.warnings,
            "stage_metadata":       {k: v.to_dict() for k, v in self.stage_metadata.items()},
            "config":               self.config.to_dict(),
            "dna_record":           _safe_to_dict(self.dna_record),
            "protein_record":       _safe_to_dict(self.protein_record),
            "instability_report":   _safe_to_dict(self.instability_report),
            "healing_report":       _safe_to_dict(self.healing_report),
            "inverse_fold_result":  _safe_to_dict(self.inverse_fold_result),
            "codon_result":         _safe_to_dict(self.codon_result),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def summary(self) -> str:
        """One-line human-readable summary of the pipeline result."""
        status  = "✓ SUCCESS" if self.success else "✗ FAILED"
        spikes  = f"spikes={len(getattr(self.instability_report, 'spike_regions', []))}"
        healed  = "healed=yes" if self.structural_repair_performed else "healed=no"
        seqs    = f"seqs={len(getattr(self.inverse_fold_result, 'sequences', []))}"
        dna_len = f"dna_len={len(self.final_corrected_dna)}" if self.final_corrected_dna else "dna=none"
        elapsed = f"{self.total_elapsed_s:.1f}s"
        return (
            f"[{self.sequence_id}] {status} | {spikes} | {healed} | "
            f"{seqs} | {dna_len} | {elapsed}"
        )

    def write(
        self,
        output_path: str | Path | None = None,
        overwrite: bool = False,
    ) -> Path:
        """
        Serialise the full result to a JSON file.

        Parameters
        ----------
        output_path : str | Path | None
            Defaults to ``{config.output_dir}/{sequence_id}_pipeline_result.json``.
        overwrite : bool

        Returns
        -------
        Path

        Raises
        ------
        FileExistsError
        """
        if output_path is None:
            output_path = (
                Path(self.config.output_dir) / f"{self.sequence_id}_pipeline_result.json"
            )
        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists() and not overwrite:
            raise FileExistsError(
                f"Pipeline result '{output_path}' already exists. "
                "Pass overwrite=True to replace it."
            )
        output_path.write_text(self.to_json(), encoding="utf-8")
        logger.info("Pipeline result written to '%s'.", output_path)
        return output_path


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class PipelineError(Exception):
    """Base exception for orchestration-level failures."""


class FatalStageError(PipelineError):
    """
    Raised when a mandatory stage fails and the pipeline cannot continue.

    Attributes
    ----------
    stage : Stage
    cause : Exception
    """

    def __init__(self, stage: Stage, cause: Exception) -> None:
        self.stage = stage
        self.cause = cause
        super().__init__(
            f"Fatal failure in stage '{stage.value}': "
            f"{type(cause).__name__}: {cause}"
        )


class InvalidConfigError(PipelineError):
    """Raised when :class:`PipelineConfig` contains invalid parameters."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_output_dir(base: Path, subdirectory: str) -> Path:
    """Create and return ``base / subdirectory``."""
    d = base / subdirectory
    d.mkdir(parents=True, exist_ok=True)
    return d


def _record_stage(
    metadata: dict[str, StageMetadata],
    stage: Stage,
    status: str,
    elapsed: float,
    seed: int | None = None,
    output_paths: list[str] | None = None,
    warnings: list[str] | None = None,
    error: str | None = None,
    error_type: str | None = None,
) -> StageMetadata:
    """Create a :class:`StageMetadata` entry and add it to *metadata*."""
    sm = StageMetadata(
        stage=stage,
        status=status,
        elapsed_s=elapsed,
        seed_used=seed,
        output_paths=output_paths or [],
        warnings=warnings or [],
        error=error,
        error_type=error_type,
    )
    metadata[stage.value] = sm
    return sm


def _collect_warnings(obj) -> list[str]:
    """Safely extract a ``warnings`` list from any stage output object."""
    try:
        w = getattr(obj, "warnings", [])
        return list(w) if w else []
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Codon optimisation adapter
# ---------------------------------------------------------------------------

#: Naïve most-frequent human codon for each amino acid.
#: Used only when the ``codon_optimization`` module is unavailable.
_FALLBACK_CODON_TABLE: Final[dict[str, str]] = {
    "A": "GCC", "C": "TGC", "D": "GAC", "E": "GAG",
    "F": "TTC", "G": "GGC", "H": "CAC", "I": "ATC",
    "K": "AAG", "L": "CTG", "M": "ATG", "N": "AAC",
    "P": "CCC", "Q": "CAG", "R": "AGG", "S": "AGC",
    "T": "ACC", "V": "GTG", "W": "TGG", "Y": "TAC",
    "*": "TGA", "X": "NNN",
}


def _fallback_backtranslate(protein_sequence: str, sequence_id: str, organism: str) -> CodonOptimizationResult:
    """
    Naïve codon assignment using the most-frequent human codon per AA.

    Called when the ``codon_optimization`` module is not installed.  The
    result is flagged with ``method = "naive_fallback"`` and a warning so
    callers can detect and replace it.

    Parameters
    ----------
    protein_sequence : str
    sequence_id : str
    organism : str

    Returns
    -------
    CodonOptimizationResult
    """
    codons = []
    unknown = []
    for i, aa in enumerate(protein_sequence):
        codon = _FALLBACK_CODON_TABLE.get(aa.upper())
        if codon is None:
            codon = "NNN"
            unknown.append(f"{aa}{i+1}")
        codons.append(codon)
    dna = "".join(codons)
    warnings: list[str] = [
        "codon_optimization module not found; using naïve most-frequent-codon "
        "back-translation (CAI not computed)."
    ]
    if unknown:
        warnings.append(
            f"Unknown amino acids at positions {unknown[:10]}"
            + (f" (and {len(unknown)-10} more)" if len(unknown) > 10 else "")
            + " — replaced with NNN."
        )
    return CodonOptimizationResult(
        sequence_id=sequence_id,
        protein_sequence=protein_sequence,
        optimised_dna=dna,
        cai=-1.0,
        organism=organism,
        method="naive_fallback",
        warnings=warnings,
    )


def _run_codon_optimization(
    protein_sequence: str,
    sequence_id: str,
    organism: str,
    seed: int | None,
) -> CodonOptimizationResult:
    """
    Attempt to import ``codon_optimization`` and run it; fall back to naïve.

    Parameters
    ----------
    protein_sequence : str
    sequence_id : str
    organism : str
    seed : int | None

    Returns
    -------
    CodonOptimizationResult
    """
    try:
        import codon_optimization  # type: ignore[import]
        return codon_optimization.optimise_codons(
            protein_sequence=protein_sequence,
            sequence_id=sequence_id,
            organism=organism,
            seed=seed,
        )
    except ImportError:
        logger.warning(
            "codon_optimization module not installed; using naïve fallback."
        )
        return _fallback_backtranslate(protein_sequence, sequence_id, organism)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def _validate_config(config: PipelineConfig) -> None:
    """
    Validate a :class:`PipelineConfig` before the pipeline starts.

    Raises
    ------
    InvalidConfigError
        On any validation failure.
    """
    if not config.raw_dna or not config.raw_dna.strip():
        raise InvalidConfigError("'raw_dna' must be a non-empty string.")
    if not config.sequence_id or not config.sequence_id.strip():
        raise InvalidConfigError("'sequence_id' must be a non-empty string.")
    if not (1 <= config.esm3_z_threshold <= 10):
        raise InvalidConfigError(
            f"'esm3_z_threshold' must be in [1, 10], got {config.esm3_z_threshold}."
        )
    if config.translation_frame not in (0, 1, 2, 3):
        raise InvalidConfigError(
            f"'translation_frame' must be 0 (auto), 1, 2, or 3; "
            f"got {config.translation_frame}."
        )
    if config.rfdiffusion_num_designs < 1:
        raise InvalidConfigError(
            f"'rfdiffusion_num_designs' must be ≥ 1, got {config.rfdiffusion_num_designs}."
        )
    if config.proteinmpnn_num_seqs < 1:
        raise InvalidConfigError(
            f"'proteinmpnn_num_seqs' must be ≥ 1, got {config.proteinmpnn_num_seqs}."
        )
    if config.proteinmpnn_temperatures is not None:
        for t in config.proteinmpnn_temperatures:
            if not (0.0 < t <= 2.0):
                raise InvalidConfigError(
                    f"All ProteinMPNN temperatures must be in (0, 2.0]; got {t}."
                )
    if config.reference_pdb_path is not None:
        pdb = Path(config.reference_pdb_path)
        if not pdb.exists():
            raise InvalidConfigError(
                f"'reference_pdb_path' does not exist: {pdb}"
            )


# ---------------------------------------------------------------------------
# Primary public API
# ---------------------------------------------------------------------------


def run_pipeline(config: PipelineConfig) -> PipelineResult:
    """
    Execute the full GeneForge pipeline and return a :class:`PipelineResult`.

    This is the single entry point for all pipeline runs.  Call it with a
    fully populated :class:`PipelineConfig`; it handles all stage dispatch,
    error handling, metadata collection, and output serialisation.

    Parameters
    ----------
    config : PipelineConfig
        Complete run configuration.  See :class:`PipelineConfig` for
        per-field documentation.

    Returns
    -------
    PipelineResult
        Populated with all stage outputs.  :attr:`PipelineResult.success`
        is ``False`` if any mandatory stage failed; partial results up to
        the point of failure are always present.

    Raises
    ------
    InvalidConfigError
        If the configuration fails validation before any stage runs.

    Notes
    -----
    This function is synchronous.  For async contexts, wrap with
    ``asyncio.to_thread``::

        result = await asyncio.to_thread(run_pipeline, config)
    """
    # ------------------------------------------------------------------ #
    # Initialise run context                                               #
    # ------------------------------------------------------------------ #
    run_id     = str(uuid.uuid4())
    seq_id     = config.sequence_id
    t_pipeline = time.perf_counter()

    logger.info(
        "GeneForge pipeline starting | run_id=%s | seq_id=%s | seed=%s",
        run_id, seq_id, config.seed,
    )

    _validate_config(config)

    base_dir = Path(config.output_dir).resolve()
    base_dir.mkdir(parents=True, exist_ok=True)

    # Shared state threaded through stages
    metadata:        dict[str, StageMetadata] = {}
    skipped_stages:  list[Stage] = []
    failed_stages:   list[Stage] = []
    pipeline_warnings: list[str] = []

    # Stage outputs (filled in as we go)
    dna_record       = None
    protein_record   = None
    instability_rep  = None
    healing_rep      = None
    inv_fold_result  = None
    codon_result: CodonOptimizationResult | None = None
    final_dna: str | None = None

    # ------------------------------------------------------------------ #
    # Stage 1 — DNA Preprocessing (FATAL)                                 #
    # ------------------------------------------------------------------ #
    logger.info("[Stage 1/6] DNA Preprocessing …")
    stage = Stage.DNA_PREPROCESSING
    seed1 = _stage_seed(config.seed, stage)
    t0 = time.perf_counter()
    dna_dir = _make_output_dir(base_dir, "dna")
    try:
        import dna_preprocessing as dna_pp  # type: ignore[import]

        dna_record = dna_pp.preprocess_sequence(
            raw_sequence=config.raw_dna,
            sequence_id=seq_id,
            reference_sequence=config.reference_dna,
        )

        fasta_path = dna_dir / f"{seq_id}_preprocessed.fasta"
        mut_path   = dna_dir / f"{seq_id}_mutations.json"
        dna_pp.write_fasta(dna_record, fasta_path, overwrite=config.overwrite)
        dna_pp.write_mutation_report(dna_record, mut_path, overwrite=config.overwrite)

        elapsed = time.perf_counter() - t0
        _record_stage(
            metadata, stage, "success", elapsed,
            seed=seed1,
            output_paths=[str(fasta_path), str(mut_path)],
            warnings=_collect_warnings(dna_record),
        )
        logger.info("  Stage 1 OK: L=%d, GC=%.2f%%", dna_record.length, dna_record.gc_content * 100)

    except Exception as exc:
        elapsed = time.perf_counter() - t0
        failed_stages.append(stage)
        _record_stage(
            metadata, stage, "failed", elapsed,
            error=str(exc), error_type=type(exc).__name__,
        )
        logger.error("  Stage 1 FAILED: %s: %s", type(exc).__name__, exc)
        raise FatalStageError(stage, exc) from exc

    # ------------------------------------------------------------------ #
    # Stage 2 — Protein Translation (FATAL)                               #
    # ------------------------------------------------------------------ #
    logger.info("[Stage 2/6] Protein Translation …")
    stage = Stage.PROTEIN_TRANSLATION
    seed2 = _stage_seed(config.seed, stage)
    t0 = time.perf_counter()
    protein_dir = _make_output_dir(base_dir, "protein")
    try:
        import protein_translation as pt  # type: ignore[import]

        if config.translation_frame == 0:
            # Auto: find best ORF
            orf = pt.find_best_orf(dna_record)
            protein_record = pt.translate_dna(dna_record, frame=orf.frame, strand=orf.strand)
        else:
            protein_record = pt.translate_dna(dna_record, frame=config.translation_frame)

        prot_fasta = protein_dir / f"{seq_id}.fasta"
        prot_rep   = protein_dir / f"{seq_id}_translation.json"
        pt.write_protein_fasta(protein_record, prot_fasta, overwrite=config.overwrite)
        pt.write_translation_report(protein_record, prot_rep, overwrite=config.overwrite)

        elapsed = time.perf_counter() - t0
        _record_stage(
            metadata, stage, "success", elapsed,
            seed=seed2,
            output_paths=[str(prot_fasta), str(prot_rep)],
            warnings=_collect_warnings(protein_record),
        )
        logger.info("  Stage 2 OK: protein_length=%d", protein_record.length)

    except Exception as exc:
        elapsed = time.perf_counter() - t0
        failed_stages.append(stage)
        _record_stage(
            metadata, stage, "failed", elapsed,
            error=str(exc), error_type=type(exc).__name__,
        )
        logger.error("  Stage 2 FAILED: %s: %s", type(exc).__name__, exc)
        raise FatalStageError(stage, exc) from exc

    # ------------------------------------------------------------------ #
    # Stage 3 — ESM-3 Instability Detection (FATAL)                       #
    # ------------------------------------------------------------------ #
    logger.info("[Stage 3/6] ESM-3 Instability Detection …")
    stage = Stage.ESM3_INSTABILITY
    seed3 = _stage_seed(config.seed, stage)
    t0 = time.perf_counter()
    instability_dir = _make_output_dir(base_dir, "instability")
    try:
        import esm3_instability_detector as esm  # type: ignore[import]

        instability_rep = esm.detect_instability(
            protein_input=protein_record.sequence,
            model_id=config.esm3_model_id,
            device=config.device,
            z_threshold=config.esm3_z_threshold,
            ppl_abs_threshold=config.esm3_ppl_abs_threshold,
            seed=seed3,
        )
        # Attach sequence_id so downstream stages can use it
        instability_rep.sequence_id = seq_id

        inst_report_path = instability_dir / f"{seq_id}_instability.json"
        inst_report_path.write_text(instability_rep.to_json(), encoding="utf-8")

        n_spikes = len(instability_rep.spike_regions)
        elapsed = time.perf_counter() - t0
        _record_stage(
            metadata, stage, "success", elapsed,
            seed=seed3,
            output_paths=[str(inst_report_path)],
            warnings=_collect_warnings(instability_rep),
        )
        logger.info(
            "  Stage 3 OK: spikes=%d, mean_ppl=%.3f",
            n_spikes, instability_rep.mean_perplexity,
        )

        if n_spikes == 0:
            msg = "ESM-3 found no instability spikes — protein appears stable."
            logger.info("  %s", msg)
            pipeline_warnings.append(msg)

            if config.stop_on_no_instability:
                logger.info(
                    "  stop_on_no_instability=True: stopping after Stage 3."
                )
                skipped_stages += [
                    Stage.RFDIFFUSION_HEALING,
                    Stage.INVERSE_FOLDING,
                    Stage.CODON_OPTIMIZATION,
                ]
                for s in skipped_stages:
                    _record_stage(metadata, s, "skipped", 0.0)
                # Use original protein for final DNA (naïve back-translation)
                codon_result = _fallback_backtranslate(
                    protein_record.sequence, seq_id, config.codon_organism
                )
                final_dna = codon_result.optimised_dna
                total = time.perf_counter() - t_pipeline
                result = PipelineResult(
                    run_id=run_id, sequence_id=seq_id, success=True,
                    config=config, total_elapsed_s=total,
                    final_corrected_dna=final_dna,
                    dna_record=dna_record, protein_record=protein_record,
                    instability_report=instability_rep,
                    codon_result=codon_result,
                    skipped_stages=skipped_stages,
                    failed_stages=failed_stages,
                    stage_metadata=metadata,
                    warnings=pipeline_warnings,
                )
                logger.info("Pipeline early-exit: %s", result.summary())
                return result

    except Exception as exc:
        elapsed = time.perf_counter() - t0
        failed_stages.append(stage)
        _record_stage(
            metadata, stage, "failed", elapsed,
            error=str(exc), error_type=type(exc).__name__,
        )
        logger.error("  Stage 3 FAILED: %s: %s", type(exc).__name__, exc)
        raise FatalStageError(stage, exc) from exc

    # ------------------------------------------------------------------ #
    # Stage 4 — RFdiffusion Healing (RECOVERABLE)                         #
    # ------------------------------------------------------------------ #
    logger.info("[Stage 4/6] RFdiffusion Structural Healing …")
    stage = Stage.RFDIFFUSION_HEALING
    seed4 = _stage_seed(config.seed, stage)
    t0 = time.perf_counter()
    healing_dir = _make_output_dir(base_dir, "healing")

    _pdb_for_stage5 = config.reference_pdb_path   # may be updated if healing succeeds

    if config.reference_pdb_path is None:
        # No PDB provided: skip structural stages
        msg = (
            "No reference_pdb_path provided; "
            "skipping Stage 4 (RFdiffusion) and Stage 5 (ProteinMPNN)."
        )
        logger.warning("  %s", msg)
        pipeline_warnings.append(msg)
        skipped_stages += [Stage.RFDIFFUSION_HEALING, Stage.INVERSE_FOLDING]
        _record_stage(metadata, Stage.RFDIFFUSION_HEALING, "skipped", 0.0,
                      warnings=[msg])
        _record_stage(metadata, Stage.INVERSE_FOLDING, "skipped", 0.0)
    else:
        try:
            import rfdiffusion_healer as rfd  # type: ignore[import]

            healing_rep = rfd.heal_protein(
                instability_report=instability_rep,
                input_pdb_path=config.reference_pdb_path,
                output_dir=healing_dir,
                checkpoint_path=config.rfdiffusion_checkpoint,
                device=config.device,
                noise_scale=config.rfdiffusion_noise_scale,
                num_designs=config.rfdiffusion_num_designs,
                flank_residues=config.rfdiffusion_flank_residues,
                region_merge_gap=config.rfdiffusion_merge_gap,
                seed=seed4,
                overwrite=config.overwrite,
            )

            healing_rep_path = healing_dir / f"{seq_id}_healing.json"
            rfd.write_healing_report(healing_rep, healing_rep_path, overwrite=config.overwrite)

            _pdb_for_stage5 = healing_rep.output_pdb_path

            elapsed = time.perf_counter() - t0
            _record_stage(
                metadata, stage, "success", elapsed,
                seed=seed4,
                output_paths=[str(_pdb_for_stage5), str(healing_rep_path)],
                warnings=_collect_warnings(healing_rep),
            )
            logger.info(
                "  Stage 4 OK: mean_plddt=%.2f, healed=%s",
                healing_rep.mean_plddt, healing_rep.healing_success,
            )

        except Exception as exc:
            elapsed = time.perf_counter() - t0
            # Check for the known recoverable case: no regions to repair
            try:
                import rfdiffusion_healer as rfd  # type: ignore[import]
                no_repair = rfd.NoRegionsToRepairError
            except ImportError:
                no_repair = type(None)

            if isinstance(exc, no_repair):
                msg = f"Stage 4 skip: no regions to repair ({exc})"
                logger.info("  %s", msg)
                pipeline_warnings.append(msg)
                skipped_stages.append(stage)
                _record_stage(metadata, stage, "skipped", elapsed,
                              warnings=[msg])
            else:
                # Recoverable but unexpected: warn and skip Stage 5 too
                msg = (
                    f"Stage 4 error ({type(exc).__name__}: {exc}); "
                    "skipping structural healing and inverse folding."
                )
                logger.warning("  %s", msg)
                pipeline_warnings.append(msg)
                failed_stages.append(stage)
                _record_stage(
                    metadata, stage, "failed", elapsed,
                    error=str(exc), error_type=type(exc).__name__,
                )
                skipped_stages.append(Stage.INVERSE_FOLDING)
                _record_stage(metadata, Stage.INVERSE_FOLDING, "skipped", 0.0,
                              warnings=["Skipped due to Stage 4 failure."])

    # ------------------------------------------------------------------ #
    # Stage 5 — ProteinMPNN Inverse Folding (RECOVERABLE)                 #
    # ------------------------------------------------------------------ #
    if Stage.INVERSE_FOLDING not in skipped_stages:
        logger.info("[Stage 5/6] ProteinMPNN Inverse Folding …")
        stage = Stage.INVERSE_FOLDING
        seed5 = _stage_seed(config.seed, stage)
        t0 = time.perf_counter()
        inv_dir = _make_output_dir(base_dir, "inverse_fold")

        # Construct a minimal healing_report proxy if Stage 4 was skipped
        # but we have a PDB (no-regions case).
        _healing_for_mpnn = healing_rep

        if _healing_for_mpnn is None and _pdb_for_stage5 is not None:
            # Build a lightweight proxy that satisfies run_inverse_folding's interface
            _healing_for_mpnn = _make_healing_proxy(
                sequence_id=seq_id,
                output_pdb_path=_pdb_for_stage5,
                repair_regions=[],
            )

        if _healing_for_mpnn is None:
            msg = "Stage 5 skipped: no PDB available for inverse folding."
            logger.info("  %s", msg)
            skipped_stages.append(stage)
            _record_stage(metadata, stage, "skipped", 0.0, warnings=[msg])
        else:
            try:
                import proteinmpnn_inverse_fold as pmpnn  # type: ignore[import]

                inv_fold_result = pmpnn.run_inverse_folding(
                    healing_report=_healing_for_mpnn,
                    output_dir=inv_dir,
                    checkpoint_path=config.proteinmpnn_checkpoint,
                    chain_id=config.proteinmpnn_chain_id,
                    temperatures=config.proteinmpnn_temperatures,
                    num_seqs=config.proteinmpnn_num_seqs,
                    device=config.device,
                    seed=seed5,
                    overwrite=config.overwrite,
                )

                elapsed = time.perf_counter() - t0
                _record_stage(
                    metadata, stage, "success", elapsed,
                    seed=seed5,
                    output_paths=[
                        str(inv_dir / f"{seq_id}_designed.fasta"),
                        str(inv_dir / f"{seq_id}_designed.json"),
                    ],
                    warnings=_collect_warnings(inv_fold_result),
                )
                logger.info(
                    "  Stage 5 OK: seqs=%d, best_score=%.4f",
                    len(inv_fold_result.sequences),
                    inv_fold_result.best_sequence.sequence_score,
                )

            except Exception as exc:
                elapsed = time.perf_counter() - t0
                msg = (
                    f"Stage 5 error ({type(exc).__name__}: {exc}); "
                    "codon optimisation will use original protein sequence."
                )
                logger.warning("  %s", msg)
                pipeline_warnings.append(msg)
                failed_stages.append(stage)
                _record_stage(
                    metadata, stage, "failed", elapsed,
                    error=str(exc), error_type=type(exc).__name__,
                )

    # ------------------------------------------------------------------ #
    # Stage 6 — Codon Optimisation (RECOVERABLE)                          #
    # ------------------------------------------------------------------ #
    logger.info("[Stage 6/6] Codon Optimisation …")
    stage = Stage.CODON_OPTIMIZATION
    seed6 = _stage_seed(config.seed, stage)
    t0 = time.perf_counter()
    codon_dir = _make_output_dir(base_dir, "codons")

    # Choose the protein sequence to back-translate
    if inv_fold_result is not None:
        protein_to_codon = inv_fold_result.best_sequence.sequence
        protein_source   = "proteinmpnn_best"
    else:
        protein_to_codon = protein_record.sequence
        protein_source   = "original_translation"

    if protein_to_codon:
        # Strip stop codon character if present
        protein_to_codon = protein_to_codon.rstrip("*")

    try:
        codon_result = _run_codon_optimization(
            protein_sequence=protein_to_codon,
            sequence_id=seq_id,
            organism=config.codon_organism,
            seed=seed6,
        )
        final_dna = codon_result.optimised_dna

        codon_fasta = codon_dir / f"{seq_id}_optimised.fasta"
        codon_json  = codon_dir / f"{seq_id}_codons.json"

        # Write FASTA
        _write_dna_fasta(
            sequence=final_dna,
            sequence_id=seq_id,
            description=(
                f"codon_optimised|method={codon_result.method}"
                f"|protein_source={protein_source}"
                f"|cai={codon_result.cai:.4f}"
            ),
            output_path=codon_fasta,
            overwrite=config.overwrite,
        )
        codon_json.write_text(codon_result.to_json(), encoding="utf-8")

        elapsed = time.perf_counter() - t0
        _record_stage(
            metadata, stage, "success", elapsed,
            seed=seed6,
            output_paths=[str(codon_fasta), str(codon_json)],
            warnings=_collect_warnings(codon_result),
        )
        logger.info(
            "  Stage 6 OK: dna_len=%d, method=%s, CAI=%s",
            len(final_dna), codon_result.method,
            f"{codon_result.cai:.4f}" if codon_result.cai >= 0 else "n/a",
        )

    except Exception as exc:
        elapsed = time.perf_counter() - t0
        msg = (
            f"Stage 6 error ({type(exc).__name__}: {exc}); "
            "using naïve fallback back-translation."
        )
        logger.warning("  %s", msg)
        pipeline_warnings.append(msg)
        failed_stages.append(stage)
        _record_stage(
            metadata, stage, "failed", elapsed,
            error=str(exc), error_type=type(exc).__name__,
        )
        # Fallback: naïve back-translation so we always have a final_dna
        codon_result = _fallback_backtranslate(
            protein_to_codon, seq_id, config.codon_organism
        )
        final_dna = codon_result.optimised_dna

    # ------------------------------------------------------------------ #
    # Stage 7 — Assemble PipelineResult                                   #
    # ------------------------------------------------------------------ #
    total_elapsed = time.perf_counter() - t_pipeline
    mandatory_failed = any(
        s in failed_stages
        for s in [Stage.DNA_PREPROCESSING, Stage.PROTEIN_TRANSLATION, Stage.ESM3_INSTABILITY]
    )
    success = not mandatory_failed and bool(final_dna)

    result = PipelineResult(
        run_id=run_id,
        sequence_id=seq_id,
        success=success,
        config=config,
        total_elapsed_s=total_elapsed,
        final_corrected_dna=final_dna,
        dna_record=dna_record,
        protein_record=protein_record,
        instability_report=instability_rep,
        healing_report=healing_rep,
        inverse_fold_result=inv_fold_result,
        codon_result=codon_result,
        skipped_stages=skipped_stages,
        failed_stages=failed_stages,
        stage_metadata=metadata,
        warnings=pipeline_warnings,
    )

    # Write consolidated result JSON
    try:
        result.write(overwrite=config.overwrite)
    except FileExistsError:
        logger.warning(
            "Pipeline result JSON already exists and overwrite=False; skipping write."
        )
    except Exception as exc:
        logger.warning("Could not write pipeline result JSON: %s", exc)

    logger.info("Pipeline complete: %s", result.summary())
    return result


# ---------------------------------------------------------------------------
# HealingReport proxy for when Stage 4 was skipped but PDB is available
# ---------------------------------------------------------------------------


def _make_healing_proxy(
    sequence_id: str,
    output_pdb_path: Path | str,
    repair_regions: list,
) -> object:
    """
    Build a minimal object that satisfies ``run_inverse_folding``'s interface.

    When Stage 4 is skipped (no instability spikes found, but a PDB exists),
    ProteinMPNN should still be run to generate a sequence for the unchanged
    backbone.  This proxy presents the reference PDB as the "healed" output.

    The proxy only needs the three attributes that
    ``proteinmpnn_inverse_fold.run_inverse_folding`` reads:
      - ``sequence_id``
      - ``output_pdb_path``
      - ``repair_regions``

    Parameters
    ----------
    sequence_id : str
    output_pdb_path : Path | str
    repair_regions : list
        Empty list → all positions are "fixed" in ProteinMPNN.

    Returns
    -------
    object
        A plain namespace object with the required attributes.
    """
    class _HealingProxy:
        pass

    proxy = _HealingProxy()
    proxy.sequence_id     = sequence_id
    proxy.output_pdb_path = Path(output_pdb_path)
    proxy.repair_regions  = repair_regions
    return proxy


# ---------------------------------------------------------------------------
# DNA FASTA writer (orchestrator-local, avoids depending on dna_preprocessing
# for a simple text format)
# ---------------------------------------------------------------------------


def _write_dna_fasta(
    sequence: str,
    sequence_id: str,
    description: str,
    output_path: Path,
    overwrite: bool,
    line_width: int = 60,
) -> None:
    """
    Write a single DNA sequence to a FASTA file.

    Parameters
    ----------
    sequence : str
    sequence_id : str
    description : str
    output_path : Path
    overwrite : bool
    line_width : int
    """
    if output_path.exists() and not overwrite:
        logger.debug(
            "DNA FASTA '%s' already exists; skipping write.", output_path
        )
        return
    header = f">{sequence_id} {description}"
    wrapped = "\n".join(
        sequence[i : i + line_width] for i in range(0, len(sequence), line_width)
    )
    output_path.write_text(f"{header}\n{wrapped}\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Convenience function: run from a dict / simple arguments
# ---------------------------------------------------------------------------


def run_pipeline_from_dict(config_dict: dict) -> PipelineResult:
    """
    Create a :class:`PipelineConfig` from a dict and run the pipeline.

    Useful for JSON-driven API calls.

    Parameters
    ----------
    config_dict : dict
        Keys correspond to :class:`PipelineConfig` field names.  ``raw_dna``
        is required; all others are optional.

    Returns
    -------
    PipelineResult

    Raises
    ------
    InvalidConfigError
        If required keys are absent or values fail validation.
    TypeError
        If unknown keys are passed.

    Examples
    --------
    >>> result = run_pipeline_from_dict({
    ...     "raw_dna": "ATGGCCTAA",
    ...     "sequence_id": "test_protein",
    ...     "output_dir": "/tmp/geneforge_out",
    ...     "seed": 7,
    ... })
    """
    try:
        config = PipelineConfig(**config_dict)
    except TypeError as exc:
        raise InvalidConfigError(
            f"Invalid configuration key or type: {exc}"
        ) from exc
    return run_pipeline(config)


# ---------------------------------------------------------------------------
# Memory management helpers
# ---------------------------------------------------------------------------


def release_all_models() -> None:
    """
    Unload all stage models from memory and clear CUDA caches.

    Calls each stage module's ``release_model()`` function if the module
    is importable.  Safe to call multiple times.  Useful before loading a
    fresh pipeline run on a memory-constrained GPU.
    """
    for module_name in (
        "esm3_instability_detector",
        "rfdiffusion_healer",
        "proteinmpnn_inverse_fold",
    ):
        try:
            import importlib
            mod = importlib.import_module(module_name)
            if hasattr(mod, "release_model"):
                mod.release_model()
                logger.debug("Released model from '%s'.", module_name)
        except ImportError:
            pass
        except Exception as exc:
            logger.warning(
                "Could not release model from '%s': %s", module_name, exc
            )


def models_loaded_status() -> dict[str, bool]:
    """
    Return a dict indicating which stage models are currently in memory.

    Returns
    -------
    dict[str, bool]
        Keys: ``"esm3"``, ``"rfdiffusion"``, ``"proteinmpnn"``.
    """
    status = {}
    for key, module_name in (
        ("esm3",        "esm3_instability_detector"),
        ("rfdiffusion", "rfdiffusion_healer"),
        ("proteinmpnn", "proteinmpnn_inverse_fold"),
    ):
        try:
            import importlib
            mod = importlib.import_module(module_name)
            status[key] = bool(getattr(mod, "model_is_loaded", lambda: False)())
        except ImportError:
            status[key] = False
    return status
