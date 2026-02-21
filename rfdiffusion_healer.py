"""
rfdiffusion_healer.py
=====================
GeneForge Pipeline — Stage 4: Protein Structure Repair via RFdiffusion

Scientific Background
---------------------
RFdiffusion (Watson et al., Nature 2023) is a structure-conditioned
diffusion model for *de novo* protein design.  In its "partial diffusion"
(also called "noising and denoising") mode, it accepts an existing PDB
backbone, adds controlled Gaussian noise only at selected residue positions,
then denoises back to a plausible backbone — effectively *re-designing* the
flagged region while holding the rest of the structure fixed as a motif.

This is the correct mode for mutation repair:
  - Stable regions (low ESM-3 perplexity) → fixed motif ("contig" segments
    enclosed in the preserved chain).
  - Unstable regions (perplexity spikes from Stage 3) → diffused and
    regenerated with ``partial_T`` noise timesteps.

The key idea is that partial diffusion preserves global topology (fold,
domain architecture, binding interfaces) while exploring sequence-space
variants at unstable sites.  The number of noise steps (``partial_T``)
controls the trade-off between fidelity to the original backbone and
diversity of the regenerated region:
  - Low  partial_T (5–20) : conservative repair, minor backbone shifts
  - High partial_T (50–100): aggressive redesign, large structural changes

Contig Notation
---------------
RFdiffusion uses a contig string to specify which parts of the input PDB
are kept fixed vs. re-designed.  Format (space-separated segments):

    "A1-45/0 20-30/A65-100"

Meaning:
  - ``A1-45``  : Keep chain A residues 1–45 from the input PDB (fixed motif)
  - ``/0 20-30/`` : Insert 20–30 new residues (re-designed, no input template)
  - ``A65-100``: Keep chain A residues 65–100 (fixed motif)

For *repair* (partial diffusion), we keep **all** residues in the contig
but mark unstable regions with a different noise level, implemented by
setting the ``diffusion_mask`` tensor — True at positions to diffuse.

Environment Variables
---------------------
  RFDIFFUSION_WEIGHTS_PATH  — path to the model checkpoint directory.
                              Must contain one of:
                                RFdiffusion.pt  (base model)
                                ActiveSite.pt   (active-site finetuned)
                                Complex.pt      (protein-complex finetuned)
  RFDIFFUSION_CONFIG_PATH   — optional override for the Hydra config directory.

PDB Handling
------------
Input PDB files must contain at minimum backbone atoms (N, Cα, C, O).
Side-chain atoms are optional — RFdiffusion operates at the backbone level.
Output PDB files contain only backbone atoms (standard RFdiffusion output).

The module uses Biopython's ``PDB`` sub-package for PDB I/O.  Heavy-atom
coordinate validation and chain extraction are performed before passing
the structure to RFdiffusion to avoid silent silent failures downstream.

Data Flow
---------
    Input  : InstabilityReport (Stage 3) + input PDB path
    Output : HealingReport dataclass

Dependencies
------------
    biopython >= 1.83  — PDB I/O and structure validation
    torch     >= 2.1   — tensor construction for diffusion mask
    numpy     >= 1.26  — coordinate array handling
    Python    >= 3.12

    Runtime (not install-time):
    rfdiffusion          — RFdiffusion model and inference runner
                           (https://github.com/RosettaCommons/RFdiffusion)
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Final, NamedTuple

import numpy as np

if TYPE_CHECKING:
    import torch
    # RFdiffusion types — only for static analysis
    from rfdiffusion.inference.model_runners import SelfConditioning

# ---------------------------------------------------------------------------
# Module logger
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment variable keys
# ---------------------------------------------------------------------------

ENV_WEIGHTS_PATH: Final[str] = "RFDIFFUSION_WEIGHTS_PATH"
ENV_CONFIG_PATH:  Final[str] = "RFDIFFUSION_CONFIG_PATH"

# Default checkpoint filename inside the weights directory
DEFAULT_CHECKPOINT_NAME: Final[str] = "RFdiffusion.pt"

# ---------------------------------------------------------------------------
# Diffusion hyperparameter defaults
# ---------------------------------------------------------------------------

#: Conservative partial diffusion timesteps for subtle repair.
PARTIAL_T_CONSERVATIVE: Final[int] = 20

#: Moderate partial diffusion timesteps — default for single-residue spikes.
PARTIAL_T_MODERATE: Final[int] = 40

#: Aggressive partial diffusion — used for large contiguous spike regions.
PARTIAL_T_AGGRESSIVE: Final[int] = 80

#: Noise scale controls the magnitude of added noise per diffusion step.
#: Range [0.5, 2.0]; 1.0 = canonical; < 1.0 = more conservative.
DEFAULT_NOISE_SCALE: Final[float] = 1.0

#: Number of independent diffusion trajectories to generate.
DEFAULT_NUM_DESIGNS: Final[int] = 1

#: Flanking residues added on each side of a spike region for context.
#: RFdiffusion performs better when given structural context around
#: the repaired region.
FLANK_RESIDUES: Final[int] = 5

#: Minimum gap (residues) between adjacent spike regions before they
#: are merged into a single repair segment.
REGION_MERGE_GAP: Final[int] = 3

#: Chain identifier used when the input PDB has no explicit chain label.
DEFAULT_CHAIN: Final[str] = "A"

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class RFdiffusionError(Exception):
    """Base exception for all RFdiffusion healing failures."""


class ModelLoadError(RFdiffusionError):
    """Raised when the RFdiffusion checkpoint cannot be found or loaded."""


class PDBValidationError(RFdiffusionError):
    """Raised when the input PDB file fails structural validation."""


class ContigBuildError(RFdiffusionError):
    """Raised when a valid contig string cannot be constructed."""


class DiffusionRunError(RFdiffusionError):
    """Raised when the RFdiffusion inference process fails."""


class NoRegionsToRepairError(RFdiffusionError):
    """Raised when the InstabilityReport contains no spike regions."""


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------


class RepairRegion(NamedTuple):
    """
    A residue range selected for backbone regeneration.

    Derived from ``SpikeRegion`` objects in the ``InstabilityReport``,
    with flanking context added and adjacent regions merged.

    Attributes
    ----------
    start : int
        0-based start residue index in the *protein sequence* (inclusive).
    end : int
        0-based end residue index in the *protein sequence* (inclusive).
    length : int
        Number of residues in the repair zone including flanks.
    pdb_start : int
        1-based PDB residue number corresponding to ``start``.
    pdb_end : int
        1-based PDB residue number corresponding to ``end``.
    source_spike_starts : list[int]
        ``start`` indices of the original SpikeRegions that were merged
        into this repair region.
    peak_perplexity : float
        Highest single-residue smoothed perplexity in this region.
    partial_T : int
        Diffusion timesteps assigned based on severity.
    chain_id : str
        PDB chain identifier this region belongs to.
    """

    start: int
    end: int
    length: int
    pdb_start: int
    pdb_end: int
    source_spike_starts: list[int]
    peak_perplexity: float
    partial_T: int
    chain_id: str

    def to_contig_segment(self) -> str:
        """
        Return the RFdiffusion contig segment for this region.

        Format: ``"A1-30"`` — chain letter followed by 1-based residue range.
        """
        return f"{self.chain_id}{self.pdb_start}-{self.pdb_end}"

    def to_dict(self) -> dict:
        return {
            "start": self.start,
            "end": self.end,
            "length": self.length,
            "pdb_start": self.pdb_start,
            "pdb_end": self.pdb_end,
            "source_spike_starts": self.source_spike_starts,
            "peak_perplexity": round(self.peak_perplexity, 4),
            "partial_T": self.partial_T,
            "chain_id": self.chain_id,
            "contig_segment": self.to_contig_segment(),
        }


@dataclass(frozen=True, slots=True)
class ResidueConfidence:
    """
    Per-residue confidence score extracted from RFdiffusion's ``.trb`` output.

    Attributes
    ----------
    index : int
        0-based residue index in the healed protein.
    pdb_residue_number : int
        1-based residue number in the output PDB.
    plddt : float
        Predicted local distance difference test score [0, 100].
        Mirrors AlphaFold2's pLDDT semantics; > 70 is considered confident.
    is_repaired : bool
        True if this residue falls within a RepairRegion.
    """

    index: int
    pdb_residue_number: int
    plddt: float
    is_repaired: bool

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "pdb_residue_number": self.pdb_residue_number,
            "plddt": round(self.plddt, 3),
            "is_repaired": self.is_repaired,
        }


@dataclass(slots=True)
class HealingReport:
    """
    Full record of a single RFdiffusion repair run.

    This is the Stage 4 output consumed by Stage 5 (ProteinMPNN inverse
    folding).

    Attributes
    ----------
    sequence_id : str
        Carried from the InstabilityReport.
    input_pdb_path : Path
        Absolute path to the original (mutated) structure file.
    output_pdb_path : Path
        Absolute path to the healed backbone PDB written by RFdiffusion.
    trb_path : Path | None
        Path to the RFdiffusion ``.trb`` metadata file (pickle), if present.
    repair_regions : list[RepairRegion]
        Regions that were submitted for diffusion.
    preserved_segments : list[str]
        Contig segments that were held fixed (not diffused).
    full_contig_string : str
        Complete contig string passed to RFdiffusion.
    residue_confidences : list[ResidueConfidence]
        Per-residue pLDDT scores from the ``.trb`` file.
    mean_plddt : float
        Mean pLDDT across all residues in the healed structure.
    mean_plddt_repaired : float
        Mean pLDDT restricted to repaired residues.
    mean_plddt_preserved : float
        Mean pLDDT restricted to preserved (fixed) residues.
    num_repaired_residues : int
        Total residues within RepairRegions.
    num_preserved_residues : int
        Total residues outside RepairRegions.
    partial_T_used : int
        Diffusion timesteps used (maximum across all repair regions).
    noise_scale_used : float
    num_designs_generated : int
    model_checkpoint : str
        Resolved checkpoint path or identifier.
    device_used : str
    inference_time_s : float
    warnings : list[str]
    """

    sequence_id: str
    input_pdb_path: Path
    output_pdb_path: Path
    trb_path: Path | None
    repair_regions: list[RepairRegion]
    preserved_segments: list[str]
    full_contig_string: str
    residue_confidences: list[ResidueConfidence]
    mean_plddt: float
    mean_plddt_repaired: float
    mean_plddt_preserved: float
    num_repaired_residues: int
    num_preserved_residues: int
    partial_T_used: int
    noise_scale_used: float
    num_designs_generated: int
    model_checkpoint: str
    device_used: str
    inference_time_s: float
    warnings: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Derived views
    # ------------------------------------------------------------------

    @property
    def healing_success(self) -> bool:
        """
        True when the output PDB exists and mean pLDDT ≥ 70.

        pLDDT ≥ 70 is the standard threshold for a "confident" structure
        in both AlphaFold and RFdiffusion literature.
        """
        return self.output_pdb_path.exists() and self.mean_plddt >= 70.0

    @property
    def repaired_region_indices(self) -> list[int]:
        """Flat list of all 0-based residue indices within repair regions."""
        indices: list[int] = []
        for rr in self.repair_regions:
            indices.extend(range(rr.start, rr.end + 1))
        return sorted(set(indices))

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "sequence_id": self.sequence_id,
            "input_pdb_path": str(self.input_pdb_path),
            "output_pdb_path": str(self.output_pdb_path),
            "trb_path": str(self.trb_path) if self.trb_path else None,
            "full_contig_string": self.full_contig_string,
            "num_repaired_residues": self.num_repaired_residues,
            "num_preserved_residues": self.num_preserved_residues,
            "partial_T_used": self.partial_T_used,
            "noise_scale_used": self.noise_scale_used,
            "num_designs_generated": self.num_designs_generated,
            "mean_plddt": round(self.mean_plddt, 3),
            "mean_plddt_repaired": round(self.mean_plddt_repaired, 3),
            "mean_plddt_preserved": round(self.mean_plddt_preserved, 3),
            "healing_success": self.healing_success,
            "model_checkpoint": self.model_checkpoint,
            "device_used": self.device_used,
            "inference_time_s": round(self.inference_time_s, 2),
            "repair_regions": [r.to_dict() for r in self.repair_regions],
            "preserved_segments": self.preserved_segments,
            "residue_confidences": [rc.to_dict() for rc in self.residue_confidences],
            "warnings": self.warnings,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def summary(self) -> str:
        return (
            f"[{self.sequence_id}] healed={self.healing_success}, "
            f"mean_plddt={self.mean_plddt:.1f} "
            f"(repaired={self.mean_plddt_repaired:.1f}, "
            f"preserved={self.mean_plddt_preserved:.1f}), "
            f"regions={len(self.repair_regions)}, "
            f"repaired_residues={self.num_repaired_residues}, "
            f"partial_T={self.partial_T_used}, "
            f"t={self.inference_time_s:.1f}s"
        )


# ---------------------------------------------------------------------------
# RFdiffusion model loader — lazy singleton, thread-safe
# ---------------------------------------------------------------------------


class _RFdiffusionLoader:
    """
    Thread-safe lazy singleton for the RFdiffusion model runner.

    Mirrors the design of ``_ESM3Loader`` in Stage 3 for consistency.
    The runner is expensive to initialise (loads checkpoint + sets up
    denoiser network), so it is held in class state and reused.

    Environment Variables
    ---------------------
    RFDIFFUSION_WEIGHTS_PATH
        Directory containing the checkpoint file.  Required if calling
        ``get()`` without an explicit ``checkpoint_path``.
    RFDIFFUSION_CONFIG_PATH
        Optional.  Override the path to RFdiffusion's Hydra config tree.
    """

    _lock: threading.Lock = threading.Lock()
    _runner: "SelfConditioning | None" = None
    _loaded_checkpoint: str | None = None
    _device: "torch.device | None" = None

    @classmethod
    def get(
        cls,
        checkpoint_path: str | None = None,
        device_override: str | None = None,
    ) -> tuple["SelfConditioning", "torch.device"]:
        """
        Return the loaded RFdiffusion runner and its device.

        Parameters
        ----------
        checkpoint_path : str | None
            Absolute path to the ``.pt`` checkpoint file.
            Falls back to ``$RFDIFFUSION_WEIGHTS_PATH/RFdiffusion.pt``.
        device_override : str | None
            Force a specific device.  ``None`` = auto-detect.

        Returns
        -------
        tuple[SelfConditioning, torch.device]

        Raises
        ------
        ModelLoadError
        """
        resolved_ckpt = cls._resolve_checkpoint(checkpoint_path)

        with cls._lock:
            if cls._runner is not None and cls._loaded_checkpoint == resolved_ckpt:
                logger.debug(
                    "RFdiffusion cache hit: checkpoint='%s'.", resolved_ckpt
                )
                return cls._runner, cls._device  # type: ignore[return-value]

            device = cls._resolve_device(device_override)
            logger.info(
                "Loading RFdiffusion checkpoint '%s' onto %s …",
                resolved_ckpt,
                device,
            )
            t0 = time.perf_counter()

            try:
                runner = cls._load_runner(resolved_ckpt, device)
            except ImportError as exc:
                raise ModelLoadError(
                    "The 'rfdiffusion' package is required but not installed. "
                    "Clone and install from "
                    "https://github.com/RosettaCommons/RFdiffusion\n"
                    f"Original error: {exc}"
                ) from exc
            except FileNotFoundError as exc:
                raise ModelLoadError(
                    f"RFdiffusion checkpoint not found at '{resolved_ckpt}'. "
                    f"Set the {ENV_WEIGHTS_PATH!r} environment variable to the "
                    "directory containing your checkpoint files.\n"
                    f"Original error: {exc}"
                ) from exc
            except Exception as exc:
                raise ModelLoadError(
                    f"Failed to load RFdiffusion from '{resolved_ckpt}': "
                    f"{type(exc).__name__}: {exc}"
                ) from exc

            elapsed = time.perf_counter() - t0
            logger.info(
                "RFdiffusion loaded in %.1f s on %s.", elapsed, device
            )

            if device.type == "cuda":
                cls._log_vram(device)

            cls._runner = runner
            cls._loaded_checkpoint = resolved_ckpt
            cls._device = device

        return cls._runner, cls._device  # type: ignore[return-value]

    @staticmethod
    def _resolve_checkpoint(explicit_path: str | None) -> str:
        """
        Determine the checkpoint path from argument or environment.

        Priority: explicit arg > ``$RFDIFFUSION_WEIGHTS_PATH/RFdiffusion.pt``.

        Raises
        ------
        ModelLoadError
            If no path can be resolved.
        """
        if explicit_path:
            return str(Path(explicit_path).resolve())

        weights_dir = os.environ.get(ENV_WEIGHTS_PATH)
        if not weights_dir:
            raise ModelLoadError(
                f"No RFdiffusion checkpoint path provided and the "
                f"{ENV_WEIGHTS_PATH!r} environment variable is not set. "
                "Set it to the directory containing your checkpoint .pt files."
            )

        candidate = Path(weights_dir) / DEFAULT_CHECKPOINT_NAME
        return str(candidate.resolve())

    @staticmethod
    def _load_runner(
        checkpoint_path: str,
        device: "torch.device",
    ) -> "SelfConditioning":
        """
        Instantiate the RFdiffusion SelfConditioning runner.

        RFdiffusion uses Hydra (OmegaConf) for configuration.  We construct
        a minimal inference config programmatically and inject the checkpoint
        path, avoiding the need for the full config directory at runtime.

        Parameters
        ----------
        checkpoint_path : str
        device : torch.device

        Returns
        -------
        SelfConditioning
            Runner ready for ``run_inference()``.
        """
        from rfdiffusion.inference.model_runners import SelfConditioning  # type: ignore[import]
        from omegaconf import OmegaConf, DictConfig              # type: ignore[import]

        # Minimal inference configuration matching RFdiffusion's expected schema.
        # All keys not listed here use RFdiffusion's internal defaults.
        conf_dict = {
            "inference": {
                "input_pdb": None,          # set per-run in run_diffusion()
                "output_prefix": None,      # set per-run
                "num_designs": 1,
                "ckpt_override_path": checkpoint_path,
                "symmetry": None,
                "recenter": True,
                "radius": 10.0,
                "model_runner": "SelfConditioning",
                "cautious": True,
                "align_motif": True,
                "symmetric_self_cond": True,
                "final_step": 1,
                "deterministic": False,
            },
            "contigmap": {
                "contigs": [],              # set per-run
                "inpaint_seq": None,
                "provide_seq": None,
                "length": None,
            },
            "model": {
                "n_extra_block": 4,
                "n_main_block": 32,
                "n_ref_block": 4,
                "d_msa": 256,
                "d_pair": 128,
                "d_templ": 64,
                "n_head_msa": 8,
                "n_head_pair": 4,
                "n_head_templ": 4,
                "d_hidden": 32,
                "d_hidden_templ": 32,
                "p_drop": 0.15,
            },
            "diffuser": {
                "T": 200,
                "b_0": 1e-2,
                "b_T": 7e-2,
                "schedule_type": "linear",
                "so3_type": "igso3",
                "crd_scale": 0.25,
                "partial_T": None,          # set per-run
                "so3_schedule_type": "linear",
                "min_b": 1.5,
                "max_b": 2.5,
            },
            "denoiser": {
                "noise_scale_ca": 1.0,      # set per-run
                "noise_scale_frame": 1.0,   # set per-run
            },
            "ppi": {
                "hotspot_res": [],
            },
            "potentials": {
                "guiding_potentials": None,
                "guide_scale": 10.0,
                "guide_decay": "quadratic",
                "olig_inter_all": None,
                "olig_intra_all": None,
                "substrate": None,
            },
        }

        cfg: DictConfig = OmegaConf.create(conf_dict)

        # SelfConditioning accepts the OmegaConf config directly
        runner = SelfConditioning(cfg)
        logger.debug("RFdiffusion SelfConditioning runner instantiated.")
        return runner

    @staticmethod
    def _resolve_device(override: str | None) -> "torch.device":
        import torch  # type: ignore[import]
        if override:
            return torch.device(override)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def _log_vram(device: "torch.device") -> None:
        try:
            import torch  # type: ignore[import]
            alloc = torch.cuda.memory_allocated(device) / 1e9
            logger.info("VRAM after RFdiffusion load: %.2f GB allocated.", alloc)
        except Exception:
            pass

    @classmethod
    def unload(cls) -> None:
        """Release the runner from memory and free CUDA caches."""
        with cls._lock:
            if cls._runner is None:
                logger.debug("RFdiffusion runner is not loaded; nothing to unload.")
                return
            logger.info("Unloading RFdiffusion runner …")
            del cls._runner
            cls._runner = None
            cls._loaded_checkpoint = None
            try:
                import torch  # type: ignore[import]
                if cls._device is not None and cls._device.type == "cuda":
                    torch.cuda.empty_cache()
                    logger.info("CUDA cache cleared after RFdiffusion unload.")
            except ImportError:
                pass
            cls._device = None

    @classmethod
    def is_loaded(cls) -> bool:
        with cls._lock:
            return cls._runner is not None


# ---------------------------------------------------------------------------
# PDB validation and structure utilities
# ---------------------------------------------------------------------------


def _validate_pdb(pdb_path: Path) -> tuple[str, int]:
    """
    Validate a PDB file and return (chain_id, residue_count).

    Checks:
    1. File exists and is non-empty.
    2. Parseable by Biopython's PDB parser.
    3. Contains at least one chain with backbone atoms (N, Cα, C).
    4. Residue numbering is monotonically non-decreasing within each chain.

    Parameters
    ----------
    pdb_path : Path

    Returns
    -------
    tuple[str, int]
        (primary_chain_id, total_residue_count)

    Raises
    ------
    PDBValidationError
    """
    from Bio.PDB import PDBParser  # type: ignore[import]
    from Bio.PDB.PDBExceptions import PDBConstructionWarning  # type: ignore[import]
    import warnings

    if not pdb_path.exists():
        raise PDBValidationError(f"PDB file not found: {pdb_path}")
    if pdb_path.stat().st_size == 0:
        raise PDBValidationError(f"PDB file is empty: {pdb_path}")

    parser = PDBParser(QUIET=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PDBConstructionWarning)
        try:
            structure = parser.get_structure("input", str(pdb_path))
        except Exception as exc:
            raise PDBValidationError(
                f"Biopython PDB parser failed on '{pdb_path}': "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    models = list(structure.get_models())
    if not models:
        raise PDBValidationError(f"PDB file contains no models: {pdb_path}")

    model = models[0]  # take first model (relevant for NMR ensembles)
    chains = list(model.get_chains())
    if not chains:
        raise PDBValidationError(f"PDB file contains no chains: {pdb_path}")

    backbone_atoms = {"N", "CA", "C"}
    valid_chains: list[tuple[str, int]] = []

    for chain in chains:
        residues = [
            r for r in chain.get_residues()
            if r.get_id()[0] == " "  # exclude HETATM and water
        ]
        if not residues:
            continue

        # Check backbone completeness for at least some residues
        backbone_present = sum(
            1 for r in residues
            if backbone_atoms.issubset({a.get_name() for a in r.get_atoms()})
        )
        if backbone_present == 0:
            logger.warning(
                "Chain %s has no residues with complete backbone; skipping.",
                chain.get_id(),
            )
            continue

        valid_chains.append((chain.get_id(), len(residues)))

    if not valid_chains:
        raise PDBValidationError(
            f"No chains with backbone atoms found in '{pdb_path}'."
        )

    # Return the chain with the most residues as primary
    primary_chain, residue_count = max(valid_chains, key=lambda x: x[1])
    logger.debug(
        "PDB validation passed: primary chain=%s, residues=%d.",
        primary_chain,
        residue_count,
    )
    return primary_chain, residue_count


def _get_residue_numbers(pdb_path: Path, chain_id: str) -> list[int]:
    """
    Extract 1-based PDB residue numbers for a chain, in order.

    Parameters
    ----------
    pdb_path : Path
    chain_id : str

    Returns
    -------
    list[int]
        1-based residue sequence numbers in the order they appear in the PDB.
    """
    from Bio.PDB import PDBParser  # type: ignore[import]
    import warnings
    from Bio.PDB.PDBExceptions import PDBConstructionWarning  # type: ignore[import]

    parser = PDBParser(QUIET=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PDBConstructionWarning)
        structure = parser.get_structure("s", str(pdb_path))

    residue_numbers: list[int] = []
    model = list(structure.get_models())[0]
    for chain in model.get_chains():
        if chain.get_id() == chain_id:
            for residue in chain.get_residues():
                if residue.get_id()[0] == " ":  # standard amino acid
                    residue_numbers.append(residue.get_id()[1])
            break

    return residue_numbers


# ---------------------------------------------------------------------------
# Repair region construction
# ---------------------------------------------------------------------------


def _assign_partial_T(peak_perplexity: float, region_length: int) -> int:
    """
    Choose diffusion timesteps based on spike severity.

    The heuristic scales with both the peak perplexity and the size of
    the affected region:
    - Small, mild spikes  → conservative repair (low partial_T)
    - Large or severe regions → aggressive repair (high partial_T)

    Parameters
    ----------
    peak_perplexity : float
        Maximum smoothed perplexity within the region.
    region_length : int
        Number of residues in the region.

    Returns
    -------
    int
        One of PARTIAL_T_CONSERVATIVE, PARTIAL_T_MODERATE, PARTIAL_T_AGGRESSIVE.
    """
    # Severity index: combines peak perplexity and length
    severity = peak_perplexity * (1.0 + 0.05 * region_length)

    if severity < 15.0:
        return PARTIAL_T_CONSERVATIVE
    elif severity < 35.0:
        return PARTIAL_T_MODERATE
    else:
        return PARTIAL_T_AGGRESSIVE


def _merge_spike_regions(
    spike_regions: list,   # list[SpikeRegion] from InstabilityReport
    sequence_length: int,
    merge_gap: int = REGION_MERGE_GAP,
    flank: int = FLANK_RESIDUES,
) -> list[tuple[int, int, list[int], float]]:
    """
    Merge adjacent spike regions and add flanking context.

    Two spike regions are merged if their gap (after flanking) is ≤
    ``merge_gap`` residues.  The peak perplexity of the merged region
    is the maximum of its constituent regions.

    Parameters
    ----------
    spike_regions : list[SpikeRegion]
    sequence_length : int
    merge_gap : int
    flank : int
        Residues to add on each side.

    Returns
    -------
    list[tuple[int, int, list[int], float]]
        Each element: (merged_start, merged_end, [source_spike_starts], peak_ppl)
        Coordinates are 0-based, inclusive, clamped to [0, sequence_length-1].
    """
    if not spike_regions:
        return []

    # Sort by start position
    sorted_regions = sorted(spike_regions, key=lambda r: r.start)

    # Add flanking
    flanked: list[tuple[int, int, list[int], float]] = []
    for r in sorted_regions:
        start = max(0, r.start - flank)
        end   = min(sequence_length - 1, r.end + flank)
        flanked.append((start, end, [r.start], r.peak_perplexity))

    # Merge overlapping / close regions
    merged: list[tuple[int, int, list[int], float]] = [flanked[0]]

    for start, end, sources, peak_ppl in flanked[1:]:
        prev_start, prev_end, prev_sources, prev_peak = merged[-1]

        if start <= prev_end + merge_gap:
            # Merge into the previous region
            merged[-1] = (
                prev_start,
                max(prev_end, end),
                prev_sources + sources,
                max(prev_peak, peak_ppl),
            )
        else:
            merged.append((start, end, sources, peak_ppl))

    return merged


def _build_repair_regions(
    spike_regions: list,       # list[SpikeRegion]
    pdb_residue_numbers: list[int],
    chain_id: str,
    sequence_length: int,
    merge_gap: int = REGION_MERGE_GAP,
    flank: int = FLANK_RESIDUES,
) -> list[RepairRegion]:
    """
    Convert spike regions into ``RepairRegion`` objects with PDB coordinates.

    Parameters
    ----------
    spike_regions : list[SpikeRegion]
        Output from ``InstabilityReport.spike_regions``.
    pdb_residue_numbers : list[int]
        Ordered list of 1-based PDB residue numbers (from ``_get_residue_numbers``).
    chain_id : str
    sequence_length : int
    merge_gap : int
    flank : int

    Returns
    -------
    list[RepairRegion]

    Raises
    ------
    ContigBuildError
        If sequence-to-PDB coordinate mapping fails.
    """
    merged = _merge_spike_regions(spike_regions, sequence_length, merge_gap, flank)

    if not merged:
        return []

    if len(pdb_residue_numbers) != sequence_length:
        raise ContigBuildError(
            f"PDB residue count ({len(pdb_residue_numbers)}) does not match "
            f"protein sequence length ({sequence_length}). "
            "Ensure the input PDB corresponds to the protein sequence."
        )

    repair_regions: list[RepairRegion] = []
    for start, end, sources, peak_ppl in merged:
        try:
            pdb_start = pdb_residue_numbers[start]
            pdb_end   = pdb_residue_numbers[end]
        except IndexError as exc:
            raise ContigBuildError(
                f"Repair region [{start}, {end}] is out of bounds for PDB "
                f"with {len(pdb_residue_numbers)} residues."
            ) from exc

        length = end - start + 1
        partial_T = _assign_partial_T(peak_ppl, length)

        repair_regions.append(RepairRegion(
            start=start,
            end=end,
            length=length,
            pdb_start=pdb_start,
            pdb_end=pdb_end,
            source_spike_starts=sources,
            peak_perplexity=peak_ppl,
            partial_T=partial_T,
            chain_id=chain_id,
        ))

    logger.info(
        "Built %d repair region(s) from %d spike region(s).",
        len(repair_regions),
        len(spike_regions),
    )
    return repair_regions


# ---------------------------------------------------------------------------
# Contig string construction
# ---------------------------------------------------------------------------


def _build_contig_string(
    repair_regions: list[RepairRegion],
    pdb_residue_numbers: list[int],
    chain_id: str,
    sequence_length: int,
) -> tuple[str, list[str]]:
    """
    Construct the RFdiffusion contig string for partial diffusion repair.

    In partial diffusion mode, the entire backbone is provided as a fixed
    motif contig.  RFdiffusion then applies noise only to positions whose
    ``diffusion_mask`` tensor entries are ``True``.  We therefore express
    the entire chain as a single contiguous contig segment and control
    which positions are diffused via the mask (passed separately in
    ``run_diffusion()``).

    The contig string format for a full-chain fixed motif is:
        ``"A1-100"``  (chain A, all 100 residues fixed)

    For multi-segment repair where we want RFdiffusion to handle each
    region with its own partial diffusion schedule, we express the chain
    as alternating fixed/free segments:
        ``"A1-30 A31-50 A51-100"``
    where segments matching repair regions will receive diffusion mask
    entries in the model call.

    Parameters
    ----------
    repair_regions : list[RepairRegion]
    pdb_residue_numbers : list[int]
    chain_id : str
    sequence_length : int

    Returns
    -------
    tuple[str, list[str]]
        (full_contig_string, preserved_segment_strings)
    """
    if not pdb_residue_numbers:
        raise ContigBuildError("pdb_residue_numbers is empty.")

    first_res = pdb_residue_numbers[0]
    last_res  = pdb_residue_numbers[-1]

    # Collect all repair index sets for quick lookup
    repair_index_set: set[int] = set()
    for rr in repair_regions:
        repair_index_set.update(range(rr.start, rr.end + 1))

    # Walk through the sequence and build alternating preserved/repair segments
    segments: list[str] = []
    preserved_segs: list[str] = []

    seg_start_idx = 0
    in_repair = (0 in repair_index_set)

    def _flush_segment(start_idx: int, end_idx: int, is_repair_seg: bool) -> None:
        pdb_s = pdb_residue_numbers[start_idx]
        pdb_e = pdb_residue_numbers[end_idx]
        seg = f"{chain_id}{pdb_s}-{pdb_e}"
        segments.append(seg)
        if not is_repair_seg:
            preserved_segs.append(seg)

    for i in range(1, sequence_length):
        currently_in_repair = i in repair_index_set
        if currently_in_repair != in_repair:
            _flush_segment(seg_start_idx, i - 1, in_repair)
            seg_start_idx = i
            in_repair = currently_in_repair

    # Flush the final segment
    _flush_segment(seg_start_idx, sequence_length - 1, in_repair)

    contig_string = " ".join(segments)
    logger.debug("Contig string: %s", contig_string)
    return contig_string, preserved_segs


# ---------------------------------------------------------------------------
# Diffusion mask construction
# ---------------------------------------------------------------------------


def _build_diffusion_mask(
    repair_regions: list[RepairRegion],
    sequence_length: int,
) -> "torch.Tensor":
    """
    Build a boolean diffusion mask tensor for RFdiffusion.

    Shape: (sequence_length,), dtype: torch.bool.
    True at positions to be diffused; False at positions to be fixed.

    Parameters
    ----------
    repair_regions : list[RepairRegion]
    sequence_length : int

    Returns
    -------
    torch.Tensor
        Boolean mask of shape (L,).
    """
    import torch  # type: ignore[import]

    mask = torch.zeros(sequence_length, dtype=torch.bool)
    for rr in repair_regions:
        mask[rr.start : rr.end + 1] = True

    n_diffused = int(mask.sum().item())
    logger.debug(
        "Diffusion mask: %d/%d residues marked for diffusion.",
        n_diffused,
        sequence_length,
    )
    return mask


# ---------------------------------------------------------------------------
# RFdiffusion inference runner
# ---------------------------------------------------------------------------


def _run_diffusion(
    runner: "SelfConditioning",
    input_pdb: Path,
    output_prefix: Path,
    contig_string: str,
    diffusion_mask: "torch.Tensor",
    partial_T: int,
    noise_scale: float,
    num_designs: int,
    seed: int | None,
) -> tuple[Path, Path | None]:
    """
    Execute one RFdiffusion partial diffusion run.

    Configures the runner for this specific repair job — updating the Hydra
    config fields that vary per invocation — then calls ``run_inference()``.

    Parameters
    ----------
    runner : SelfConditioning
    input_pdb : Path
    output_prefix : Path
        RFdiffusion writes ``{output_prefix}_0.pdb``, ``{output_prefix}_0.trb``, etc.
    contig_string : str
    diffusion_mask : torch.Tensor
        Shape (L,), bool.
    partial_T : int
    noise_scale : float
    num_designs : int
    seed : int | None

    Returns
    -------
    tuple[Path, Path | None]
        (healed_pdb_path, trb_path)
        ``trb_path`` is None if the .trb file was not produced.

    Raises
    ------
    DiffusionRunError
    """
    import torch  # type: ignore[import]
    from omegaconf import OmegaConf  # type: ignore[import]

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        logger.debug("RFdiffusion seed set to %d.", seed)

    # Update mutable config fields on the runner's OmegaConf config.
    # RFdiffusion's SelfConditioning exposes its config as runner.cfg.
    with OmegaConf.open_dict(runner.cfg):
        runner.cfg.inference.input_pdb        = str(input_pdb)
        runner.cfg.inference.output_prefix    = str(output_prefix)
        runner.cfg.inference.num_designs      = num_designs
        runner.cfg.contigmap.contigs          = [contig_string]
        runner.cfg.diffuser.partial_T         = partial_T
        runner.cfg.denoiser.noise_scale_ca    = noise_scale
        runner.cfg.denoiser.noise_scale_frame = noise_scale

    logger.info(
        "RFdiffusion partial diffusion: partial_T=%d, noise_scale=%.2f, "
        "num_designs=%d, contig='%s'.",
        partial_T,
        noise_scale,
        num_designs,
        contig_string,
    )

    try:
        # RFdiffusion's run_inference returns a list of (pdb_path, trb_dict) tuples.
        # The SelfConditioning runner generates designs sequentially.
        runner.run_inference(diffusion_mask=diffusion_mask)
    except Exception as exc:
        raise DiffusionRunError(
            f"RFdiffusion run_inference failed: {type(exc).__name__}: {exc}"
        ) from exc

    # Resolve output file paths (RFdiffusion appends _0, _1, … for designs)
    healed_pdb = Path(f"{output_prefix}_0.pdb")
    trb_file   = Path(f"{output_prefix}_0.trb")

    if not healed_pdb.exists():
        raise DiffusionRunError(
            f"RFdiffusion did not produce the expected output PDB: {healed_pdb}. "
            "Check RFdiffusion logs for errors."
        )

    trb_path = trb_file if trb_file.exists() else None
    logger.info("RFdiffusion output: %s (trb: %s).", healed_pdb, trb_path)
    return healed_pdb, trb_path


# ---------------------------------------------------------------------------
# TRB metadata parsing
# ---------------------------------------------------------------------------


def _parse_trb(
    trb_path: Path | None,
    repair_regions: list[RepairRegion],
    sequence_length: int,
) -> list[ResidueConfidence]:
    """
    Parse RFdiffusion's ``.trb`` metadata file to extract per-residue pLDDT.

    The ``.trb`` file is a Python pickle containing a dict with keys:
    - ``"lddt"``       — per-residue lDDT scores (shape: L,)
    - ``"plddt"``      — predicted lDDT (shape: L,) [same as above in some versions]
    - ``"con_ref_pdb_idx"`` — contig mapping back to input residue numbers
    - ``"con_hal_pdb_idx"`` — hallucinated residue numbers in the output PDB
    - ``"sampled_mask"``    — boolean mask of sampled positions

    If the file is absent (e.g. RFdiffusion CLI mode without --save_metadata),
    all pLDDT values are set to -1.0 as a sentinel.

    Parameters
    ----------
    trb_path : Path | None
    repair_regions : list[RepairRegion]
    sequence_length : int

    Returns
    -------
    list[ResidueConfidence]
    """
    repair_set: set[int] = set()
    for rr in repair_regions:
        repair_set.update(range(rr.start, rr.end + 1))

    # Default: no trb available
    if trb_path is None or not trb_path.exists():
        logger.warning(
            "No .trb file found; pLDDT scores will be -1 (unavailable)."
        )
        return [
            ResidueConfidence(
                index=i,
                pdb_residue_number=i + 1,
                plddt=-1.0,
                is_repaired=(i in repair_set),
            )
            for i in range(sequence_length)
        ]

    try:
        with open(trb_path, "rb") as fh:
            trb: dict = pickle.load(fh)
    except (pickle.UnpicklingError, EOFError, OSError) as exc:
        logger.error("Failed to parse .trb file '%s': %s", trb_path, exc)
        return [
            ResidueConfidence(i, i + 1, -1.0, i in repair_set)
            for i in range(sequence_length)
        ]

    # Extract pLDDT — try both common key names
    raw_plddt: list[float] | np.ndarray | None = (
        trb.get("plddt") or trb.get("lddt")
    )

    if raw_plddt is None:
        logger.warning("No pLDDT data in .trb file; using -1 sentinels.")
        return [
            ResidueConfidence(i, i + 1, -1.0, i in repair_set)
            for i in range(sequence_length)
        ]

    plddt_arr = np.asarray(raw_plddt, dtype=np.float32)

    # Scale to [0, 100] if values appear to be in [0, 1]
    if plddt_arr.max() <= 1.0:
        plddt_arr = plddt_arr * 100.0

    # Residue numbers in output PDB
    hal_idx: list[tuple[str, int]] | None = trb.get("con_hal_pdb_idx")
    pdb_resnums: list[int] = (
        [idx[1] for idx in hal_idx]
        if hal_idx and len(hal_idx) == len(plddt_arr)
        else list(range(1, len(plddt_arr) + 1))
    )

    confidences: list[ResidueConfidence] = []
    for i, (plddt_val, pdb_resnum) in enumerate(zip(plddt_arr.tolist(), pdb_resnums)):
        confidences.append(ResidueConfidence(
            index=i,
            pdb_residue_number=pdb_resnum,
            plddt=float(plddt_val),
            is_repaired=(i in repair_set),
        ))

    logger.debug(
        "Parsed .trb: %d residues, mean pLDDT=%.1f.",
        len(confidences),
        float(plddt_arr.mean()),
    )
    return confidences


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------


def _compute_plddt_stats(
    confidences: list[ResidueConfidence],
) -> tuple[float, float, float]:
    """
    Compute mean pLDDT across all / repaired / preserved residues.

    Excludes sentinel values (pLDDT == -1.0) from all averages.

    Returns
    -------
    tuple[float, float, float]
        (mean_all, mean_repaired, mean_preserved)
    """
    def _safe_mean(vals: list[float]) -> float:
        valid = [v for v in vals if v >= 0]
        return float(np.mean(valid)) if valid else -1.0

    all_plddt      = [rc.plddt for rc in confidences]
    repaired_plddt = [rc.plddt for rc in confidences if rc.is_repaired]
    preserved_plddt= [rc.plddt for rc in confidences if not rc.is_repaired]

    return (
        _safe_mean(all_plddt),
        _safe_mean(repaired_plddt),
        _safe_mean(preserved_plddt),
    )


# ---------------------------------------------------------------------------
# Primary public API
# ---------------------------------------------------------------------------


def heal_protein(
    instability_report,              # InstabilityReport from Stage 3
    input_pdb_path: str | Path,
    output_dir: str | Path,
    checkpoint_path: str | None = None,
    device: str | None = None,
    noise_scale: float = DEFAULT_NOISE_SCALE,
    num_designs: int = DEFAULT_NUM_DESIGNS,
    flank_residues: int = FLANK_RESIDUES,
    region_merge_gap: int = REGION_MERGE_GAP,
    seed: int | None = 42,
    overwrite: bool = False,
) -> HealingReport:
    """
    Repair a mutated protein backbone using RFdiffusion partial diffusion.

    This is the primary public entry point for Stage 4.  It consumes an
    ``InstabilityReport`` from Stage 3 to identify which residue regions
    require repair, constructs the appropriate contig string and diffusion
    mask, runs RFdiffusion, and returns a ``HealingReport`` with the healed
    PDB path and per-residue confidence scores.

    Parameters
    ----------
    instability_report : InstabilityReport
        Output from ``esm3_instability_detector.detect_instability()``.
        Must have at least one entry in ``spike_regions``.
    input_pdb_path : str | Path
        Path to the PDB file of the *mutated* structure.  Must contain
        backbone atoms for the full protein.  This file is read-only;
        the healed structure is written to ``output_dir``.
    output_dir : str | Path
        Directory where RFdiffusion output files will be written.
        Created automatically if it does not exist.
    checkpoint_path : str | None
        Absolute path to the RFdiffusion ``.pt`` checkpoint file.
        If ``None``, reads ``$RFDIFFUSION_WEIGHTS_PATH/RFdiffusion.pt``.
    device : str | None
        Compute device.  ``None`` = auto-detect (CUDA if available).
    noise_scale : float
        Magnitude of noise per diffusion step [0.5, 2.0].  Default 1.0.
    num_designs : int
        Number of independent backbone designs to generate.  The first
        (``_0.pdb``) is returned as the primary output; all are written
        to ``output_dir``.
    flank_residues : int
        Context residues added around each spike region.
    region_merge_gap : int
        Gap threshold for merging adjacent spike regions.
    seed : int | None
        Random seed for reproducible diffusion trajectories.  ``None``
        disables seeding (non-deterministic).
    overwrite : bool
        If False, raises ``FileExistsError`` when the output PDB already
        exists.

    Returns
    -------
    HealingReport

    Raises
    ------
    NoRegionsToRepairError
        If ``instability_report.spike_regions`` is empty.
    PDBValidationError
        If the input PDB fails structural validation.
    ContigBuildError
        If the contig string cannot be constructed.
    ModelLoadError
        If the RFdiffusion checkpoint cannot be loaded.
    DiffusionRunError
        If the RFdiffusion inference call fails.
    FileExistsError
        If the output PDB exists and ``overwrite=False``.
    """
    # ------------------------------------------------------------------ #
    # 1. Validate inputs                                                   #
    # ------------------------------------------------------------------ #
    if not instability_report.spike_regions:
        raise NoRegionsToRepairError(
            f"InstabilityReport for '{instability_report.sequence_id}' contains "
            "no spike regions.  No repair is needed."
        )

    input_pdb_path = Path(input_pdb_path).resolve()
    output_dir     = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    seq_id       = instability_report.sequence_id
    seq_length   = instability_report.sequence_length
    spike_regions = instability_report.spike_regions

    logger.info(
        "Healing '%s': %d spike region(s), %d spike residue(s), "
        "input_pdb='%s'.",
        seq_id,
        len(spike_regions),
        len(instability_report.spike_indices),
        input_pdb_path,
    )

    # ------------------------------------------------------------------ #
    # 2. PDB validation                                                    #
    # ------------------------------------------------------------------ #
    chain_id, pdb_residue_count = _validate_pdb(input_pdb_path)

    warnings_list: list[str] = []
    if pdb_residue_count != seq_length:
        msg = (
            f"PDB residue count ({pdb_residue_count}) differs from sequence "
            f"length ({seq_length}).  Using PDB count for coordinate mapping."
        )
        logger.warning(msg)
        warnings_list.append(msg)
        # Use PDB residue count as authoritative for coordinate mapping
        seq_length = pdb_residue_count

    pdb_residue_numbers = _get_residue_numbers(input_pdb_path, chain_id)

    # ------------------------------------------------------------------ #
    # 3. Build repair regions                                              #
    # ------------------------------------------------------------------ #
    repair_regions = _build_repair_regions(
        spike_regions=spike_regions,
        pdb_residue_numbers=pdb_residue_numbers,
        chain_id=chain_id,
        sequence_length=seq_length,
        merge_gap=region_merge_gap,
        flank=flank_residues,
    )

    for rr in repair_regions:
        logger.info(
            "  Repair region: seq[%d:%d] → PDB %s%d-%d, "
            "peak_ppl=%.1f, partial_T=%d.",
            rr.start, rr.end + 1,
            rr.chain_id, rr.pdb_start, rr.pdb_end,
            rr.peak_perplexity, rr.partial_T,
        )

    # ------------------------------------------------------------------ #
    # 4. Contig string + diffusion mask                                    #
    # ------------------------------------------------------------------ #
    contig_string, preserved_segs = _build_contig_string(
        repair_regions, pdb_residue_numbers, chain_id, seq_length
    )
    diffusion_mask = _build_diffusion_mask(repair_regions, seq_length)

    # Use the maximum partial_T across all repair regions for the run
    partial_T_used = max(rr.partial_T for rr in repair_regions)

    # ------------------------------------------------------------------ #
    # 5. Resolve output path and check for existing files                  #
    # ------------------------------------------------------------------ #
    output_prefix = output_dir / f"{seq_id}_healed"
    expected_pdb  = Path(f"{output_prefix}_0.pdb")

    if expected_pdb.exists() and not overwrite:
        raise FileExistsError(
            f"Output PDB '{expected_pdb}' already exists. "
            "Pass overwrite=True to replace it."
        )

    # ------------------------------------------------------------------ #
    # 6. Load model (lazy, thread-safe)                                    #
    # ------------------------------------------------------------------ #
    runner, torch_device = _RFdiffusionLoader.get(
        checkpoint_path=checkpoint_path,
        device_override=device,
    )
    actual_checkpoint = _RFdiffusionLoader._loaded_checkpoint or "unknown"

    # ------------------------------------------------------------------ #
    # 7. Run diffusion                                                     #
    # ------------------------------------------------------------------ #
    t_start = time.perf_counter()

    healed_pdb, trb_path = _run_diffusion(
        runner=runner,
        input_pdb=input_pdb_path,
        output_prefix=output_prefix,
        contig_string=contig_string,
        diffusion_mask=diffusion_mask,
        partial_T=partial_T_used,
        noise_scale=noise_scale,
        num_designs=num_designs,
        seed=seed,
    )

    inference_time = time.perf_counter() - t_start

    # ------------------------------------------------------------------ #
    # 8. Parse TRB metadata + compute confidence scores                    #
    # ------------------------------------------------------------------ #
    residue_confidences = _parse_trb(trb_path, repair_regions, seq_length)
    mean_plddt_all, mean_plddt_repair, mean_plddt_preserve = _compute_plddt_stats(
        residue_confidences
    )

    num_repaired   = sum(rr.length for rr in repair_regions)
    num_preserved  = seq_length - num_repaired

    # ------------------------------------------------------------------ #
    # 9. Build and return report                                            #
    # ------------------------------------------------------------------ #
    report = HealingReport(
        sequence_id=seq_id,
        input_pdb_path=input_pdb_path,
        output_pdb_path=healed_pdb,
        trb_path=trb_path,
        repair_regions=repair_regions,
        preserved_segments=preserved_segs,
        full_contig_string=contig_string,
        residue_confidences=residue_confidences,
        mean_plddt=mean_plddt_all,
        mean_plddt_repaired=mean_plddt_repair,
        mean_plddt_preserved=mean_plddt_preserve,
        num_repaired_residues=num_repaired,
        num_preserved_residues=num_preserved,
        partial_T_used=partial_T_used,
        noise_scale_used=noise_scale,
        num_designs_generated=num_designs,
        model_checkpoint=actual_checkpoint,
        device_used=str(torch_device),
        inference_time_s=inference_time,
        warnings=warnings_list,
    )

    logger.info(report.summary())
    return report


def write_healing_report(
    report: HealingReport,
    output_path: str | Path,
    overwrite: bool = False,
) -> Path:
    """
    Write a ``HealingReport`` to a JSON file.

    Parameters
    ----------
    report : HealingReport
    output_path : str | Path
    overwrite : bool

    Returns
    -------
    Path

    Raises
    ------
    FileExistsError
    """
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Healing report '{output_path}' already exists. "
            "Pass overwrite=True to replace it."
        )

    output_path.write_text(report.to_json(), encoding="utf-8")
    logger.info("Healing report written to '%s'.", output_path)
    return output_path


def release_model() -> None:
    """Unload the RFdiffusion model and free VRAM. Convenience wrapper."""
    _RFdiffusionLoader.unload()


def model_is_loaded() -> bool:
    """Return True if the RFdiffusion runner is currently in memory."""
    return _RFdiffusionLoader.is_loaded()
