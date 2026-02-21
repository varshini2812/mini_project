"""
proteinmpnn_inverse_fold.py
===========================
GeneForge Pipeline — Stage 5: Sequence Design via ProteinMPNN Inverse Folding

Scientific Background
---------------------
ProteinMPNN (Dauparas et al., *Science* 2022) is a message-passing neural
network trained to solve the *inverse folding* problem: given a protein
backbone geometry, predict the amino acid sequence most likely to fold into
that backbone.

The model operates on backbone Cα–N–C–O coordinates extracted from a PDB
file and encodes them as a k-nearest-neighbour graph (k=48 by default) over
residue pairs.  An encoder processes the geometry into edge and node
embeddings; an autoregressive decoder then samples sequences conditional on
those embeddings, one residue at a time.

Key design properties exploited in this module:

**Fixed vs designed positions.**  ProteinMPNN supports a binary ``chain_M``
mask (shape: ``(1, L)``).  Residues with ``chain_M = 0.0`` are "fixed" —
their identity is revealed to the encoder but their positions are not
sampled during decoding.  In GeneForge's repair context:

  - *Designed positions* (``chain_M = 1.0``): residue ranges that were
    re-built by RFdiffusion (from ``HealingReport.repair_regions``).  The
    model is free to choose any amino acid here.
  - *Fixed positions* (``chain_M = 0.0``): residues from the original
    structure that RFdiffusion preserved.  We constrain these to their
    original identity via a one-hot ``bias_by_res`` tensor (large logit
    added to the correct AA), which forces the decoder to reproduce the
    original sequence at these sites without hard-clamping it.

**Temperature sampling.** The decoder's softmax temperature controls the
diversity–quality trade-off:

  - ``T = 0.1`` (default): greedy-ish — close to maximum-likelihood
    sequence; highest mean score, lowest diversity.
  - ``T ≥ 0.5``: more diverse; useful for exploring sequence space at
    designed positions.

**Scoring.**  For each generated sequence, the module computes:

  - ``sequence_score``: mean negative log-probability over *designed*
    positions only — lower is better (higher confidence).  Primary ranking key.
  - ``global_score``: mean negative log-probability over *all* positions.
  - ``recovery_rate``: fraction of designed positions where the generated AA
    matches the reference (input PDB) sequence.
  - ``per_residue_log_probs``: the full ``(L,)`` log-probability vector of
    the sampled amino acid at each position, preserved for downstream analysis.

**Multiple designs.**  ``num_seqs`` independent samples are drawn for each
temperature.  All sequences are pooled and returned ranked by
``sequence_score`` ascending (best = lowest = most confident).

Checkpoint Variants
-------------------
ProteinMPNN ships with several noise-level variants in
``vanilla_model_weights/``:

  ============ ===============================
  File          Description
  ============ ===============================
  v_48_002.pt  noise 0.02 Å (very precise)
  v_48_010.pt  noise 0.10 Å (moderate)
  v_48_020.pt  noise 0.20 Å (default, robust)
  v_48_030.pt  noise 0.30 Å (tolerant)
  ============ ===============================

Higher noise tolerance makes the model more robust to imperfect backbone
geometries — important for RFdiffusion outputs where the repaired region
may have minor coordinate artefacts.  **v_48_020.pt is the recommended
default** for post-RFdiffusion sequence design.

PDB Coordinate Extraction
--------------------------
ProteinMPNN requires backbone coordinates as a ``(L, 4, 3)`` float tensor
in the order **N, Cα, C, O**.  Side-chain atoms are ignored.  This module
uses Biopython's ``PDB`` sub-package to extract and validate these
coordinates, with HETATM and water residues excluded.

Missing atoms are handled by two-stage imputation:
  1. If a single backbone atom is missing, reconstruct from centroid of
     present atoms at that residue.
  2. If all atoms are missing, fill with zeros and let the validity mask
     signal ProteinMPNN to skip that position.

Environment Variables
---------------------
  PROTEINMPNN_WEIGHTS_PATH  — directory containing the ``.pt`` checkpoint
                              files.  Required if no ``checkpoint_path``
                              is passed to ``run_inverse_folding()``.

Dependencies
------------
  biopython >= 1.83  — PDB coordinate extraction
  torch     >= 2.1   — tensor computation and model inference
  numpy     >= 1.26  — array operations
  Python    >= 3.12

  Runtime (not install-time):
  proteinmpnn           — ProteinMPNN model code
                          (https://github.com/dauparas/ProteinMPNN)
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, NamedTuple

import numpy as np

# ---------------------------------------------------------------------------
# Module logger
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment variable key
# ---------------------------------------------------------------------------

ENV_WEIGHTS_PATH: Final[str] = "PROTEINMPNN_WEIGHTS_PATH"

#: Default checkpoint filename inside the weights directory.
DEFAULT_CHECKPOINT_NAME: Final[str] = "v_48_020.pt"

# ---------------------------------------------------------------------------
# ProteinMPNN model constants
# ---------------------------------------------------------------------------

#: The 21-character amino acid alphabet used by ProteinMPNN.
#: Index 20 = X (unknown / mask token).
AA_ALPHABET: Final[str] = "ACDEFGHIKLMNPQRSTVWYX"

#: Number of letters in the ProteinMPNN alphabet.
NUM_LETTERS: Final[int] = 21

#: ProteinMPNN v_48 architecture hyperparameters.
MODEL_NODE_FEATURES:      Final[int]   = 128
MODEL_EDGE_FEATURES:      Final[int]   = 128
MODEL_HIDDEN_DIM:         Final[int]   = 128
MODEL_NUM_ENCODER_LAYERS: Final[int]   = 3
MODEL_NUM_DECODER_LAYERS: Final[int]   = 3
MODEL_K_NEIGHBORS:        Final[int]   = 48
MODEL_AUGMENT_EPS:        Final[float] = 0.0
MODEL_DROPOUT:            Final[float] = 0.0

#: Backbone atom names in the order ProteinMPNN expects.
BACKBONE_ATOMS: Final[tuple[str, ...]] = ("N", "CA", "C", "O")

#: Large logit bias applied to fixed positions to enforce identity.
#: 1e4 makes the fixed AA overwhelmingly probable without hard-clamping.
FIXED_POSITION_BIAS: Final[float] = 1e4

#: Default sampling temperature.
DEFAULT_TEMPERATURE: Final[float] = 0.1

#: Default number of independent sequences to sample per temperature.
DEFAULT_NUM_SEQS: Final[int] = 8

#: Default PDB chain identifier.
DEFAULT_CHAIN: Final[str] = "A"

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class InverseFoldError(Exception):
    """Base exception for all ProteinMPNN inverse folding failures."""


class ModelLoadError(InverseFoldError):
    """Raised when the ProteinMPNN checkpoint cannot be found or loaded."""


class PDBExtractionError(InverseFoldError):
    """Raised when backbone coordinates cannot be extracted from the PDB."""


class InferenceError(InverseFoldError):
    """Raised when the ProteinMPNN forward or sample pass fails."""


class InvalidInputError(InverseFoldError):
    """Raised when arguments to the public API fail validation."""


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------


class DesignedSequence(NamedTuple):
    """
    A single inverse-folded sequence candidate with associated scores.

    Attributes
    ----------
    rank : int
        1-based rank (1 = best by ``sequence_score``).
    sequence : str
        Full-length amino acid sequence (length L).
        Characters outside designed positions mirror the input PDB sequence
        (due to the fixed-position bias); designed positions carry the
        ProteinMPNN-chosen AA.
    designed_subsequence : str
        Amino acids *only* at designed (non-fixed) positions, concatenated
        in residue order.  Empty string if all positions were fixed.
    sequence_score : float
        Mean **negative** log-probability over designed positions.
        Lower = higher model confidence.  Primary ranking key.
    global_score : float
        Mean negative log-probability over *all* positions.
    recovery_rate : float
        Fraction of designed positions where the sampled AA matches the
        reference sequence read from the input PDB.  Range [0, 1].
    per_residue_log_probs : list[float]
        Log-probability of the *sampled* amino acid at each position.
        Length L.  Useful for identifying low-confidence residues.
    temperature : float
        Sampling temperature used for this sequence.
    sample_index : int
        0-based index of this sample within the same temperature run.

    Notes
    -----
    ``sequence_score`` is the primary ranking criterion because it focuses
    confidence assessment on the positions that were actually redesigned.
    A low global_score with a high sequence_score indicates confidence in
    fixed positions but uncertainty in the repair — a useful diagnostic.
    """

    rank: int
    sequence: str
    designed_subsequence: str
    sequence_score: float
    global_score: float
    recovery_rate: float
    per_residue_log_probs: list[float]
    temperature: float
    sample_index: int

    def to_fasta_record(self, sequence_id: str) -> str:
        """
        Return a FASTA-formatted string for this sequence.

        The header encodes rank, score, temperature, and recovery in a
        compact parseable format compatible with downstream tools.
        """
        header = (
            f">{sequence_id}|rank={self.rank}"
            f"|score={self.sequence_score:.4f}"
            f"|global_score={self.global_score:.4f}"
            f"|recovery={self.recovery_rate:.3f}"
            f"|T={self.temperature}"
            f"|sample={self.sample_index}"
        )
        # Wrap at 60 chars (standard FASTA line width)
        wrapped = "\n".join(
            self.sequence[i : i + 60]
            for i in range(0, len(self.sequence), 60)
        )
        return f"{header}\n{wrapped}"

    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "sequence": self.sequence,
            "designed_subsequence": self.designed_subsequence,
            "sequence_score": round(self.sequence_score, 6),
            "global_score": round(self.global_score, 6),
            "recovery_rate": round(self.recovery_rate, 4),
            "per_residue_log_probs": [round(v, 5) for v in self.per_residue_log_probs],
            "temperature": self.temperature,
            "sample_index": self.sample_index,
        }


@dataclass(slots=True)
class InverseFoldResult:
    """
    Complete result of a ProteinMPNN inverse folding run.

    This is the Stage 5 output, consumed by the final GeneForge assembly
    step (backtranslation / codon optimisation).

    Attributes
    ----------
    sequence_id : str
        Carried from the upstream ``HealingReport``.
    input_pdb_path : Path
        The RFdiffusion-healed backbone PDB used as input.
    sequences : list[DesignedSequence]
        All generated sequences ranked by ``sequence_score`` (best first).
    designed_positions : list[int]
        0-based residue indices that were free to design.
    fixed_positions : list[int]
        0-based residue indices held fixed (bias-constrained).
    num_designed_residues : int
    num_fixed_residues : int
    sequence_length : int
    reference_sequence : str
        Amino acid sequence read from the input PDB.
    temperatures_used : list[float]
    num_seqs_per_temperature : int
    model_checkpoint : str
    device_used : str
    inference_time_s : float
    warnings : list[str]
    """

    sequence_id: str
    input_pdb_path: Path
    sequences: list[DesignedSequence]
    designed_positions: list[int]
    fixed_positions: list[int]
    num_designed_residues: int
    num_fixed_residues: int
    sequence_length: int
    reference_sequence: str
    temperatures_used: list[float]
    num_seqs_per_temperature: int
    model_checkpoint: str
    device_used: str
    inference_time_s: float
    warnings: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Derived views
    # ------------------------------------------------------------------

    @property
    def best_sequence(self) -> DesignedSequence:
        """Top-ranked sequence (lowest sequence_score)."""
        if not self.sequences:
            raise InverseFoldError(
                "InverseFoldResult has no sequences — inference may have failed."
            )
        return self.sequences[0]

    @property
    def mean_best_score(self) -> float:
        """Mean sequence_score across all candidates."""
        if not self.sequences:
            return float("inf")
        return float(np.mean([s.sequence_score for s in self.sequences]))

    @property
    def score_std(self) -> float:
        """Standard deviation of sequence_score across all candidates."""
        if len(self.sequences) < 2:
            return 0.0
        return float(np.std([s.sequence_score for s in self.sequences]))

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_fasta(self, top_n: int | None = None) -> str:
        """
        Return a multi-record FASTA string for the top-N sequences.

        Parameters
        ----------
        top_n : int | None
            Number of sequences to include.  ``None`` = all.
        """
        seqs = self.sequences[:top_n] if top_n else self.sequences
        return "\n".join(s.to_fasta_record(self.sequence_id) for s in seqs)

    def write_fasta(
        self,
        output_path: str | Path,
        top_n: int | None = None,
        overwrite: bool = False,
    ) -> Path:
        """Write FASTA to disk and return the resolved path."""
        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists() and not overwrite:
            raise FileExistsError(
                f"FASTA output '{output_path}' already exists. "
                "Pass overwrite=True to replace it."
            )
        output_path.write_text(self.to_fasta(top_n), encoding="utf-8")
        logger.info("FASTA written to '%s'.", output_path)
        return output_path

    def to_dict(self) -> dict:
        return {
            "sequence_id": self.sequence_id,
            "input_pdb_path": str(self.input_pdb_path),
            "sequence_length": self.sequence_length,
            "num_designed_residues": self.num_designed_residues,
            "num_fixed_residues": self.num_fixed_residues,
            "designed_positions": self.designed_positions,
            "fixed_positions": self.fixed_positions,
            "reference_sequence": self.reference_sequence,
            "num_sequences": len(self.sequences),
            "temperatures_used": self.temperatures_used,
            "num_seqs_per_temperature": self.num_seqs_per_temperature,
            "best_sequence_score": round(self.best_sequence.sequence_score, 6),
            "best_global_score": round(self.best_sequence.global_score, 6),
            "best_recovery_rate": round(self.best_sequence.recovery_rate, 4),
            "mean_score": round(self.mean_best_score, 6),
            "score_std": round(self.score_std, 6),
            "model_checkpoint": self.model_checkpoint,
            "device_used": self.device_used,
            "inference_time_s": round(self.inference_time_s, 2),
            "sequences": [s.to_dict() for s in self.sequences],
            "warnings": self.warnings,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def summary(self) -> str:
        best = self.best_sequence
        return (
            f"[{self.sequence_id}] "
            f"seqs={len(self.sequences)}, "
            f"best_score={best.sequence_score:.4f}, "
            f"best_recovery={best.recovery_rate:.3f}, "
            f"designed={self.num_designed_residues}/{self.sequence_length}, "
            f"T={self.temperatures_used}, "
            f"t={self.inference_time_s:.1f}s"
        )


# ---------------------------------------------------------------------------
# ProteinMPNN model loader — lazy singleton, thread-safe
# ---------------------------------------------------------------------------


class _ProteinMPNNLoader:
    """
    Thread-safe lazy singleton for the ProteinMPNN model.

    The model is held in class state after first load and reused across
    calls — following the same pattern as ``_ESM3Loader`` (Stage 3) and
    ``_RFdiffusionLoader`` (Stage 4).
    """

    _lock: threading.Lock = threading.Lock()
    _model = None
    _loaded_checkpoint: str | None = None
    _device = None

    @classmethod
    def get(
        cls,
        checkpoint_path: str | None = None,
        device_override: str | None = None,
    ):
        """
        Return the loaded ProteinMPNN model and its device.

        Parameters
        ----------
        checkpoint_path : str | None
            Falls back to ``$PROTEINMPNN_WEIGHTS_PATH/v_48_020.pt``.
        device_override : str | None
            Force a specific device; ``None`` = auto-detect.

        Returns
        -------
        tuple[ProteinMPNN, torch.device]

        Raises
        ------
        ModelLoadError
        """
        resolved_ckpt = cls._resolve_checkpoint(checkpoint_path)

        with cls._lock:
            if cls._model is not None and cls._loaded_checkpoint == resolved_ckpt:
                logger.debug(
                    "ProteinMPNN cache hit: checkpoint='%s'.", resolved_ckpt
                )
                return cls._model, cls._device

            device = cls._resolve_device(device_override)
            logger.info(
                "Loading ProteinMPNN checkpoint '%s' onto %s …",
                resolved_ckpt, device,
            )
            t0 = time.perf_counter()

            try:
                model = cls._load_model(resolved_ckpt, device)
            except ImportError as exc:
                raise ModelLoadError(
                    "The 'protein_mpnn_utils' module is required but not found. "
                    "Clone ProteinMPNN from "
                    "https://github.com/dauparas/ProteinMPNN and ensure "
                    "'protein_mpnn_utils.py' is on sys.path.\n"
                    f"Original error: {exc}"
                ) from exc
            except FileNotFoundError as exc:
                raise ModelLoadError(
                    f"ProteinMPNN checkpoint not found at '{resolved_ckpt}'. "
                    f"Set {ENV_WEIGHTS_PATH!r} to the directory containing "
                    "your .pt checkpoint files.\n"
                    f"Original error: {exc}"
                ) from exc
            except Exception as exc:
                raise ModelLoadError(
                    f"Failed to load ProteinMPNN from '{resolved_ckpt}': "
                    f"{type(exc).__name__}: {exc}"
                ) from exc

            elapsed = time.perf_counter() - t0
            logger.info("ProteinMPNN loaded in %.1f s on %s.", elapsed, device)
            if device.type == "cuda":
                cls._log_vram(device)

            cls._model = model
            cls._loaded_checkpoint = resolved_ckpt
            cls._device = device

        return cls._model, cls._device

    @staticmethod
    def _resolve_checkpoint(explicit: str | None) -> str:
        if explicit:
            return str(Path(explicit).resolve())
        weights_dir = os.environ.get(ENV_WEIGHTS_PATH)
        if not weights_dir:
            raise ModelLoadError(
                f"No ProteinMPNN checkpoint provided and {ENV_WEIGHTS_PATH!r} "
                "is not set.  Set it to the directory containing your .pt files."
            )
        return str((Path(weights_dir) / DEFAULT_CHECKPOINT_NAME).resolve())

    @staticmethod
    def _load_model(checkpoint_path: str, device):
        import torch  # type: ignore[import]
        from protein_mpnn_utils import ProteinMPNN  # type: ignore[import]

        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(checkpoint_path)

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )

        k_neighbors = checkpoint.get("num_edges", MODEL_K_NEIGHBORS)

        model = ProteinMPNN(
            num_letters=NUM_LETTERS,
            node_features=MODEL_NODE_FEATURES,
            edge_features=MODEL_EDGE_FEATURES,
            hidden_dim=MODEL_HIDDEN_DIM,
            num_encoder_layers=MODEL_NUM_ENCODER_LAYERS,
            num_decoder_layers=MODEL_NUM_DECODER_LAYERS,
            augment_eps=MODEL_AUGMENT_EPS,
            k_neighbors=k_neighbors,
            dropout=MODEL_DROPOUT,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)
        model.eval()

        noise = checkpoint.get("noise_level", "unknown")
        logger.debug("ProteinMPNN: k=%d, noise_level=%s.", k_neighbors, noise)
        return model

    @staticmethod
    def _resolve_device(override: str | None):
        import torch  # type: ignore[import]
        if override:
            return torch.device(override)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def _log_vram(device) -> None:
        try:
            import torch  # type: ignore[import]
            alloc = torch.cuda.memory_allocated(device) / 1e9
            logger.info("VRAM after ProteinMPNN load: %.2f GB.", alloc)
        except Exception:
            pass

    @classmethod
    def unload(cls) -> None:
        """Release the model from memory and clear CUDA caches."""
        with cls._lock:
            if cls._model is None:
                logger.debug("ProteinMPNN not loaded; nothing to unload.")
                return
            logger.info("Unloading ProteinMPNN model …")
            del cls._model
            cls._model = None
            cls._loaded_checkpoint = None
            try:
                import torch  # type: ignore[import]
                if cls._device is not None and cls._device.type == "cuda":
                    torch.cuda.empty_cache()
                    logger.info("CUDA cache cleared.")
            except ImportError:
                pass
            cls._device = None

    @classmethod
    def is_loaded(cls) -> bool:
        with cls._lock:
            return cls._model is not None


# ---------------------------------------------------------------------------
# PDB coordinate extraction
# ---------------------------------------------------------------------------


def _extract_backbone_coords(
    pdb_path: Path,
    chain_id: str,
) -> tuple[np.ndarray, list[str], list[int]]:
    """
    Extract backbone atom coordinates from a PDB file for a single chain.

    Returns a float32 array of shape ``(L, 4, 3)`` in the atom order
    **N, Cα, C, O** as required by ProteinMPNN, alongside the one-letter
    amino acid sequence and PDB residue numbers.

    Missing individual backbone atoms are imputed from the centroid of the
    other three atoms at that residue.  If all backbone atoms for a residue
    are absent, coordinates are zeroed and the residue is noted in warnings.

    Parameters
    ----------
    pdb_path : Path
    chain_id : str

    Returns
    -------
    tuple[np.ndarray, list[str], list[int]]
        ``(coords, sequence, resnums)``
        - ``coords``:   float32 array, shape ``(L, 4, 3)``
        - ``sequence``: one-letter AA codes, length L
        - ``resnums``:  1-based PDB residue numbers, length L

    Raises
    ------
    PDBExtractionError
    """
    from Bio.PDB import PDBParser  # type: ignore[import]
    from Bio.PDB.Polypeptide import protein_letters_3to1  # type: ignore[import]
    from Bio.PDB.PDBExceptions import PDBConstructionWarning  # type: ignore[import]
    import warnings

    if not pdb_path.exists():
        raise PDBExtractionError(f"PDB file not found: {pdb_path}")

    parser = PDBParser(QUIET=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PDBConstructionWarning)
        try:
            structure = parser.get_structure("s", str(pdb_path))
        except Exception as exc:
            raise PDBExtractionError(
                f"Biopython failed to parse '{pdb_path}': "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    model = next(structure.get_models())
    target_chain = None
    for chain in model.get_chains():
        if chain.get_id() == chain_id:
            target_chain = chain
            break

    if target_chain is None:
        available = [c.get_id() for c in model.get_chains()]
        raise PDBExtractionError(
            f"Chain '{chain_id}' not found in '{pdb_path}'. "
            f"Available chains: {available}"
        )

    residues = [
        r for r in target_chain.get_residues()
        if r.get_id()[0] == " "   # exclude HETATM and water
    ]

    if not residues:
        raise PDBExtractionError(
            f"Chain '{chain_id}' in '{pdb_path}' contains no standard residues."
        )

    L = len(residues)
    coords   = np.zeros((L, 4, 3), dtype=np.float32)
    sequence: list[str] = []
    resnums:  list[int] = []
    all_missing: list[int] = []

    for i, residue in enumerate(residues):
        resname    = residue.get_resname().strip()
        one_letter = protein_letters_3to1.get(resname, "X")
        sequence.append(one_letter)
        resnums.append(residue.get_id()[1])

        atom_map = {
            a.get_name(): a.get_vector().get_array()
            for a in residue.get_atoms()
        }

        present: list[np.ndarray | None] = [
            atom_map.get(name) for name in BACKBONE_ATOMS
        ]
        missing_idx = [j for j, c in enumerate(present) if c is None]

        if len(missing_idx) == len(BACKBONE_ATOMS):
            all_missing.append(resnums[-1])
            logger.debug(
                "Residue %s%d: all backbone atoms missing.",
                chain_id, resnums[-1],
            )
        elif missing_idx:
            present_only = [c for c in present if c is not None]
            centroid     = np.mean(present_only, axis=0)
            for j in missing_idx:
                present[j] = centroid.copy()
                logger.debug(
                    "Residue %s%d: '%s' imputed from centroid.",
                    chain_id, resnums[-1], BACKBONE_ATOMS[j],
                )

        for j, coord in enumerate(present):
            if coord is not None:
                coords[i, j] = coord

    if all_missing:
        logger.warning(
            "Chain %s: %d residue(s) with all backbone atoms missing: %s.",
            chain_id, len(all_missing), all_missing,
        )

    logger.debug("Extracted backbone: chain=%s, L=%d.", chain_id, L)
    return coords, sequence, resnums


# ---------------------------------------------------------------------------
# Feature tensor construction
# ---------------------------------------------------------------------------


def _aa_to_int(aa: str) -> int:
    """Map a one-letter AA code to its ProteinMPNN index (0–20)."""
    idx = AA_ALPHABET.find(aa)
    return idx if idx >= 0 else 20


def _seq_to_tensor(sequence: list[str], device) -> "torch.Tensor":
    """Convert a list of AA codes to a ``(1, L)`` int64 tensor."""
    import torch  # type: ignore[import]
    ids = [_aa_to_int(aa) for aa in sequence]
    return torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)


def _build_chain_mask(
    sequence_length: int,
    designed_indices: set[int],
    device,
) -> "torch.Tensor":
    """
    Build the ProteinMPNN ``chain_M`` mask.

    Shape ``(1, L)``, float32.  1.0 = designed (free to sample),
    0.0 = fixed (identity revealed to encoder, not sampled).
    """
    import torch  # type: ignore[import]
    mask = torch.zeros(sequence_length, dtype=torch.float32, device=device)
    for idx in designed_indices:
        mask[idx] = 1.0
    return mask.unsqueeze(0)


def _build_validity_mask(coords: np.ndarray, device) -> "torch.Tensor":
    """
    Build the ProteinMPNN residue validity mask.

    A residue is valid if its Cα atom has non-zero coordinates.
    Shape ``(1, L)``, float32.
    """
    import torch  # type: ignore[import]
    ca_coords = coords[:, 1, :]  # (L, 3) — Cα is index 1
    valid = (np.abs(ca_coords).sum(axis=1) > 1e-6).astype(np.float32)
    return torch.tensor(valid, dtype=torch.float32, device=device).unsqueeze(0)


def _build_bias_by_res(
    reference_sequence: list[str],
    designed_indices: set[int],
    device,
    fixed_bias: float = FIXED_POSITION_BIAS,
) -> "torch.Tensor":
    """
    Build the ``bias_by_res`` tensor to soft-constrain fixed positions.

    At each fixed position, a large positive bias is applied to the logit
    of the reference amino acid, making it overwhelmingly likely to be
    selected without hard-clamping.  Designed positions receive zero bias.

    Shape ``(1, L, 21)``, float32.
    """
    import torch  # type: ignore[import]
    L    = len(reference_sequence)
    bias = torch.zeros((1, L, NUM_LETTERS), dtype=torch.float32, device=device)
    for i, aa in enumerate(reference_sequence):
        if i not in designed_indices:
            bias[0, i, _aa_to_int(aa)] = fixed_bias
    return bias


def _build_residue_idx(resnums: list[int], device) -> "torch.Tensor":
    """
    Build the ``residue_idx`` tensor from PDB residue numbers.

    Shape ``(1, L)``, int64.  ProteinMPNN uses this for relative position
    encoding in the kNN graph; actual PDB numbers are correct to use.
    """
    import torch  # type: ignore[import]
    return torch.tensor(resnums, dtype=torch.long, device=device).unsqueeze(0)


def _build_chain_encoding(sequence_length: int, chain_int: int, device) -> "torch.Tensor":
    """
    Build the ``chain_encoding_all`` tensor.

    All residues in a single-chain job share the same integer chain ID.
    Shape ``(1, L)``, int64.
    """
    import torch  # type: ignore[import]
    enc = torch.full(
        (sequence_length,), chain_int, dtype=torch.long, device=device
    )
    return enc.unsqueeze(0)


def _coords_to_tensor(coords: np.ndarray, device) -> "torch.Tensor":
    """Convert ``(L, 4, 3)`` numpy array to a ``(1, L, 4, 3)`` float32 tensor."""
    import torch  # type: ignore[import]
    return torch.tensor(coords, dtype=torch.float32, device=device).unsqueeze(0)


# ---------------------------------------------------------------------------
# Scoring utilities
# ---------------------------------------------------------------------------


def _compute_sequence_score(
    log_probs,   # torch.Tensor (1, L, 21)
    sampled_seq, # torch.Tensor (1, L) int64
    chain_M,     # torch.Tensor (1, L) float32
) -> tuple[float, float]:
    """
    Compute sequence_score and global_score from ProteinMPNN log-probs.

    ``sequence_score`` = mean negative log-prob over designed positions.
    ``global_score``   = mean negative log-prob over all positions.
    Both are **lower = better** (higher model confidence).
    """
    # Gather the log-prob of the sampled AA at each position
    sampled_log_p = log_probs.gather(
        dim=2, index=sampled_seq.unsqueeze(2)
    ).squeeze(2)  # (1, L)

    neg_log_p    = -sampled_log_p.squeeze(0)   # (L,)
    design_mask  = chain_M.squeeze(0)           # (L,)
    n_designed   = design_mask.sum().item()

    if n_designed > 0:
        seq_score = float(
            (neg_log_p * design_mask).sum().item() / n_designed
        )
    else:
        seq_score = float(neg_log_p.mean().item())

    global_score = float(neg_log_p.mean().item())
    return seq_score, global_score


def _compute_recovery(
    sampled_seq,   # torch.Tensor (1, L) int64
    reference_seq, # torch.Tensor (1, L) int64
    chain_M,       # torch.Tensor (1, L) float32
) -> float:
    """
    Compute recovery rate: fraction of designed positions matching reference.

    Returns 0.0 if no positions are designed.
    """
    design_mask = chain_M.squeeze(0).bool()  # (L,)
    n_designed  = design_mask.sum().item()

    if n_designed == 0:
        return 0.0

    match = (
        sampled_seq.squeeze(0)[design_mask]
        == reference_seq.squeeze(0)[design_mask]
    ).float().mean().item()

    return float(match)


def _tensor_to_sequence(seq_tensor) -> str:
    """Convert a ``(1, L)`` or ``(L,)`` int64 tensor to an AA string."""
    flat = seq_tensor.squeeze(0).tolist()
    return "".join(
        AA_ALPHABET[i] if 0 <= i < NUM_LETTERS else "X" for i in flat
    )


def _per_residue_log_probs(
    log_probs,   # torch.Tensor (1, L, 21)
    sampled_seq, # torch.Tensor (1, L) int64
) -> list[float]:
    """
    Extract the log-probability of the sampled AA at each position.

    Returns a list of length L.
    """
    gathered = log_probs.gather(
        dim=2, index=sampled_seq.unsqueeze(2)
    ).squeeze(2).squeeze(0)   # (L,)
    return [round(float(v), 6) for v in gathered.tolist()]


# ---------------------------------------------------------------------------
# Core sampling loop
# ---------------------------------------------------------------------------


def _run_sampling(
    model,
    coords_t,
    S_ref,
    mask,
    chain_M,
    residue_idx,
    chain_encoding,
    bias_by_res,
    designed_indices: set[int],
    reference_sequence: list[str],
    temperature: float,
    num_seqs: int,
    device,
    seed: int | None,
) -> list[dict]:
    """
    Sample ``num_seqs`` sequences from ProteinMPNN at a given temperature.

    Each iteration provides a fresh ``randn`` noise tensor to ensure
    independent samples from the autoregressive decoder.  When ``seed``
    is set, sample ``i`` uses ``seed + i`` so that results are
    reproducible yet non-identical across the batch.

    Parameters
    ----------
    model : ProteinMPNN
    coords_t : torch.Tensor  (1, L, 4, 3)
    S_ref : torch.Tensor     (1, L) int64 — reference sequence
    mask : torch.Tensor      (1, L) float32 — validity
    chain_M : torch.Tensor   (1, L) float32 — design mask
    residue_idx : torch.Tensor  (1, L) int64
    chain_encoding : torch.Tensor  (1, L) int64
    bias_by_res : torch.Tensor  (1, L, 21) float32
    designed_indices : set[int]
    reference_sequence : list[str]
    temperature : float
    num_seqs : int
    device : torch.device
    seed : int | None

    Returns
    -------
    list[dict]
        One dict per sample with keys: sequence, designed_subsequence,
        sequence_score, global_score, recovery_rate,
        per_residue_log_probs, temperature, sample_index.
    """
    import torch  # type: ignore[import]

    L       = coords_t.shape[1]
    results: list[dict] = []

    for i in range(num_seqs):
        if seed is not None:
            torch.manual_seed(seed + i)
        randn = torch.randn(1, L, device=device)

        try:
            with torch.no_grad():
                sample_out = model.sample(
                    X=coords_t,
                    randn=randn,
                    S_true=S_ref,
                    chain_mask=chain_M,
                    chain_encoding_all=chain_encoding,
                    residue_idx=residue_idx,
                    mask=mask,
                    temperature=temperature,
                    chain_M_pos=chain_M,
                    bias_by_res=bias_by_res,
                )
        except Exception as exc:
            raise InferenceError(
                f"ProteinMPNN sample() failed at T={temperature}, "
                f"sample {i}: {type(exc).__name__}: {exc}"
            ) from exc

        S_sampled = sample_out["S"]          # (1, L) int64
        log_probs = sample_out["log_probs"]  # (1, L, 21)

        seq_score, global_score = _compute_sequence_score(
            log_probs, S_sampled, chain_M
        )
        recovery  = _compute_recovery(S_sampled, S_ref, chain_M)
        per_res_lp = _per_residue_log_probs(log_probs, S_sampled)
        full_seq   = _tensor_to_sequence(S_sampled)
        designed_sub = "".join(
            full_seq[j] for j in sorted(designed_indices)
        )

        results.append({
            "sequence":              full_seq,
            "designed_subsequence":  designed_sub,
            "sequence_score":        seq_score,
            "global_score":          global_score,
            "recovery_rate":         recovery,
            "per_residue_log_probs": per_res_lp,
            "temperature":           temperature,
            "sample_index":          i,
        })

        logger.debug(
            "Sample T=%.2f #%d: score=%.4f, recovery=%.3f.",
            temperature, i, seq_score, recovery,
        )

    return results


# ---------------------------------------------------------------------------
# Primary public API
# ---------------------------------------------------------------------------


def run_inverse_folding(
    healing_report,
    output_dir: str | Path,
    checkpoint_path: str | None = None,
    chain_id: str = DEFAULT_CHAIN,
    temperatures: list[float] | None = None,
    num_seqs: int = DEFAULT_NUM_SEQS,
    device: str | None = None,
    seed: int | None = 42,
    overwrite: bool = False,
) -> InverseFoldResult:
    """
    Run ProteinMPNN inverse folding on a healed protein backbone.

    This is the primary public entry point for Stage 5.  It consumes a
    ``HealingReport`` from Stage 4 to determine which residue positions
    were redesigned by RFdiffusion (*designed* positions, free for
    ProteinMPNN to choose any AA) and which were preserved (*fixed*
    positions, soft-constrained to their original identity via a large
    logit bias).

    Parameters
    ----------
    healing_report : HealingReport
        Output of ``rfdiffusion_healer.heal_protein()``.  Provides:
        - ``output_pdb_path`` — healed backbone to design sequences for.
        - ``repair_regions``  — which residue index ranges were diffused.
        - ``sequence_id``     — carried forward into the result.
    output_dir : str | Path
        Directory for FASTA and JSON outputs.  Created if absent.
    checkpoint_path : str | None
        Path to a ``.pt`` checkpoint.  Falls back to
        ``$PROTEINMPNN_WEIGHTS_PATH/v_48_020.pt``.
    chain_id : str
        PDB chain to design.  Defaults to ``"A"``.
    temperatures : list[float] | None
        Sampling temperatures.  ``None`` → ``[0.1]``.
        Multiple temperatures are pooled before ranking.
    num_seqs : int
        Independent sequences to sample *per temperature*.
    device : str | None
        Compute device.  ``None`` = auto-detect.
    seed : int | None
        Base random seed for reproducibility.  Each sample uses
        ``seed + sample_index``.  ``None`` = non-deterministic.
    overwrite : bool
        Whether to overwrite existing output files.

    Returns
    -------
    InverseFoldResult
        All generated sequences ranked by ``sequence_score`` ascending.

    Raises
    ------
    InvalidInputError
        If argument validation fails.
    PDBExtractionError
        If backbone coordinates cannot be extracted.
    ModelLoadError
        If the checkpoint cannot be loaded.
    InferenceError
        If ProteinMPNN sampling fails.
    FileExistsError
        If outputs exist and ``overwrite=False``.
    """
    # ------------------------------------------------------------------ #
    # 1. Validate inputs                                                   #
    # ------------------------------------------------------------------ #
    if temperatures is None:
        temperatures = [DEFAULT_TEMPERATURE]
    if not temperatures:
        raise InvalidInputError("'temperatures' must be a non-empty list.")
    for t in temperatures:
        if not (0.0 < t <= 2.0):
            raise InvalidInputError(
                f"Temperature {t} is out of range (0, 2.0]."
            )
    if num_seqs < 1:
        raise InvalidInputError(f"'num_seqs' must be ≥ 1, got {num_seqs}.")

    pdb_path   = Path(healing_report.output_pdb_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    seq_id     = healing_report.sequence_id

    logger.info(
        "Inverse folding '%s': pdb='%s', T=%s, num_seqs=%d.",
        seq_id, pdb_path, temperatures, num_seqs,
    )

    # ------------------------------------------------------------------ #
    # 2. Resolve designed positions from repair regions                    #
    # ------------------------------------------------------------------ #
    designed_set: set[int] = set()
    for rr in healing_report.repair_regions:
        designed_set.update(range(rr.start, rr.end + 1))

    logger.info(
        "Designed positions: %d residue(s) across %d repair region(s).",
        len(designed_set), len(healing_report.repair_regions),
    )

    # ------------------------------------------------------------------ #
    # 3. Extract backbone coordinates                                      #
    # ------------------------------------------------------------------ #
    coords, pdb_sequence, resnums = _extract_backbone_coords(pdb_path, chain_id)
    L = len(pdb_sequence)

    warnings_list: list[str] = []

    # Clamp designed indices to valid range
    out_of_range = {i for i in designed_set if not (0 <= i < L)}
    if out_of_range:
        msg = (
            f"Dropping {len(out_of_range)} designed index/indices outside "
            f"[0, {L}): {sorted(out_of_range)}"
        )
        logger.warning(msg)
        warnings_list.append(msg)
    designed_set -= out_of_range

    if not designed_set:
        msg = (
            "No designed positions within PDB coordinate range. "
            "Treating all positions as designed."
        )
        logger.warning(msg)
        warnings_list.append(msg)
        designed_set = set(range(L))

    fixed_set = set(range(L)) - designed_set

    logger.info(
        "L=%d, designed=%d, fixed=%d.",
        L, len(designed_set), len(fixed_set),
    )

    # ------------------------------------------------------------------ #
    # 4. Load model (lazy, thread-safe)                                    #
    # ------------------------------------------------------------------ #
    model, torch_device = _ProteinMPNNLoader.get(
        checkpoint_path=checkpoint_path,
        device_override=device,
    )
    actual_ckpt = _ProteinMPNNLoader._loaded_checkpoint or "unknown"

    # ------------------------------------------------------------------ #
    # 5. Build feature tensors                                             #
    # ------------------------------------------------------------------ #
    coords_t    = _coords_to_tensor(coords, torch_device)
    S_ref       = _seq_to_tensor(pdb_sequence, torch_device)
    mask        = _build_validity_mask(coords, torch_device)
    chain_M     = _build_chain_mask(L, designed_set, torch_device)
    residue_idx = _build_residue_idx(resnums, torch_device)
    chain_enc   = _build_chain_encoding(L, chain_int=1, device=torch_device)
    bias_by_res = _build_bias_by_res(pdb_sequence, designed_set, torch_device)

    # ------------------------------------------------------------------ #
    # 6. Sample sequences at each temperature                              #
    # ------------------------------------------------------------------ #
    t_start     = time.perf_counter()
    all_results: list[dict] = []

    for T in temperatures:
        logger.info("Sampling %d sequences at T=%.3f …", num_seqs, T)
        all_results.extend(
            _run_sampling(
                model=model,
                coords_t=coords_t,
                S_ref=S_ref,
                mask=mask,
                chain_M=chain_M,
                residue_idx=residue_idx,
                chain_encoding=chain_enc,
                bias_by_res=bias_by_res,
                designed_indices=designed_set,
                reference_sequence=pdb_sequence,
                temperature=T,
                num_seqs=num_seqs,
                device=torch_device,
                seed=seed,
            )
        )

    inference_time = time.perf_counter() - t_start

    # ------------------------------------------------------------------ #
    # 7. Rank and package                                                  #
    # ------------------------------------------------------------------ #
    all_results.sort(key=lambda r: r["sequence_score"])

    sequences: list[DesignedSequence] = [
        DesignedSequence(
            rank=rank_idx,
            sequence=r["sequence"],
            designed_subsequence=r["designed_subsequence"],
            sequence_score=r["sequence_score"],
            global_score=r["global_score"],
            recovery_rate=r["recovery_rate"],
            per_residue_log_probs=r["per_residue_log_probs"],
            temperature=r["temperature"],
            sample_index=r["sample_index"],
        )
        for rank_idx, r in enumerate(all_results, start=1)
    ]

    result = InverseFoldResult(
        sequence_id=seq_id,
        input_pdb_path=pdb_path,
        sequences=sequences,
        designed_positions=sorted(designed_set),
        fixed_positions=sorted(fixed_set),
        num_designed_residues=len(designed_set),
        num_fixed_residues=len(fixed_set),
        sequence_length=L,
        reference_sequence="".join(pdb_sequence),
        temperatures_used=temperatures,
        num_seqs_per_temperature=num_seqs,
        model_checkpoint=actual_ckpt,
        device_used=str(torch_device),
        inference_time_s=inference_time,
        warnings=warnings_list,
    )

    logger.info(result.summary())

    # ------------------------------------------------------------------ #
    # 8. Write outputs                                                     #
    # ------------------------------------------------------------------ #
    fasta_path = output_dir / f"{seq_id}_designed.fasta"
    json_path  = output_dir / f"{seq_id}_designed.json"
    result.write_fasta(fasta_path, overwrite=overwrite)
    write_inverse_fold_report(result, json_path, overwrite=overwrite)

    return result


def write_inverse_fold_report(
    result: InverseFoldResult,
    output_path: str | Path,
    overwrite: bool = False,
) -> Path:
    """
    Write an ``InverseFoldResult`` to a JSON file.

    Parameters
    ----------
    result : InverseFoldResult
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
            f"Report '{output_path}' already exists. "
            "Pass overwrite=True to replace it."
        )

    output_path.write_text(result.to_json(), encoding="utf-8")
    logger.info("Inverse fold report written to '%s'.", output_path)
    return output_path


def release_model() -> None:
    """Unload the ProteinMPNN model and free CUDA memory."""
    _ProteinMPNNLoader.unload()


def model_is_loaded() -> bool:
    """Return True if the ProteinMPNN model is currently in memory."""
    return _ProteinMPNNLoader.is_loaded()
