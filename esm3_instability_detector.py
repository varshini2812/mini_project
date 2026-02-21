"""
esm3_instability_detector.py
============================
GeneForge Pipeline — Stage 3: Protein Instability Detection via ESM-3

Scientific Background
---------------------
ESM-3 (Evolutionary Scale Model 3, EvolutionaryScale 2024) is a multimodal
protein language model trained on ~2.78 billion protein sequences across
sequence, structure, and function tracks.  Like its predecessors (ESM-1v,
ESM-2), it learns a statistical model of residue co-evolution that encodes
implicit structural and functional constraints.

This module exploits the *masked marginal probability* of each residue as a
proxy for structural plausibility.  For a protein sequence x = (x₁ … xL):

    Masked log-probability at position i:
        log P(xᵢ | x_{≠i}; θ)

    Obtained by:
    1. Replace xᵢ with the mask token.
    2. Forward-pass through ESM-3.
    3. Apply log-softmax over the vocabulary at position i.
    4. Index into the true residue's token.

    Per-residue perplexity:
        PPL(i) = exp(−log P(xᵢ | x_{≠i}; θ))

A residue with high perplexity is *unexpected* given its sequence context —
the model assigns low probability to it.  Clusters of high-perplexity
residues (spikes) indicate that the local sequence neighbourhood is
evolutionarily improbable, which correlates strongly with:
    - Destabilising mutations (ΔΔG > 0)
    - Buried hydrophobic-to-charged substitutions
    - Disrupted disulfide networks or active-site geometry
    - Premature stop artefacts translated as 'X'

Spike Detection Algorithm
--------------------------
Raw per-residue PPL values carry high-frequency noise from the discrete
amino acid alphabet.  We apply:

    1. Optional Gaussian smoothing (σ = 1.5 residues, window = 9) to
       produce a smooth PPL landscape while preserving genuine spikes.

    2. Z-score thresholding:
           z(i) = (PPL_smooth(i) − μ) / σ_global
           SPIKE if z(i) ≥ Z_THRESHOLD  (default 2.0)

    3. Hard absolute threshold:
           SPIKE if PPL_smooth(i) ≥ PPL_ABS_THRESHOLD  (default 20.0)
       Amino acids have a natural alphabet of 20, so an average perplexity
       of ≥20 means the model is performing at or below chance — a
       biologically severe signal.

    4. Contiguous spike regions are merged into ``SpikeRegion`` objects
       with flanking residue context for interpretability.

ESM-3 API Notes
---------------
EvolutionaryScale distributes ESM-3 through the ``esm`` Python package
(``pip install esm``, requires acceptance of the non-commercial licence at
https://github.com/evolutionaryScale/esm).  The public open weights are
``esm3_sm_open_v1`` (1.4 B parameters).  Larger variants (``esm3_open_v1``)
are accessible via the EvolutionaryScale Forge API.

Model loading:
    from esm.models.esm3 import ESM3
    from esm.sdk.api   import ESMProtein, LogitsConfig, LogitsOutput
    model: ESM3 = ESM3.from_pretrained("esm3_sm_open_v1").to(device)

Tokenisation:
    protein_obj = ESMProtein(sequence=seq)
    tokens      = model.encode(protein_obj)          # ESMProteinTensor

Logit extraction for masked marginal scoring:
    We replace individual token positions with model.tokenizers.sequence.mask_token_id,
    run a forward pass requesting sequence logits, then read
    LogitsOutput.sequence — shape (L+2, vocab_size), where positions 0 and
    L+1 are BOS/EOS.

Environment Variable
--------------------
    GENEFORGE_ESM3_MODEL  — model identifier or local checkpoint path.
                            Defaults to ``esm3_sm_open_v1``.
                            Set to an absolute path for offline/air-gapped use.

Data Flow
---------
    Input  : str | ProteinRecord  (from protein_translation.py)
    Output : InstabilityReport    (dataclass)

Dependencies
------------
    esm      >= 3.0   (EvolutionaryScale)  — model inference
    torch    >= 2.1                         — tensor operations
    numpy    >= 1.26                        — score processing
    Python   >= 3.12
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final, NamedTuple, Union

import numpy as np

if TYPE_CHECKING:
    # These are imported lazily at runtime; listed here for static analysis only.
    import torch
    from esm.models.esm3 import ESM3
    from esm.sdk.api import ESMProtein, LogitsConfig, LogitsOutput

# ---------------------------------------------------------------------------
# Module logger
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants & defaults
# ---------------------------------------------------------------------------

#: Environment variable controlling which ESM-3 model is loaded.
ENV_MODEL_KEY: Final[str] = "GENEFORGE_ESM3_MODEL"

#: Default ESM-3 model identifier (open weights, 1.4 B params).
DEFAULT_MODEL_ID: Final[str] = "esm3_sm_open_v1"

#: Z-score threshold above which a residue is classified as a spike.
#: 2.0 ≈ top ~2.3 % of a normal distribution.
Z_THRESHOLD: Final[float] = 2.0

#: Absolute perplexity ceiling.  Alphabet size = 20 standard amino acids;
#: PPL ≥ 20 means the model assigns ≤ 1/20 probability — at-or-below chance.
PPL_ABS_THRESHOLD: Final[float] = 20.0

#: Gaussian smoothing kernel half-width in residues.
SMOOTH_SIGMA: Final[float] = 1.5

#: Gaussian kernel window size (must be odd).
SMOOTH_WINDOW: Final[int] = 9

#: Minimum contiguous spike length (residues) to report as a SpikeRegion.
MIN_SPIKE_LEN: Final[int] = 1

#: Maximum sequence length processable without chunking.
#: Sequences above this trigger a warning; chunking is applied automatically.
MAX_SAFE_SEQ_LEN: Final[int] = 1022  # ESM-3 positional embedding limit

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class InstabilityDetectorError(Exception):
    """Base exception for all ESM-3 instability detection failures."""


class ModelLoadError(InstabilityDetectorError):
    """Raised when the ESM-3 model cannot be loaded."""


class InferenceError(InstabilityDetectorError):
    """Raised when a forward pass fails during perplexity computation."""


class SequenceTooShortError(InstabilityDetectorError):
    """Raised when the protein sequence is too short for meaningful scoring."""


class InvalidInputError(InstabilityDetectorError):
    """Raised when the input type or content is not valid."""


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ResidueScore:
    """
    Per-residue instability annotation.

    Attributes
    ----------
    index : int
        0-based position in the protein sequence.
    residue : str
        Single-letter amino acid code at this position.
    masked_log_prob : float
        log P(xᵢ | x_{≠i}; θ) — higher (less negative) is better.
    perplexity : float
        exp(−masked_log_prob) — lower is more stable.
    perplexity_smoothed : float
        Gaussian-smoothed perplexity used for spike detection.
    z_score : float
        (perplexity_smoothed − μ) / σ_global.
    is_spike : bool
        True when this residue is flagged as a perplexity spike.
    """

    index: int
    residue: str
    masked_log_prob: float
    perplexity: float
    perplexity_smoothed: float
    z_score: float
    is_spike: bool

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "residue": self.residue,
            "masked_log_prob": round(self.masked_log_prob, 6),
            "perplexity": round(self.perplexity, 4),
            "perplexity_smoothed": round(self.perplexity_smoothed, 4),
            "z_score": round(self.z_score, 4),
            "is_spike": self.is_spike,
        }


class SpikeRegion(NamedTuple):
    """
    A contiguous stretch of flagged (high-perplexity) residues.

    Attributes
    ----------
    start : int
        0-based index of the first spike residue (inclusive).
    end : int
        0-based index of the last spike residue (inclusive).
    length : int
        Number of residues in the region.
    peak_perplexity : float
        Maximum smoothed perplexity within the region.
    peak_index : int
        0-based index of the peak-perplexity residue.
    mean_perplexity : float
        Mean smoothed perplexity across the region.
    residues : str
        Amino acid string spanning the region.
    """

    start: int
    end: int
    length: int
    peak_perplexity: float
    peak_index: int
    mean_perplexity: float
    residues: str

    def to_dict(self) -> dict:
        return {
            "start": self.start,
            "end": self.end,
            "length": self.length,
            "peak_perplexity": round(self.peak_perplexity, 4),
            "peak_index": self.peak_index,
            "mean_perplexity": round(self.mean_perplexity, 4),
            "residues": self.residues,
        }


@dataclass(slots=True)
class InstabilityReport:
    """
    Full instability characterisation for one protein sequence.

    This is the Stage 3 output consumed by Stage 4 (RFdiffusion healer).

    Attributes
    ----------
    sequence_id : str
        Identifier from the source ProteinRecord or supplied directly.
    sequence : str
        The protein sequence that was scored.
    residue_scores : list[ResidueScore]
        Per-residue annotations in sequence order.
    spike_regions : list[SpikeRegion]
        Contiguous high-perplexity regions, sorted by start position.
    spike_indices : list[int]
        Flat list of all 0-based spike residue indices.
    mean_perplexity : float
        Sequence-level mean of raw per-residue perplexity.
    std_perplexity : float
        Standard deviation of raw per-residue perplexity.
    max_perplexity : float
        Maximum raw per-residue perplexity (worst single residue).
    fraction_unstable : float
        Fraction of residues classified as spikes: |spikes| / L.
    model_id : str
        ESM-3 model identifier used.
    device_used : str
        "cuda:N" or "cpu" — the device on which inference ran.
    inference_time_s : float
        Wall-clock time (seconds) for the full masked-marginal computation.
    z_threshold : float
        Z-score threshold applied for spike detection.
    ppl_abs_threshold : float
        Absolute perplexity threshold applied.
    sequence_length : int
        Length of the scored sequence.
    warnings : list[str]
        Non-fatal issues detected during scoring.
    """

    sequence_id: str
    sequence: str
    residue_scores: list[ResidueScore]
    spike_regions: list[SpikeRegion]
    spike_indices: list[int]
    mean_perplexity: float
    std_perplexity: float
    max_perplexity: float
    fraction_unstable: float
    model_id: str
    device_used: str
    inference_time_s: float
    z_threshold: float
    ppl_abs_threshold: float
    sequence_length: int
    warnings: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Derived views
    # ------------------------------------------------------------------

    @property
    def is_stable(self) -> bool:
        """True when no spike regions were detected."""
        return len(self.spike_regions) == 0

    @property
    def needs_repair(self) -> bool:
        """True when the fraction of unstable residues exceeds 5 %."""
        return self.fraction_unstable > 0.05

    def perplexity_array(self) -> np.ndarray:
        """Return raw per-residue perplexity as a NumPy array (shape: L,)."""
        return np.array([r.perplexity for r in self.residue_scores], dtype=np.float32)

    def smoothed_perplexity_array(self) -> np.ndarray:
        """Return smoothed per-residue perplexity as a NumPy array (shape: L,)."""
        return np.array(
            [r.perplexity_smoothed for r in self.residue_scores], dtype=np.float32
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "sequence_id": self.sequence_id,
            "model_id": self.model_id,
            "device_used": self.device_used,
            "sequence_length": self.sequence_length,
            "inference_time_s": round(self.inference_time_s, 3),
            "mean_perplexity": round(self.mean_perplexity, 4),
            "std_perplexity": round(self.std_perplexity, 4),
            "max_perplexity": round(self.max_perplexity, 4),
            "fraction_unstable": round(self.fraction_unstable, 4),
            "is_stable": self.is_stable,
            "needs_repair": self.needs_repair,
            "z_threshold": self.z_threshold,
            "ppl_abs_threshold": self.ppl_abs_threshold,
            "spike_indices": self.spike_indices,
            "spike_regions": [sr.to_dict() for sr in self.spike_regions],
            "residue_scores": [rs.to_dict() for rs in self.residue_scores],
            "warnings": self.warnings,
        }

    def to_json(self, indent: int = 2) -> str:
        import json
        return json.dumps(self.to_dict(), indent=indent)

    def summary(self) -> str:
        """One-line human-readable summary for logging / CLI output."""
        return (
            f"[{self.sequence_id}] L={self.sequence_length}, "
            f"mean_ppl={self.mean_perplexity:.2f}, "
            f"max_ppl={self.max_perplexity:.2f}, "
            f"spikes={len(self.spike_indices)} ({self.fraction_unstable:.1%}), "
            f"regions={len(self.spike_regions)}, "
            f"stable={self.is_stable}, "
            f"model={self.model_id}, "
            f"device={self.device_used}, "
            f"t={self.inference_time_s:.1f}s"
        )


# ---------------------------------------------------------------------------
# ESM-3 model loader — lazy singleton, thread-safe
# ---------------------------------------------------------------------------


class _ESM3Loader:
    """
    Thread-safe lazy singleton that loads ESM-3 exactly once per process.

    The model is held in class-level state so that multiple calls to
    ``detect_instability()`` within the same process pay the load cost only
    once.  Memory is released explicitly via ``unload()``.

    Design principles
    -----------------
    - **Lazy**: nothing is imported or allocated until the first call to
      ``get()``.
    - **Thread-safe**: a ``threading.Lock`` prevents double-loading when
      multiple threads call ``get()`` concurrently.
    - **GPU-aware**: auto-detects CUDA; falls back to CPU gracefully.
    - **Configurable**: model identifier read from the
      ``GENEFORGE_ESM3_MODEL`` environment variable, overridable at call
      site.
    - **Memory-safe**: after ``unload()``, CUDA memory is released via
      ``torch.cuda.empty_cache()``.
    """

    _lock: threading.Lock = threading.Lock()
    _model: "ESM3 | None" = None
    _loaded_model_id: str | None = None
    _device: "torch.device | None" = None

    @classmethod
    def get(
        cls,
        model_id: str | None = None,
        device_override: str | None = None,
    ) -> tuple["ESM3", "torch.device"]:
        """
        Return the loaded ESM-3 model and the device it lives on.

        If the model is already loaded with the same ``model_id``, the
        cached instance is returned immediately (no disk I/O).

        Parameters
        ----------
        model_id : str | None
            ESM-3 model identifier or local checkpoint path.
            Reads ``GENEFORGE_ESM3_MODEL`` env var if ``None``.
            Falls back to ``DEFAULT_MODEL_ID`` if env var is also unset.
        device_override : str | None
            Force a specific device string, e.g. ``"cpu"`` or ``"cuda:1"``.
            ``None`` means auto-detect.

        Returns
        -------
        tuple[ESM3, torch.device]

        Raises
        ------
        ModelLoadError
            If the ``esm`` package is not installed or the checkpoint cannot
            be found / downloaded.
        """
        resolved_id = (
            model_id
            or os.environ.get(ENV_MODEL_KEY)
            or DEFAULT_MODEL_ID
        )

        with cls._lock:
            # Fast path — already loaded with the correct model.
            if cls._model is not None and cls._loaded_model_id == resolved_id:
                logger.debug(
                    "ESM-3 cache hit: model='%s', device=%s.",
                    resolved_id,
                    cls._device,
                )
                return cls._model, cls._device  # type: ignore[return-value]

            # Slow path — load from disk / HuggingFace hub.
            device = cls._resolve_device(device_override)

            logger.info(
                "Loading ESM-3 model '%s' onto %s …", resolved_id, device
            )
            t0 = time.perf_counter()

            try:
                model = cls._load_model(resolved_id, device)
            except ImportError as exc:
                raise ModelLoadError(
                    "The 'esm' package is required for ESM-3 inference but is not "
                    "installed.  Install it with:\n"
                    "    pip install esm\n"
                    "and accept the EvolutionaryScale non-commercial licence at "
                    "https://github.com/evolutionaryScale/esm\n"
                    f"Original error: {exc}"
                ) from exc
            except Exception as exc:
                raise ModelLoadError(
                    f"Failed to load ESM-3 model '{resolved_id}': "
                    f"{type(exc).__name__}: {exc}"
                ) from exc

            elapsed = time.perf_counter() - t0
            logger.info(
                "ESM-3 model '%s' loaded in %.1f s on %s.", resolved_id, elapsed, device
            )

            if device.type == "cuda":
                cls._log_vram(device)

            cls._model = model
            cls._loaded_model_id = resolved_id
            cls._device = device

        return cls._model, cls._device  # type: ignore[return-value]

    @staticmethod
    def _load_model(model_id: str, device: "torch.device") -> "ESM3":
        """
        Instantiate the ESM-3 model.

        Supports two source modes:
        - **Hub**: ``model_id`` is a string identifier such as
          ``"esm3_sm_open_v1"``.  Weights are downloaded from the
          EvolutionaryScale model hub on first use.
        - **Local**: ``model_id`` is an absolute or relative path to a
          directory containing a ``config.json`` and weight shard files in
          safetensors or PyTorch format, as produced by
          ``model.save_pretrained(path)``.

        Parameters
        ----------
        model_id : str
        device : torch.device

        Returns
        -------
        ESM3
            Model in ``eval()`` mode, moved to ``device``.
        """
        # Import here so the module is importable without esm installed.
        from esm.models.esm3 import ESM3  # type: ignore[import]

        if os.path.isabs(model_id) or os.path.exists(model_id):
            # Local checkpoint
            logger.debug("Loading ESM-3 from local path: %s", model_id)
            model = ESM3.from_pretrained(model_id)
        else:
            # EvolutionaryScale hub identifier
            logger.debug("Fetching ESM-3 from hub: %s", model_id)
            model = ESM3.from_pretrained(model_id)

        model = model.to(device)
        model.eval()
        return model

    @staticmethod
    def _resolve_device(device_override: str | None) -> "torch.device":
        """
        Select the compute device.

        Priority: explicit override > CUDA (if available) > CPU.

        Parameters
        ----------
        device_override : str | None

        Returns
        -------
        torch.device
        """
        import torch  # type: ignore[import]

        if device_override is not None:
            dev = torch.device(device_override)
            logger.debug("Device override: %s", dev)
            return dev

        if torch.cuda.is_available():
            dev = torch.device("cuda")
            logger.debug(
                "CUDA available — using %s (%s).",
                dev,
                torch.cuda.get_device_name(dev),
            )
            return dev

        logger.debug("CUDA not available — using CPU.")
        return torch.device("cpu")

    @staticmethod
    def _log_vram(device: "torch.device") -> None:
        """Log GPU memory allocation after model load."""
        try:
            import torch  # type: ignore[import]
            alloc = torch.cuda.memory_allocated(device) / 1e9
            reserved = torch.cuda.memory_reserved(device) / 1e9
            logger.info(
                "VRAM after model load — allocated: %.2f GB, reserved: %.2f GB.",
                alloc,
                reserved,
            )
        except Exception:
            pass  # Non-fatal — logging only.

    @classmethod
    def unload(cls) -> None:
        """
        Release the model from memory and free CUDA caches.

        Call this explicitly if the pipeline is moving to a stage that
        does not need ESM-3 and VRAM pressure is a concern.
        """
        with cls._lock:
            if cls._model is None:
                logger.debug("ESM-3 model is not loaded; nothing to unload.")
                return

            logger.info(
                "Unloading ESM-3 model '%s' …", cls._loaded_model_id
            )
            del cls._model
            cls._model = None
            cls._loaded_model_id = None

            try:
                import torch  # type: ignore[import]
                if cls._device is not None and cls._device.type == "cuda":
                    torch.cuda.empty_cache()
                    logger.info("CUDA cache cleared.")
            except ImportError:
                pass

            cls._device = None
            logger.info("ESM-3 model unloaded.")

    @classmethod
    def is_loaded(cls) -> bool:
        """Return True if the model is currently resident in memory."""
        with cls._lock:
            return cls._model is not None


# ---------------------------------------------------------------------------
# Score computation — masked marginal log-probabilities
# ---------------------------------------------------------------------------


def _gaussian_kernel(sigma: float, window: int) -> np.ndarray:
    """
    Compute a normalised 1-D Gaussian kernel for perplexity smoothing.

    Parameters
    ----------
    sigma : float
        Standard deviation in residue units.
    window : int
        Kernel width (must be odd; if even, incremented by 1).

    Returns
    -------
    np.ndarray
        Shape (window,), sums to 1.0.
    """
    if window % 2 == 0:
        window += 1
    half = window // 2
    x = np.arange(-half, half + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    return (kernel / kernel.sum()).astype(np.float32)


def _smooth_perplexity(
    ppl: np.ndarray,
    sigma: float = SMOOTH_SIGMA,
    window: int = SMOOTH_WINDOW,
) -> np.ndarray:
    """
    Apply Gaussian smoothing to a per-residue perplexity array.

    Edge effects are handled with ``mode='same'`` and ``np.convolve``,
    which pads with zeros; this slightly underestimates perplexity at
    the N- and C-termini.  For production use with very short proteins
    (< 20 aa) this is acceptable; for peptides, pass ``sigma=0`` to
    disable smoothing.

    Parameters
    ----------
    ppl : np.ndarray
        Shape (L,), raw per-residue perplexity.
    sigma : float
        Gaussian standard deviation in residue units.
    window : int
        Kernel width.

    Returns
    -------
    np.ndarray
        Shape (L,), smoothed perplexity.
    """
    if sigma <= 0.0 or len(ppl) < 3:
        return ppl.copy()

    kernel = _gaussian_kernel(sigma, window)
    return np.convolve(ppl, kernel, mode="same").astype(np.float32)


def _compute_masked_log_probs(
    sequence: str,
    model: "ESM3",
    device: "torch.device",
) -> np.ndarray:
    """
    Compute per-residue masked marginal log-probabilities using ESM-3.

    For each position i in ``sequence``, this function:
        1. Encodes the sequence into an ``ESMProteinTensor``.
        2. Replaces token i+1 (accounting for BOS offset) with the mask token.
        3. Requests sequence logits from the model.
        4. Computes log-softmax over the vocabulary at position i+1.
        5. Extracts the log-probability of the true residue.

    This is an O(L) inference loop — L forward passes for a sequence of
    length L.  For production throughput on long sequences, consider
    batched approximate methods (random masking subsets), but the
    per-position approach is exact and required for faithful spike
    detection.

    Parameters
    ----------
    sequence : str
        Clean amino acid string (no stop codon, no gaps, standard 20 aa + X).
    model : ESM3
        Loaded ESM-3 model in eval mode on ``device``.
    device : torch.device

    Returns
    -------
    np.ndarray
        Shape (L,), dtype float32.  Each element is
        log P(xᵢ | x_{≠i}; θ) ∈ (−∞, 0].

    Raises
    ------
    InferenceError
        On any failure during tokenisation or forward pass.
    """
    import torch  # type: ignore[import]
    import torch.nn.functional as F  # type: ignore[import]
    from esm.sdk.api import ESMProtein, LogitsConfig  # type: ignore[import]

    L = len(sequence)
    log_probs = np.zeros(L, dtype=np.float32)

    # Tokenise once — we will mutate a clone per position.
    try:
        protein_obj = ESMProtein(sequence=sequence)
        protein_tensor = model.encode(protein_obj)
    except Exception as exc:
        raise InferenceError(
            f"ESM-3 tokenisation failed for sequence of length {L}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    # Retrieve the mask token id from the model's sequence tokeniser.
    try:
        mask_token_id: int = model.tokenizers.sequence.mask_token_id
    except AttributeError:
        # Fallback: ESM-3 uses token 32 as mask in the sequence track.
        mask_token_id = 32
        logger.warning(
            "Could not read mask_token_id from model.tokenizers.sequence; "
            "using fallback value %d.",
            mask_token_id,
        )

    # Logit configuration — only sequence track needed.
    try:
        logit_config = LogitsConfig(sequence=True, return_embeddings=False)
    except TypeError:
        # Older API versions use positional args or a different signature.
        logit_config = LogitsConfig(sequence=True)

    with torch.no_grad():
        for i in range(L):
            # Clone the token tensor to avoid mutating the canonical encoding.
            tokens_clone = protein_tensor.sequence.clone()  # shape: (L+2,)

            # Position 0 is BOS, positions 1…L are residues, L+1 is EOS.
            tokens_clone[i + 1] = mask_token_id

            # Build a modified protein tensor with the masked sequence.
            masked_protein = _replace_sequence_tokens(
                protein_tensor, tokens_clone
            )

            try:
                logits_output = model.logits(masked_protein, logit_config)
            except Exception as exc:
                raise InferenceError(
                    f"ESM-3 forward pass failed at position {i} "
                    f"(residue '{sequence[i]}'): {type(exc).__name__}: {exc}"
                ) from exc

            # logits_output.sequence: shape (1, L+2, vocab_size) or (L+2, vocab_size)
            seq_logits = logits_output.sequence
            if seq_logits.dim() == 3:
                seq_logits = seq_logits.squeeze(0)  # → (L+2, vocab_size)

            position_logits = seq_logits[i + 1]  # (vocab_size,)
            lp_all = F.log_softmax(position_logits, dim=-1)

            true_token_id: int = protein_tensor.sequence[i + 1].item()  # type: ignore[union-attr]
            log_probs[i] = lp_all[true_token_id].item()

    return log_probs


def _replace_sequence_tokens(protein_tensor, new_sequence_tokens: "torch.Tensor"):
    """
    Return a copy of ``protein_tensor`` with the sequence track replaced.

    ESM-3's ``ESMProteinTensor`` is a dataclass-like container.  We build
    a shallow replacement to avoid mutating the original (which is reused
    across positions in the masking loop).

    Parameters
    ----------
    protein_tensor : ESMProteinTensor
        Original protein tensor from ``model.encode()``.
    new_sequence_tokens : torch.Tensor
        Shape (L+2,) — the modified sequence token IDs.

    Returns
    -------
    ESMProteinTensor
        New tensor object with only the sequence track swapped.
    """
    # ESMProteinTensor supports ``replace()`` in ESM-3 ≥ 3.0.
    # For robustness we also handle the case where it does not.
    try:
        return protein_tensor.replace(sequence=new_sequence_tokens)
    except AttributeError:
        pass

    # Fallback: use dataclasses.replace if ESMProteinTensor is a dataclass.
    try:
        import dataclasses  # stdlib
        return dataclasses.replace(protein_tensor, sequence=new_sequence_tokens)
    except (TypeError, AttributeError):
        pass

    # Last resort: shallow copy and manual attribute assignment.
    import copy
    clone = copy.copy(protein_tensor)
    clone.sequence = new_sequence_tokens
    return clone


# ---------------------------------------------------------------------------
# Spike detection
# ---------------------------------------------------------------------------


def _compute_z_scores(ppl_smooth: np.ndarray) -> np.ndarray:
    """
    Compute z-scores for a smoothed perplexity array.

    z(i) = (ppl_smooth(i) − μ) / σ

    A small epsilon (1e-8) is added to the denominator to prevent
    division by zero for constant-perplexity sequences (e.g. polyAla).

    Parameters
    ----------
    ppl_smooth : np.ndarray
        Shape (L,).

    Returns
    -------
    np.ndarray
        Shape (L,), dtype float32.
    """
    mu = np.mean(ppl_smooth)
    sigma = np.std(ppl_smooth)
    return ((ppl_smooth - mu) / (sigma + 1e-8)).astype(np.float32)


def _build_residue_scores(
    sequence: str,
    log_probs: np.ndarray,
    ppl_raw: np.ndarray,
    ppl_smooth: np.ndarray,
    z_scores: np.ndarray,
    z_threshold: float,
    ppl_abs_threshold: float,
) -> list[ResidueScore]:
    """
    Combine per-residue arrays into a list of ``ResidueScore`` objects.

    A residue is classified as a spike when either:
    - ``z_scores[i] >= z_threshold``, OR
    - ``ppl_smooth[i] >= ppl_abs_threshold``

    Parameters
    ----------
    sequence : str
    log_probs : np.ndarray         shape (L,)
    ppl_raw : np.ndarray           shape (L,)
    ppl_smooth : np.ndarray        shape (L,)
    z_scores : np.ndarray          shape (L,)
    z_threshold : float
    ppl_abs_threshold : float

    Returns
    -------
    list[ResidueScore]
    """
    scores: list[ResidueScore] = []
    for i, aa in enumerate(sequence):
        is_spike = bool(
            z_scores[i] >= z_threshold or ppl_smooth[i] >= ppl_abs_threshold
        )
        scores.append(
            ResidueScore(
                index=i,
                residue=aa,
                masked_log_prob=float(log_probs[i]),
                perplexity=float(ppl_raw[i]),
                perplexity_smoothed=float(ppl_smooth[i]),
                z_score=float(z_scores[i]),
                is_spike=is_spike,
            )
        )
    return scores


def _find_spike_regions(
    scores: list[ResidueScore],
    sequence: str,
    min_length: int = MIN_SPIKE_LEN,
) -> list[SpikeRegion]:
    """
    Merge contiguous flagged residues into ``SpikeRegion`` objects.

    Parameters
    ----------
    scores : list[ResidueScore]
    sequence : str
    min_length : int
        Regions shorter than this are discarded.

    Returns
    -------
    list[SpikeRegion]
        Sorted by ``start``.
    """
    regions: list[SpikeRegion] = []

    i = 0
    while i < len(scores):
        if not scores[i].is_spike:
            i += 1
            continue

        # Start of a new contiguous spike run.
        start = i
        while i < len(scores) and scores[i].is_spike:
            i += 1
        end = i - 1  # inclusive

        region_scores = scores[start : end + 1]
        length = end - start + 1

        if length < min_length:
            continue

        ppls = [rs.perplexity_smoothed for rs in region_scores]
        peak_local = int(np.argmax(ppls))

        regions.append(
            SpikeRegion(
                start=start,
                end=end,
                length=length,
                peak_perplexity=float(ppls[peak_local]),
                peak_index=start + peak_local,
                mean_perplexity=float(np.mean(ppls)),
                residues=sequence[start : end + 1],
            )
        )

    return regions


# ---------------------------------------------------------------------------
# Sequence validation & chunking
# ---------------------------------------------------------------------------


def _validate_sequence(sequence: str, sequence_id: str) -> list[str]:
    """
    Validate a protein sequence for ESM-3 compatibility.

    Returns a list of warning strings (may be empty).  Raises on fatal errors.

    Parameters
    ----------
    sequence : str
    sequence_id : str

    Returns
    -------
    list[str]
        Warnings.

    Raises
    ------
    SequenceTooShortError
    InvalidInputError
    """
    VALID_AA = frozenset("ACDEFGHIKLMNPQRSTVWYX")
    MIN_LEN = 5

    warnings: list[str] = []

    if not sequence:
        raise SequenceTooShortError(
            f"'{sequence_id}': protein sequence is empty."
        )
    if len(sequence) < MIN_LEN:
        raise SequenceTooShortError(
            f"'{sequence_id}': protein sequence length {len(sequence)} is below "
            f"the minimum of {MIN_LEN} residues required for meaningful ESM-3 scoring."
        )

    invalid = sorted(set(sequence) - VALID_AA)
    if invalid:
        raise InvalidInputError(
            f"'{sequence_id}': sequence contains non-standard characters: {invalid}. "
            "Valid input is the 20 standard amino acids plus X (unknown)."
        )

    x_count = sequence.count("X")
    if x_count > 0:
        pct = 100.0 * x_count / len(sequence)
        warnings.append(
            f"{x_count} ambiguous residue(s) ('X', {pct:.1f}%) present. "
            "ESM-3 will assign low probability to X; expect artificially "
            "elevated perplexity at these positions."
        )
    if x_count > len(sequence) * 0.1:
        warnings.append(
            "More than 10% of residues are 'X'. Instability scores will be "
            "dominated by ambiguous codons rather than genuine structural context. "
            "Consider re-sequencing or improving upstream base-calling."
        )

    if len(sequence) > MAX_SAFE_SEQ_LEN:
        warnings.append(
            f"Sequence length {len(sequence)} exceeds the ESM-3 positional "
            f"embedding limit ({MAX_SAFE_SEQ_LEN}).  The sequence will be scored "
            "in overlapping chunks; spike detection at chunk boundaries may be "
            "slightly less accurate."
        )

    return warnings


def _chunk_sequence(
    sequence: str,
    max_len: int = MAX_SAFE_SEQ_LEN,
    overlap: int = 50,
) -> list[tuple[int, str]]:
    """
    Split a long sequence into overlapping chunks for ESM-3 scoring.

    Parameters
    ----------
    sequence : str
    max_len : int
        Maximum chunk length.
    overlap : int
        Number of overlues shared between adjacent chunks (for edge
        smoothing — only the non-overlapping centre is kept).

    Returns
    -------
    list[tuple[int, str]]
        Each element: (global_start_index, chunk_sequence).
    """
    if len(sequence) <= max_len:
        return [(0, sequence)]

    chunks: list[tuple[int, str]] = []
    step = max_len - overlap
    pos = 0

    while pos < len(sequence):
        end = min(pos + max_len, len(sequence))
        chunks.append((pos, sequence[pos:end]))
        if end == len(sequence):
            break
        pos += step

    logger.debug(
        "Sequence split into %d chunks (max_len=%d, overlap=%d).",
        len(chunks),
        max_len,
        overlap,
    )
    return chunks


def _merge_chunk_log_probs(
    chunk_results: list[tuple[int, np.ndarray]],
    total_length: int,
    overlap: int = 50,
) -> np.ndarray:
    """
    Merge per-chunk masked log-probability arrays into a single (L,) array.

    For overlapping regions, values from the chunk whose centre is closer
    to the overlapping position are preferred (i.e. we prefer interior
    positions over edge positions for each chunk).

    Parameters
    ----------
    chunk_results : list[tuple[int, np.ndarray]]
        List of (global_start, log_probs_array) — one per chunk.
    total_length : int
        Full sequence length.
    overlap : int
        Overlap width used during chunking.

    Returns
    -------
    np.ndarray
        Shape (total_length,), dtype float32.
    """
    merged = np.full(total_length, fill_value=np.nan, dtype=np.float32)
    weights = np.zeros(total_length, dtype=np.float32)

    for chunk_start, lp in chunk_results:
        chunk_len = len(lp)
        # Weight: triangular — peaks at centre of chunk, tapers at edges.
        w = np.minimum(
            np.arange(1, chunk_len + 1, dtype=np.float32),
            np.arange(chunk_len, 0, -1, dtype=np.float32),
        )
        global_indices = np.arange(chunk_start, chunk_start + chunk_len)
        for local_i, global_i in enumerate(global_indices):
            if global_i >= total_length:
                break
            if np.isnan(merged[global_i]) or w[local_i] > weights[global_i]:
                merged[global_i] = lp[local_i]
                weights[global_i] = w[local_i]

    # Fallback: fill any remaining NaN (edge case for very short last chunks).
    nan_mask = np.isnan(merged)
    if nan_mask.any():
        logger.warning(
            "%d position(s) not covered after chunk merge; filling with global mean.",
            int(nan_mask.sum()),
        )
        mean_val = float(np.nanmean(merged)) if not np.all(nan_mask) else -3.0
        merged[nan_mask] = mean_val

    return merged


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_instability(
    protein_input: Union[str, "ProteinRecord"],  # noqa: UP007  (Python 3.12 compat alias)
    model_id: str | None = None,
    device: str | None = None,
    z_threshold: float = Z_THRESHOLD,
    ppl_abs_threshold: float = PPL_ABS_THRESHOLD,
    smooth_sigma: float = SMOOTH_SIGMA,
    smooth_window: int = SMOOTH_WINDOW,
    seed: int | None = 42,
) -> InstabilityReport:
    """
    Compute per-residue instability scores for a protein sequence via ESM-3.

    This is the primary public entry point for Stage 3 of the GeneForge
    pipeline.  It accepts either a raw amino acid string or a
    ``ProteinRecord`` from Stage 2, loads ESM-3 lazily (once per process),
    runs the masked marginal scoring loop, and returns a fully annotated
    ``InstabilityReport``.

    Parameters
    ----------
    protein_input : str | ProteinRecord
        Protein sequence to score.  If a ``ProteinRecord`` is supplied, the
        ``clean_sequence`` property is used (stop codon stripped).
    model_id : str | None
        ESM-3 model identifier or local checkpoint path.
        ``None`` reads ``GENEFORGE_ESM3_MODEL`` env var, then falls back to
        ``"esm3_sm_open_v1"``.
    device : str | None
        Compute device: ``"cuda"``, ``"cuda:1"``, ``"cpu"``, or ``None``
        for auto-detection.
    z_threshold : float
        Z-score threshold for spike classification.  Default 2.0
        (≈ top 2.3 % of sequence positions under a normal distribution).
    ppl_abs_threshold : float
        Absolute perplexity threshold.  Default 20.0 (alphabet size).
    smooth_sigma : float
        Gaussian smoothing σ in residue units.  Set to 0 to disable.
    smooth_window : int
        Gaussian kernel width (odd integer).
    seed : int | None
        If not ``None``, sets ``torch.manual_seed`` before inference for
        deterministic dropout (ESM-3 inference does not use dropout, but
        this ensures reproducibility if the model is ever updated to do so).

    Returns
    -------
    InstabilityReport

    Raises
    ------
    InvalidInputError
        If ``protein_input`` is neither a string nor a ``ProteinRecord``.
    SequenceTooShortError
        If the sequence is too short for meaningful scoring.
    ModelLoadError
        If ESM-3 weights cannot be loaded.
    InferenceError
        If a forward pass fails during masked marginal scoring.
    """
    # ------------------------------------------------------------------ #
    # 1. Resolve input                                                     #
    # ------------------------------------------------------------------ #
    if isinstance(protein_input, str):
        sequence = protein_input.replace("*", "").strip()
        sequence_id = "input_sequence"
    elif hasattr(protein_input, "clean_sequence") and hasattr(protein_input, "sequence_id"):
        # Duck-type check for ProteinRecord without a hard import.
        sequence = protein_input.clean_sequence
        sequence_id = protein_input.sequence_id
    else:
        raise InvalidInputError(
            f"protein_input must be a str or ProteinRecord; "
            f"received {type(protein_input).__name__!r}."
        )

    # ------------------------------------------------------------------ #
    # 2. Validate sequence                                                 #
    # ------------------------------------------------------------------ #
    seq_warnings = _validate_sequence(sequence, sequence_id)
    L = len(sequence)

    logger.info(
        "Instability detection for '%s': L=%d, model=%s, device=%s.",
        sequence_id,
        L,
        model_id or os.environ.get(ENV_MODEL_KEY) or DEFAULT_MODEL_ID,
        device or "auto",
    )

    # ------------------------------------------------------------------ #
    # 3. Load model (lazy, thread-safe)                                    #
    # ------------------------------------------------------------------ #
    model, torch_device = _ESM3Loader.get(model_id=model_id, device_override=device)
    actual_model_id = _ESM3Loader._loaded_model_id or DEFAULT_MODEL_ID

    # ------------------------------------------------------------------ #
    # 4. Reproducibility seed                                              #
    # ------------------------------------------------------------------ #
    if seed is not None:
        try:
            import torch as _torch  # type: ignore[import]
            _torch.manual_seed(seed)
            if torch_device.type == "cuda":
                _torch.cuda.manual_seed_all(seed)
            logger.debug("Random seed set to %d for reproducibility.", seed)
        except ImportError:
            pass

    # ------------------------------------------------------------------ #
    # 5. Compute masked marginal log-probabilities                         #
    # ------------------------------------------------------------------ #
    t_start = time.perf_counter()

    if L > MAX_SAFE_SEQ_LEN:
        chunks = _chunk_sequence(sequence, max_len=MAX_SAFE_SEQ_LEN)
        chunk_results: list[tuple[int, np.ndarray]] = []
        for chunk_start, chunk_seq in chunks:
            logger.debug(
                "'%s': scoring chunk start=%d, length=%d.",
                sequence_id,
                chunk_start,
                len(chunk_seq),
            )
            lp = _compute_masked_log_probs(chunk_seq, model, torch_device)
            chunk_results.append((chunk_start, lp))
        log_probs = _merge_chunk_log_probs(chunk_results, L)
    else:
        log_probs = _compute_masked_log_probs(sequence, model, torch_device)

    inference_time = time.perf_counter() - t_start
    logger.info(
        "'%s': masked marginal scoring complete in %.1f s (%.2f s/residue).",
        sequence_id,
        inference_time,
        inference_time / L,
    )

    # ------------------------------------------------------------------ #
    # 6. Perplexity + smoothing                                            #
    # ------------------------------------------------------------------ #
    ppl_raw: np.ndarray = np.exp(-log_probs).astype(np.float32)
    ppl_smooth = _smooth_perplexity(ppl_raw, sigma=smooth_sigma, window=smooth_window)

    # ------------------------------------------------------------------ #
    # 7. Spike detection                                                   #
    # ------------------------------------------------------------------ #
    z_scores = _compute_z_scores(ppl_smooth)
    residue_scores = _build_residue_scores(
        sequence,
        log_probs,
        ppl_raw,
        ppl_smooth,
        z_scores,
        z_threshold,
        ppl_abs_threshold,
    )
    spike_regions = _find_spike_regions(residue_scores, sequence)
    spike_indices = [rs.index for rs in residue_scores if rs.is_spike]

    # ------------------------------------------------------------------ #
    # 8. Summary statistics                                                #
    # ------------------------------------------------------------------ #
    mean_ppl = float(np.mean(ppl_raw))
    std_ppl = float(np.std(ppl_raw))
    max_ppl = float(np.max(ppl_raw))
    frac_unstable = len(spike_indices) / L if L > 0 else 0.0

    # ------------------------------------------------------------------ #
    # 9. Build report                                                      #
    # ------------------------------------------------------------------ #
    report = InstabilityReport(
        sequence_id=sequence_id,
        sequence=sequence,
        residue_scores=residue_scores,
        spike_regions=spike_regions,
        spike_indices=spike_indices,
        mean_perplexity=mean_ppl,
        std_perplexity=std_ppl,
        max_perplexity=max_ppl,
        fraction_unstable=frac_unstable,
        model_id=actual_model_id,
        device_used=str(torch_device),
        inference_time_s=inference_time,
        z_threshold=z_threshold,
        ppl_abs_threshold=ppl_abs_threshold,
        sequence_length=L,
        warnings=seq_warnings,
    )

    logger.info(report.summary())
    return report


def release_model() -> None:
    """
    Explicitly unload the ESM-3 model and free memory.

    This is a convenience wrapper around ``_ESM3Loader.unload()``.  Call
    this between pipeline runs if ESM-3 is no longer needed and you want
    to reclaim VRAM before loading RFdiffusion (Stage 4).
    """
    _ESM3Loader.unload()


def model_is_loaded() -> bool:
    """Return True if the ESM-3 model is currently resident in memory."""
    return _ESM3Loader.is_loaded()
