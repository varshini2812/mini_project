"""
protein_translation.py
======================
GeneForge Pipeline — Stage 2: DNA → Protein Translation

Responsibilities
----------------
- Accept a validated ``DNARecord`` from Stage 1 (dna_preprocessing)
- Translate the coding sequence into an amino acid sequence
- Support explicit reading frame selection (1, 2, 3) and automatic ORF detection
- Handle stop codons correctly: detect, locate, and optionally truncate
- Support all NCBI genetic code tables via Biopython
- Produce a ``ProteinRecord`` for consumption by Stage 3 (ESM-3 instability detection)

Data Flow
---------
    Input  : DNARecord  (from dna_preprocessing.py)
    Output : ProteinRecord (dataclass)

File Formats Produced
---------------------
    *.fasta  — protein FASTA (stop codon stripped) for ESM-3 / ProteinMPNN
    *.json   — translation metadata for pipeline provenance

Design Notes
------------
- A *reading frame* is the offset (0, 1, or 2 nucleotides) from the start of
  the sequence before the first full codon.  Frame 1 → offset 0 (canonical).
- Stop codons are represented as '*' in Biopython's translation output.
- ``to_stop=False`` is used deliberately so that internal stop codons are
  surfaced rather than silently dropped — they are a critical mutation signal.
- Automatic ORF detection scans all six frames (3 forward + 3 reverse
  complement) and returns the longest complete ORF (Met → stop).

Dependencies
------------
    biopython >= 1.83
    Python    >= 3.12
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, NamedTuple

from Bio.Seq import Seq
from Bio.Data import CodonTable

# Stage 1 contract — imported for type annotations and re-use of DNARecord.
# The import is guarded so this module remains importable in isolation
# (e.g. during unit testing with mocks).
try:
    from dna_preprocessing import DNARecord
except ModuleNotFoundError:  # pragma: no cover
    DNARecord = object  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Module logger
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: NCBI genetic code table 1 — the standard / universal code.
STANDARD_TABLE: Final[int] = 1

#: Minimum translated length (amino acids) considered biologically meaningful.
MIN_PROTEIN_LENGTH: Final[int] = 10

#: The stop codon character produced by Biopython translation.
STOP_SYMBOL: Final[str] = "*"

#: Sentinel used in FASTA output when the sequence ID is unavailable.
UNKNOWN_ID: Final[str] = "unknown"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class TranslationError(Exception):
    """Base exception for all protein translation failures."""


class InvalidReadingFrameError(TranslationError):
    """Raised when a reading frame outside {1, 2, 3} is requested."""


class InsufficientCodingSequenceError(TranslationError):
    """Raised when the CDS is too short to produce a meaningful protein."""


class CodonTableError(TranslationError):
    """Raised when the requested NCBI codon table ID is invalid."""


class NoORFFoundError(TranslationError):
    """Raised when auto-ORF detection fails to find any open reading frame."""


# ---------------------------------------------------------------------------
# Supporting types
# ---------------------------------------------------------------------------

class StopCodonInfo(NamedTuple):
    """Location and context of a stop codon within a translated sequence."""

    position: int        # 0-based index in the **amino acid** sequence
    codon: str           # Three-nucleotide codon (e.g. "TAA")
    is_terminal: bool    # True when this is the last character in the sequence


class ORFResult(NamedTuple):
    """Result from a single reading frame scan during auto-ORF detection."""

    strand: str          # "+" (forward) or "-" (reverse complement)
    frame: int           # 1, 2, or 3
    start_nt: int        # 0-based nt position in the *input* sequence
    protein: str         # Full translated sequence including terminal '*'
    is_complete: bool    # Starts with M and ends with '*'


# ---------------------------------------------------------------------------
# Data contract
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ProteinRecord:
    """
    Protein sequence produced by Stage 2, consumed by Stage 3.

    Attributes
    ----------
    sequence_id : str
        Carried from the source ``DNARecord``.
    sequence : str
        Full amino acid string including terminal stop ('*') if present.
        Use ``clean_sequence`` to obtain a stop-stripped form.
    source_dna_id : str
        ``sequence_id`` of the originating ``DNARecord``.
    codon_table_id : int
        NCBI genetic code table used for translation.
    strand : str
        "+" for forward (sense) strand; "-" for reverse complement.
    frame : int
        Reading frame used: 1 (offset 0), 2 (offset 1), or 3 (offset 2).
    cds_start_nt : int
        0-based nucleotide start position of the translated CDS in the
        normalised DNA sequence (accounts for frame offset).
    cds_end_nt : int
        0-based exclusive nucleotide end position of the CDS.
    length : int
        Total residue count including stop codon symbol if present.
    is_complete : bool
        True when the sequence begins with Met ('M') and ends with '*'.
    has_internal_stop : bool
        True if a stop codon appears anywhere before the final position.
    stop_codons : list[StopCodonInfo]
        All stop codons found, ordered by position.
    warnings : list[str]
        Non-fatal translation quality issues.
    """

    sequence_id: str
    sequence: str
    source_dna_id: str
    codon_table_id: int
    strand: str
    frame: int
    cds_start_nt: int
    cds_end_nt: int
    length: int
    is_complete: bool
    has_internal_stop: bool
    stop_codons: list[StopCodonInfo] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def clean_sequence(self) -> str:
        """
        Amino acid sequence with all stop codon symbols ('*') stripped.

        This is the form expected by ESM-3, RFdiffusion, and ProteinMPNN.
        """
        return self.sequence.replace(STOP_SYMBOL, "")

    @property
    def has_met_start(self) -> bool:
        """True when the protein begins with methionine."""
        return self.sequence.startswith("M")

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_fasta(self, line_width: int = 60, strip_stop: bool = True) -> str:
        """
        Return a FASTA-formatted protein string.

        Parameters
        ----------
        line_width : int
            Characters per wrapped line. Use 0 for a single-line sequence.
        strip_stop : bool
            Strip the terminal '*' character.  **Must be True** for ESM-3
            and ProteinMPNN, which do not accept stop codon symbols.
        """
        seq = self.clean_sequence if strip_stop else self.sequence
        if line_width > 0 and seq:
            wrapped = "\n".join(
                seq[i : i + line_width] for i in range(0, len(seq), line_width)
            )
        else:
            wrapped = seq
        return f">{self.sequence_id}\n{wrapped}\n"

    def to_dict(self) -> dict:
        return {
            "sequence_id": self.sequence_id,
            "source_dna_id": self.source_dna_id,
            "sequence": self.sequence,
            "clean_sequence": self.clean_sequence,
            "length": self.length,
            "codon_table_id": self.codon_table_id,
            "strand": self.strand,
            "frame": self.frame,
            "cds_start_nt": self.cds_start_nt,
            "cds_end_nt": self.cds_end_nt,
            "is_complete": self.is_complete,
            "has_met_start": self.has_met_start,
            "has_internal_stop": self.has_internal_stop,
            "stop_codons": [sc._asdict() for sc in self.stop_codons],
            "warnings": self.warnings,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_codon_table(table_id: int) -> CodonTable.CodonTable:
    """
    Retrieve and validate a Biopython NCBI codon table by integer ID.

    Parameters
    ----------
    table_id : int
        NCBI genetic code table number (e.g. 1 = standard, 2 = vertebrate
        mitochondrial, 11 = bacterial/plant plastid).

    Returns
    -------
    Bio.Data.CodonTable.CodonTable

    Raises
    ------
    CodonTableError
        If the table ID is not recognised by Biopython.
    """
    try:
        return CodonTable.unambiguous_dna_by_id[table_id]
    except KeyError:
        valid_ids = sorted(CodonTable.unambiguous_dna_by_id.keys())
        raise CodonTableError(
            f"NCBI codon table {table_id!r} is not recognised. "
            f"Valid table IDs: {valid_ids}"
        ) from None


def _validate_frame(frame: int) -> int:
    """
    Validate the requested reading frame value.

    Parameters
    ----------
    frame : int
        Must be 1, 2, or 3 (1-based, matching standard bioinformatics
        convention where frame 1 = no offset).

    Returns
    -------
    int
        Validated frame (unchanged).

    Raises
    ------
    InvalidReadingFrameError
    """
    if frame not in (1, 2, 3):
        raise InvalidReadingFrameError(
            f"Reading frame must be 1, 2, or 3; received {frame!r}. "
            "Frame 1 = no offset (default CDS start), "
            "frame 2 = skip 1 nucleotide, frame 3 = skip 2 nucleotides."
        )
    return frame


def _extract_cds(dna_sequence: str, frame: int) -> tuple[str, int, int]:
    """
    Extract the coding sequence from a normalised DNA string at the given frame.

    Trims trailing nucleotides that do not form a complete codon.

    Parameters
    ----------
    dna_sequence : str
        Normalised (uppercase, ACGTN) DNA sequence.
    frame : int
        Reading frame (1, 2, or 3).

    Returns
    -------
    tuple[str, int, int]
        ``(cds, start_nt, end_nt)`` where:
        - ``cds``      — the codon-trimmed CDS string
        - ``start_nt`` — 0-based start position in the original sequence
        - ``end_nt``   — 0-based exclusive end position

    Raises
    ------
    InsufficientCodingSequenceError
        If fewer than ``MIN_PROTEIN_LENGTH * 3`` nucleotides remain after
        applying the frame offset.
    """
    offset = frame - 1                     # frame 1 → offset 0
    cds_raw = dna_sequence[offset:]

    trim = len(cds_raw) % 3
    cds = cds_raw[: len(cds_raw) - trim] if trim else cds_raw

    start_nt = offset
    end_nt = start_nt + len(cds)

    if len(cds) < MIN_PROTEIN_LENGTH * 3:
        raise InsufficientCodingSequenceError(
            f"Only {len(cds)} nt remain after applying frame {frame} offset. "
            f"Minimum required: {MIN_PROTEIN_LENGTH * 3} nt "
            f"({MIN_PROTEIN_LENGTH} complete codons)."
        )

    return cds, start_nt, end_nt


def _translate_cds(
    cds: str,
    table_id: int,
    sequence_id: str,
) -> str:
    """
    Translate a codon-boundary-aligned CDS string to amino acids.

    Uses Biopython's ``Seq.translate()`` with ``to_stop=False`` so that
    internal stop codons appear as '*' rather than being silently dropped.

    Parameters
    ----------
    cds : str
        Normalised DNA string whose length is divisible by 3.
    table_id : int
        NCBI codon table ID.
    sequence_id : str
        For diagnostic messages.

    Returns
    -------
    str
        Amino acid sequence, potentially containing 'X' (ambiguous codon)
        and/or '*' (stop codon).

    Raises
    ------
    TranslationError
        If Biopython raises an unexpected error during translation.
    """
    try:
        protein = str(Seq(cds).translate(table=table_id, to_stop=False))
    except CodonTable.TranslationError as exc:
        raise TranslationError(
            f"Biopython translation failed for '{sequence_id}': {exc}"
        ) from exc
    except Exception as exc:
        raise TranslationError(
            f"Unexpected error translating '{sequence_id}': {type(exc).__name__}: {exc}"
        ) from exc

    return protein


def _find_stop_codons(
    protein: str,
    cds: str,
) -> list[StopCodonInfo]:
    """
    Locate all stop codon positions in a translated protein sequence.

    Parameters
    ----------
    protein : str
        Translated amino acid string (may contain '*').
    cds : str
        The nucleotide CDS string used to produce ``protein``.

    Returns
    -------
    list[StopCodonInfo]
        One entry per '*' character, ordered by position.
    """
    stops: list[StopCodonInfo] = []
    last_pos = len(protein) - 1

    for aa_pos, aa in enumerate(protein):
        if aa == STOP_SYMBOL:
            nt_start = aa_pos * 3
            codon = cds[nt_start : nt_start + 3] if nt_start + 3 <= len(cds) else "???"
            stops.append(StopCodonInfo(
                position=aa_pos,
                codon=codon,
                is_terminal=(aa_pos == last_pos),
            ))

    return stops


def _check_completeness(protein: str, sequence_id: str) -> tuple[bool, bool, list[str]]:
    """
    Assess biological completeness of a translated sequence.

    A *complete* protein:
      - begins with methionine ('M')  — canonical start codon
      - ends with a stop codon ('*') — confirmed termination

    Parameters
    ----------
    protein : str
        Full translated sequence.
    sequence_id : str
        For warning messages.

    Returns
    -------
    tuple[bool, bool, list[str]]
        ``(is_complete, has_internal_stop, warnings)``
    """
    warnings: list[str] = []

    has_internal_stop = STOP_SYMBOL in protein[:-1]

    starts_with_met = protein.startswith("M")
    ends_with_stop = protein.endswith(STOP_SYMBOL)
    is_complete = starts_with_met and ends_with_stop

    if not starts_with_met:
        warnings.append(
            f"'{sequence_id}': protein does not begin with Met (M). "
            "This may indicate a 5′-truncated CDS, an incorrect reading frame, "
            "or a non-AUG start codon."
        )

    if not ends_with_stop:
        warnings.append(
            f"'{sequence_id}': no terminal stop codon found. "
            "The CDS may be 3′-truncated or the reading frame is incorrect."
        )

    if has_internal_stop:
        # Count them for a more informative message
        n_internal = protein[:-1].count(STOP_SYMBOL)
        warnings.append(
            f"'{sequence_id}': {n_internal} internal stop codon(s) detected. "
            "Possible causes: frameshift mutation, incorrect reading frame, "
            "premature stop mutation, or pseudogene. "
            "These positions will be flagged as highly unstable by ESM-3."
        )

    if "X" in protein:
        n_x = protein.count("X")
        warnings.append(
            f"'{sequence_id}': {n_x} ambiguous residue(s) ('X') produced from "
            "codons containing 'N'. Structure prediction accuracy will be reduced "
            "at these positions."
        )

    return is_complete, has_internal_stop, warnings


# ---------------------------------------------------------------------------
# Single-frame translation (primary public function)
# ---------------------------------------------------------------------------

def translate_dna(
    dna_record: DNARecord,
    frame: int = 1,
    codon_table_id: int = STANDARD_TABLE,
    strand: str = "+",
) -> ProteinRecord:
    """
    Translate a validated ``DNARecord`` into a ``ProteinRecord``.

    This is the primary entry point for the pipeline when the reading frame
    is known in advance (e.g. the input is an annotated CDS).

    Parameters
    ----------
    dna_record : DNARecord
        Output from ``dna_preprocessing.preprocess_sequence`` or
        ``dna_preprocessing.load_fasta``.
    frame : int
        Reading frame: 1, 2, or 3.
        - Frame 1 → translation begins at nucleotide 0 (no offset).
        - Frame 2 → offset 1 nt (skips first nucleotide).
        - Frame 3 → offset 2 nt (skips first two nucleotides).
    codon_table_id : int
        NCBI genetic code table.  Defaults to 1 (standard code).
        Common alternatives: 2 (vertebrate mito), 11 (bacterial/plastid).
    strand : str
        "+" (default) to translate the forward strand as-is.
        "-" to reverse-complement before translation.

    Returns
    -------
    ProteinRecord

    Raises
    ------
    InvalidReadingFrameError
        If ``frame`` is not 1, 2, or 3.
    CodonTableError
        If ``codon_table_id`` is not a recognised NCBI table number.
    InsufficientCodingSequenceError
        If the remaining CDS after frame-offset is too short.
    TranslationError
        If Biopython raises during translation.
    ValueError
        If ``strand`` is not "+" or "-".
    """
    if strand not in ("+", "-"):
        raise ValueError(f"strand must be '+' or '-'; received {strand!r}")

    _validate_frame(frame)
    _validate_codon_table(codon_table_id)

    seq_id = dna_record.sequence_id
    dna_seq = dna_record.sequence

    # ------------------------------------------------------------------ #
    # Reverse complement for minus-strand translation                      #
    # ------------------------------------------------------------------ #
    if strand == "-":
        dna_seq = str(Seq(dna_seq).reverse_complement())
        logger.debug("'%s': using reverse complement for minus-strand translation.", seq_id)

    # ------------------------------------------------------------------ #
    # Extract coding sequence at the requested frame                       #
    # ------------------------------------------------------------------ #
    cds, cds_start_nt, cds_end_nt = _extract_cds(dna_seq, frame)

    logger.debug(
        "'%s': CDS extracted — frame=%d, strand=%s, offset=%d, length=%d nt.",
        seq_id, frame, strand, frame - 1, len(cds),
    )

    # ------------------------------------------------------------------ #
    # Warn if frame-trimming discarded nucleotides                         #
    # ------------------------------------------------------------------ #
    extra_warnings: list[str] = list(dna_record.warnings)  # inherit Stage 1 warnings
    trim = (len(dna_seq) - (frame - 1)) % 3
    if trim:
        extra_warnings.append(
            f"'{seq_id}': {trim} trailing nucleotide(s) discarded to align to "
            f"reading frame {frame}. These nucleotides are not translated."
        )

    # ------------------------------------------------------------------ #
    # Translate                                                            #
    # ------------------------------------------------------------------ #
    protein = _translate_cds(cds, codon_table_id, seq_id)

    # ------------------------------------------------------------------ #
    # Analyse result                                                       #
    # ------------------------------------------------------------------ #
    stop_codons = _find_stop_codons(protein, cds)
    is_complete, has_internal_stop, completeness_warnings = _check_completeness(
        protein, seq_id
    )
    all_warnings = extra_warnings + completeness_warnings

    record = ProteinRecord(
        sequence_id=seq_id,
        sequence=protein,
        source_dna_id=seq_id,
        codon_table_id=codon_table_id,
        strand=strand,
        frame=frame,
        cds_start_nt=cds_start_nt,
        cds_end_nt=cds_end_nt,
        length=len(protein),
        is_complete=is_complete,
        has_internal_stop=has_internal_stop,
        stop_codons=stop_codons,
        warnings=all_warnings,
    )

    logger.info(
        "Translated '%s': %d aa, frame=%d, strand=%s, complete=%s, "
        "internal_stops=%s, warnings=%d.",
        seq_id, record.length, frame, strand, is_complete,
        has_internal_stop, len(all_warnings),
    )
    return record


# ---------------------------------------------------------------------------
# All-frame ORF scanner
# ---------------------------------------------------------------------------

def _scan_all_frames(
    dna_sequence: str,
    codon_table_id: int,
    sequence_id: str,
) -> list[ORFResult]:
    """
    Translate all six reading frames (3 forward + 3 reverse complement).

    Internal helper for ``find_best_orf``.  Returns raw frame results
    including incomplete ORFs; ranking/selection is performed by the caller.

    Parameters
    ----------
    dna_sequence : str
        Normalised DNA string.
    codon_table_id : int
        NCBI codon table ID.
    sequence_id : str

    Returns
    -------
    list[ORFResult]
        Up to six entries (one per frame/strand combination).
    """
    results: list[ORFResult] = []
    strands: list[tuple[str, str]] = [
        ("+", dna_sequence),
        ("-", str(Seq(dna_sequence).reverse_complement())),
    ]

    for strand_label, seq in strands:
        for frame in (1, 2, 3):
            try:
                cds, start_nt, _ = _extract_cds(seq, frame)
            except InsufficientCodingSequenceError:
                logger.debug(
                    "'%s': frame %s%d skipped — insufficient CDS length.",
                    sequence_id, strand_label, frame,
                )
                continue

            try:
                protein = _translate_cds(cds, codon_table_id, sequence_id)
            except TranslationError as exc:
                logger.warning(
                    "'%s': frame %s%d translation error: %s",
                    sequence_id, strand_label, frame, exc,
                )
                continue

            is_complete = protein.startswith("M") and protein.endswith(STOP_SYMBOL)
            results.append(ORFResult(
                strand=strand_label,
                frame=frame,
                start_nt=start_nt,
                protein=protein,
                is_complete=is_complete,
            ))

    return results


def _longest_internal_orf(protein: str) -> str:
    """
    Extract the longest Met-initiated sub-ORF from a full-frame translation.

    When the best frame is not complete (no leading Met or no terminal stop),
    we search within the translated sequence for the longest Met-to-stop
    sub-sequence, which represents the most likely embedded ORF.

    Parameters
    ----------
    protein : str
        Full translated sequence (may contain '*').

    Returns
    -------
    str
        Longest Met-initiated sub-ORF including its stop codon, or the
        original ``protein`` if no Met is found.
    """
    best = ""
    search_start = 0

    while True:
        met_pos = protein.find("M", search_start)
        if met_pos == -1:
            break

        # Find the next stop codon from Met onward
        stop_pos = protein.find(STOP_SYMBOL, met_pos)
        if stop_pos == -1:
            # No stop codon — take sequence to end
            candidate = protein[met_pos:]
        else:
            candidate = protein[met_pos : stop_pos + 1]

        if len(candidate) > len(best):
            best = candidate

        search_start = met_pos + 1

    return best if best else protein


def find_best_orf(
    dna_record: DNARecord,
    codon_table_id: int = STANDARD_TABLE,
    require_start_codon: bool = True,
) -> ProteinRecord:
    """
    Automatically detect and translate the best open reading frame.

    Searches all six frames (3 forward + 3 reverse complement) and selects
    the highest-quality ORF using the following priority order:

    1. Complete ORFs (Met start + stop codon) with no internal stops — longest first
    2. Complete ORFs with internal stops — longest first (degraded but usable)
    3. Incomplete ORFs (missing Met or stop) — longest first
    4. Any translated frame — as a last resort

    When ``require_start_codon=True`` (default), the function additionally
    extracts the longest Met-initiated sub-ORF from the winning frame if
    the translation does not already begin with Met.

    Parameters
    ----------
    dna_record : DNARecord
        Validated record from Stage 1.
    codon_table_id : int
        NCBI genetic code table.
    require_start_codon : bool
        If True, attempt to extract the longest sub-ORF starting at a Met
        when the top-ranked frame does not begin with one.

    Returns
    -------
    ProteinRecord
        Best ORF as a fully annotated record.

    Raises
    ------
    CodonTableError
        If ``codon_table_id`` is not recognised.
    NoORFFoundError
        If no translatable frame exists (e.g. sequence is all-N).
    """
    _validate_codon_table(codon_table_id)
    seq_id = dna_record.sequence_id

    logger.info(
        "'%s': scanning all 6 reading frames for best ORF (table=%d).",
        seq_id, codon_table_id,
    )

    all_frames = _scan_all_frames(dna_record.sequence, codon_table_id, seq_id)

    if not all_frames:
        raise NoORFFoundError(
            f"No translatable reading frame found in '{seq_id}'. "
            "Check that the sequence contains valid coding nucleotides (not all-N)."
        )

    # ------------------------------------------------------------------ #
    # Rank frames                                                          #
    # ------------------------------------------------------------------ #
    def _sort_key(r: ORFResult) -> tuple[int, int, int]:
        has_no_internal_stop = int(STOP_SYMBOL not in r.protein[:-1])
        return (int(r.is_complete), has_no_internal_stop, len(r.protein))

    ranked = sorted(all_frames, key=_sort_key, reverse=True)
    best_frame = ranked[0]

    logger.info(
        "'%s': best frame selected — strand=%s, frame=%d, %d aa, complete=%s.",
        seq_id, best_frame.strand, best_frame.frame,
        len(best_frame.protein), best_frame.is_complete,
    )

    # ------------------------------------------------------------------ #
    # Sub-ORF extraction (if needed)                                       #
    # ------------------------------------------------------------------ #
    protein_seq = best_frame.protein
    if require_start_codon and not protein_seq.startswith("M"):
        sub_orf = _longest_internal_orf(protein_seq)
        if sub_orf != protein_seq:
            logger.info(
                "'%s': extracted longest Met-initiated sub-ORF (%d aa → %d aa).",
                seq_id, len(protein_seq), len(sub_orf),
            )
            protein_seq = sub_orf

    # ------------------------------------------------------------------ #
    # Re-derive CDS boundaries (approximate for RC strand)                #
    # ------------------------------------------------------------------ #
    frame = best_frame.frame
    strand = best_frame.strand
    offset = frame - 1

    if strand == "+":
        cds_start = offset
        cds_end = cds_start + len(protein_seq) * 3
    else:
        # Positions are in the reverse-complement coordinate space
        rc_len = len(dna_record.sequence)
        cds_start = rc_len - offset - len(protein_seq) * 3
        cds_end = rc_len - offset

    # ------------------------------------------------------------------ #
    # Analyse and build record                                             #
    # ------------------------------------------------------------------ #
    cds_for_stops = dna_record.sequence[cds_start:cds_end] if strand == "+" else (
        str(Seq(dna_record.sequence).reverse_complement())[offset : offset + len(protein_seq) * 3]
    )
    stop_codons = _find_stop_codons(protein_seq, cds_for_stops)
    is_complete, has_internal_stop, completeness_warnings = _check_completeness(
        protein_seq, seq_id
    )

    extra_warnings = list(dna_record.warnings) + [
        f"Reading frame auto-detected: strand={strand}, frame={frame}. "
        "Verify this matches the expected gene annotation."
    ] + completeness_warnings

    record = ProteinRecord(
        sequence_id=seq_id,
        sequence=protein_seq,
        source_dna_id=seq_id,
        codon_table_id=codon_table_id,
        strand=strand,
        frame=frame,
        cds_start_nt=max(0, cds_start),
        cds_end_nt=max(0, cds_end),
        length=len(protein_seq),
        is_complete=is_complete,
        has_internal_stop=has_internal_stop,
        stop_codons=stop_codons,
        warnings=extra_warnings,
    )

    logger.info(
        "Auto-ORF '%s': %d aa, strand=%s, frame=%d, complete=%s.",
        seq_id, record.length, strand, frame, is_complete,
    )
    return record


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def write_protein_fasta(
    record: ProteinRecord,
    output_path: str | Path,
    line_width: int = 60,
    strip_stop: bool = True,
    overwrite: bool = False,
) -> Path:
    """
    Write a ``ProteinRecord`` to a FASTA file.

    Parameters
    ----------
    record : ProteinRecord
    output_path : str | Path
    line_width : int
        Residues per wrapped line. Default 60.
    strip_stop : bool
        Strip the terminal '*'. **Must be True** for ESM-3 / ProteinMPNN
        compatibility. Default True.
    overwrite : bool
        Overwrite existing file if True.

    Returns
    -------
    Path
        Resolved output path.

    Raises
    ------
    FileExistsError
        If the file exists and ``overwrite=False``.
    """
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output file '{output_path}' already exists. "
            "Pass overwrite=True to replace it."
        )

    fasta_str = record.to_fasta(line_width=line_width, strip_stop=strip_stop)
    output_path.write_text(fasta_str, encoding="utf-8")

    logger.info(
        "Protein FASTA written to '%s' (%d aa, stop_stripped=%s).",
        output_path, len(record.clean_sequence), strip_stop,
    )
    return output_path


def write_translation_report(
    record: ProteinRecord,
    output_path: str | Path,
    overwrite: bool = False,
) -> Path:
    """
    Write translation metadata to a JSON file for pipeline provenance.

    Parameters
    ----------
    record : ProteinRecord
    output_path : str | Path
    overwrite : bool

    Returns
    -------
    Path

    Raises
    ------
    FileExistsError
        If the file exists and ``overwrite=False``.
    """
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Translation report '{output_path}' already exists. "
            "Pass overwrite=True to replace it."
        )

    output_path.write_text(record.to_json(), encoding="utf-8")

    logger.info(
        "Translation report written to '%s'.", output_path
    )
    return output_path


def load_protein_fasta(path: str | Path) -> list[ProteinRecord]:
    """
    Load protein sequences from a FASTA file into minimal ``ProteinRecord`` objects.

    Metadata fields not encoded in FASTA (frame, codon table, CDS bounds) are
    set to sentinel values.  Use this only for re-entering the pipeline from
    pre-existing protein FASTA files (e.g. from AlphaFold or UniProt).

    Parameters
    ----------
    path : str | Path
        Path to protein FASTA file.

    Returns
    -------
    list[ProteinRecord]

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file is empty or contains no sequences.
    """
    from Bio import SeqIO  # local import — Bio already declared at module level

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Protein FASTA not found: {path}")

    records: list[ProteinRecord] = []

    for sr in SeqIO.parse(str(path), "fasta"):
        seq = str(sr.seq)
        has_internal = STOP_SYMBOL in seq[:-1]
        is_complete = seq.startswith("M") and seq.endswith(STOP_SYMBOL)
        stop_codons = _find_stop_codons(seq, "")

        records.append(ProteinRecord(
            sequence_id=sr.id,
            sequence=seq,
            source_dna_id=UNKNOWN_ID,
            codon_table_id=0,    # unknown — not stored in FASTA
            strand="+",
            frame=0,             # unknown
            cds_start_nt=0,
            cds_end_nt=0,
            length=len(seq),
            is_complete=is_complete,
            has_internal_stop=has_internal,
            stop_codons=stop_codons,
            warnings=["Record loaded from FASTA; frame, strand, and CDS bounds unknown."],
        ))

    if not records:
        raise ValueError(f"No protein sequences found in '{path}'.")

    logger.info("Loaded %d protein record(s) from '%s'.", len(records), path)
    return records
