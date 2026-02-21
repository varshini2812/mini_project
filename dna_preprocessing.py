"""
dna_preprocessing.py
====================
GeneForge Pipeline — Stage 1: DNA Preprocessing

Responsibilities
----------------
- Accept mutated DNA as a raw string or FASTA file
- Validate nucleotide composition
- Normalize sequences (whitespace, case, IUPAC ambiguity codes)
- Detect open reading frame (ORF) quality issues
- Call mutations (SNPs + indels) relative to an optional reference sequence
- Produce a canonical DNARecord for downstream translation

Data Flow
---------
    Input  : Raw DNA string | FASTA file path
    Output : DNARecord (dataclass) + optional JSON mutation report

File Formats Produced
---------------------
    *.fasta  — canonical DNA sequence for downstream stages
    *.json   — mutation report / preprocessing metadata

Dependencies
------------
    biopython >= 1.83
    Python    >= 3.12
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import Align

# ---------------------------------------------------------------------------
# Module logger
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Nucleotides accepted in a validated sequence after normalization.
VALID_BASES: Final[frozenset[str]] = frozenset("ACGTN")

#: IUPAC ambiguity codes collapsed to N during normalization.
#: Full IUPAC table: https://www.bioinformatics.org/sms/iupac.html
IUPAC_TO_N: Final[dict[str, str]] = {
    "R": "N", "Y": "N", "S": "N", "W": "N",
    "K": "N", "M": "N", "B": "N", "D": "N",
    "H": "N", "V": "N",
}

#: Translation table mapping lowercase + IUPAC → uppercase / N.
_NORM_TABLE: Final[str] = str.maketrans(
    "acgtryswkmbdhvn" + "RYSWKMBDHV",
    "ACGT" + "N" * 11 + "N" * 10,
)

#: Minimum sequence length (nucleotides) to be considered translatable.
MIN_CDS_LENGTH: Final[int] = 3


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class DNAPreprocessingError(Exception):
    """Base exception for all DNA preprocessing failures."""


class InvalidSequenceError(DNAPreprocessingError):
    """Raised when a sequence contains unrecognisable characters."""


class EmptySequenceError(DNAPreprocessingError):
    """Raised when a sequence is empty or becomes empty after cleaning."""


class FASTAParsingError(DNAPreprocessingError):
    """Raised when FASTA parsing fails or the file is empty."""


# ---------------------------------------------------------------------------
# Data Contracts
# ---------------------------------------------------------------------------

@dataclass(frozen=False, slots=True)
class MutationRecord:
    """
    Represents a single detected mutation relative to a reference sequence.

    Attributes
    ----------
    position : int
        0-based nucleotide position in the **reference** coordinate space.
    ref_base : str
        Nucleotide(s) in the reference at this position ("-" for insertions).
    alt_base : str
        Nucleotide(s) in the query at this position ("-" for deletions).
    mutation_type : str
        One of "SNP", "insertion", or "deletion".
    """

    position: int
    ref_base: str
    alt_base: str
    mutation_type: str

    def to_dict(self) -> dict[str, str | int]:
        return {
            "position": self.position,
            "ref_base": self.ref_base,
            "alt_base": self.alt_base,
            "mutation_type": self.mutation_type,
        }

    def __repr__(self) -> str:
        return (
            f"MutationRecord(pos={self.position}, "
            f"{self.ref_base}→{self.alt_base}, type={self.mutation_type})"
        )


@dataclass(slots=True)
class DNARecord:
    """
    Canonical DNA record produced by Stage 1 and consumed by Stage 2.

    Attributes
    ----------
    sequence_id : str
        Identifier sourced from the FASTA header or supplied by the caller.
    sequence : str
        Uppercase, whitespace-free, IUPAC-normalised DNA (only A/C/G/T/N).
    original_sequence : str
        Verbatim input sequence before any modification.
    length : int
        Length of the normalised sequence in nucleotides.
    gc_content : float
        GC fraction in [0.0, 1.0].  0.0 for sequences composed entirely of N.
    in_frame : bool
        True when ``length % 3 == 0``; indicates a complete codon set.
    has_ambiguous_bases : bool
        True when any N is present in the normalised sequence.
    mutations : list[MutationRecord]
        Mutations detected relative to a reference.  Empty if no reference
        was supplied.
    warnings : list[str]
        Non-fatal quality issues discovered during preprocessing.
    """

    sequence_id: str
    sequence: str
    original_sequence: str
    length: int
    gc_content: float
    in_frame: bool
    has_ambiguous_bases: bool
    mutations: list[MutationRecord] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_fasta(self, line_width: int = 60) -> str:
        """
        Return a FASTA-formatted string for this record.

        Parameters
        ----------
        line_width : int
            Characters per sequence line.  Use 0 or a negative value to
            emit the sequence on a single line.
        """
        if line_width > 0:
            wrapped = "\n".join(
                self.sequence[i : i + line_width]
                for i in range(0, self.length, line_width)
            )
        else:
            wrapped = self.sequence
        return f">{self.sequence_id}\n{wrapped}\n"

    def to_dict(self) -> dict:
        return {
            "sequence_id": self.sequence_id,
            "sequence": self.sequence,
            "length": self.length,
            "gc_content": round(self.gc_content, 6),
            "in_frame": self.in_frame,
            "has_ambiguous_bases": self.has_ambiguous_bases,
            "mutations": [m.to_dict() for m in self.mutations],
            "warnings": self.warnings,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalize_sequence(raw: str) -> str:
    """
    Normalize a raw DNA string into a clean, uppercase sequence.

    Steps
    -----
    1. Strip all whitespace (spaces, tabs, newlines inside sequences).
    2. Uppercase everything.
    3. Map IUPAC ambiguity codes → N via a translation table.

    Parameters
    ----------
    raw : str
        Raw nucleotide string, possibly mixed-case with whitespace.

    Returns
    -------
    str
        Uppercase nucleotide string containing only A, C, G, T, and N.
    """
    cleaned = re.sub(r"\s+", "", raw)
    return cleaned.translate(_NORM_TABLE)


def _validate_characters(sequence: str, sequence_id: str) -> None:
    """
    Raise ``InvalidSequenceError`` if any character outside VALID_BASES is present.

    Parameters
    ----------
    sequence : str
        Already-normalised (uppercase, whitespace-free) DNA string.
    sequence_id : str
        Used in the error message for diagnostics.

    Raises
    ------
    InvalidSequenceError
    """
    invalid = sorted(set(sequence) - VALID_BASES)
    if invalid:
        raise InvalidSequenceError(
            f"Sequence '{sequence_id}' contains unrecognised character(s) "
            f"after normalisation: {invalid}.  "
            "Accepted characters are A, C, G, T, and N (IUPAC ambiguity codes "
            "are collapsed to N automatically)."
        )


def _compute_gc_content(sequence: str) -> float:
    """
    Return GC fraction as a float in [0.0, 1.0].

    Ambiguous bases (N) are excluded from both numerator and denominator.
    Returns 0.0 for sequences composed entirely of N or for empty strings.

    Parameters
    ----------
    sequence : str
        Normalised DNA string.
    """
    definite_bases = len(sequence) - sequence.count("N")
    if definite_bases == 0:
        return 0.0
    gc = sequence.count("G") + sequence.count("C")
    return gc / definite_bases


def _call_snps(reference: str, query: str) -> list[MutationRecord]:
    """
    Call single-nucleotide polymorphisms for equal-length sequences.

    This is an O(L) scan used when reference and query have the same length
    (i.e. no indels are expected).

    Parameters
    ----------
    reference : str
        Normalised reference DNA sequence.
    query : str
        Normalised mutated query sequence (same length as reference).

    Returns
    -------
    list[MutationRecord]
    """
    mutations: list[MutationRecord] = []
    for pos, (ref_base, alt_base) in enumerate(zip(reference, query)):
        if ref_base != alt_base:
            mutations.append(
                MutationRecord(
                    position=pos,
                    ref_base=ref_base,
                    alt_base=alt_base,
                    mutation_type="SNP",
                )
            )
    return mutations


def _call_mutations_with_alignment(
    reference: str,
    query: str,
    sequence_id: str,
) -> list[MutationRecord]:
    """
    Call SNPs and indels via global pairwise alignment (Needleman–Wunsch).

    Used when reference and query differ in length, indicating the presence
    of insertions or deletions.  Biopython's ``PairwiseAligner`` is used
    with affine gap penalties appropriate for genomic sequences.

    Parameters
    ----------
    reference : str
        Normalised reference DNA.
    query : str
        Normalised mutated query DNA.
    sequence_id : str
        Identifier for logging.

    Returns
    -------
    list[MutationRecord]
    """
    aligner = Align.PairwiseAligner(mode="global")
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -3
    aligner.extend_gap_score = -0.5

    # Top alignment only — iterating is memory-safe for long sequences
    alignments = aligner.align(reference, query)

    try:
        best = next(iter(alignments))
    except StopIteration:
        logger.warning(
            "No alignment produced for '%s'; returning empty mutation list.",
            sequence_id,
        )
        return []

    # Extract aligned strings
    aligned_ref: str = best[0]
    aligned_qry: str = best[1]

    mutations: list[MutationRecord] = []
    ref_pos = 0

    for ref_char, qry_char in zip(aligned_ref, aligned_qry):
        if ref_char == "-" and qry_char != "-":
            mutations.append(
                MutationRecord(
                    position=ref_pos,
                    ref_base="-",
                    alt_base=qry_char,
                    mutation_type="insertion",
                )
            )
            # Insertions do not advance the reference coordinate
        elif qry_char == "-" and ref_char != "-":
            mutations.append(
                MutationRecord(
                    position=ref_pos,
                    ref_base=ref_char,
                    alt_base="-",
                    mutation_type="deletion",
                )
            )
            ref_pos += 1
        else:
            if ref_char != qry_char:
                mutations.append(
                    MutationRecord(
                        position=ref_pos,
                        ref_base=ref_char,
                        alt_base=qry_char,
                        mutation_type="SNP",
                    )
                )
            ref_pos += 1

    logger.debug(
        "Alignment-based mutation calling for '%s': %d mutations found.",
        sequence_id,
        len(mutations),
    )
    return mutations


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def preprocess_sequence(
    raw_sequence: str,
    sequence_id: str = "sequence",
    reference_sequence: str | None = None,
) -> DNARecord:
    """
    Validate, clean, and annotate a single DNA sequence.

    This is the primary entry point when the caller supplies a raw string
    rather than a FASTA file.

    Parameters
    ----------
    raw_sequence : str
        Input DNA — may contain whitespace, lowercase letters, and IUPAC
        ambiguity codes (R, Y, S, W, K, M, B, D, H, V).
    sequence_id : str
        Human-readable identifier for this sequence.  Defaults to
        ``"sequence"`` if not supplied.
    reference_sequence : str | None
        Optional reference DNA (same gene, wild-type).  When provided,
        mutations are called by comparing query against reference.

    Returns
    -------
    DNARecord
        Fully annotated record ready for Stage 2 (translation).

    Raises
    ------
    EmptySequenceError
        If the input is empty or reduces to an empty string after cleaning.
    InvalidSequenceError
        If the normalised sequence contains characters outside A/C/G/T/N.
    DNAPreprocessingError
        For any other preprocessing failure.
    """
    warnings: list[str] = []

    # ------------------------------------------------------------------ #
    # 1. Normalize                                                         #
    # ------------------------------------------------------------------ #
    if not raw_sequence or not raw_sequence.strip():
        raise EmptySequenceError(
            f"Input sequence '{sequence_id}' is empty or contains only whitespace."
        )

    normalized = _normalize_sequence(raw_sequence)

    if not normalized:
        raise EmptySequenceError(
            f"Sequence '{sequence_id}' is empty after whitespace removal."
        )

    # ------------------------------------------------------------------ #
    # 2. Validate characters                                               #
    # ------------------------------------------------------------------ #
    _validate_characters(normalized, sequence_id)

    # ------------------------------------------------------------------ #
    # 3. Detect and warn about ambiguous bases                             #
    # ------------------------------------------------------------------ #
    n_count = normalized.count("N")
    has_ambiguous = n_count > 0
    if has_ambiguous:
        warnings.append(
            f"{n_count} ambiguous nucleotide(s) (IUPAC codes) were collapsed to "
            "'N'.  Codons containing N will translate to 'X' (unknown amino acid)."
        )

    # ------------------------------------------------------------------ #
    # 4. Frame check                                                       #
    # ------------------------------------------------------------------ #
    remainder = len(normalized) % 3
    in_frame = remainder == 0
    if not in_frame:
        warnings.append(
            f"Sequence length {len(normalized)} nt is not divisible by 3 "
            f"(remainder {remainder}).  This will produce an incomplete final "
            "codon during translation.  Consider trimming or padding before use."
        )

    # ------------------------------------------------------------------ #
    # 5. Minimum length                                                    #
    # ------------------------------------------------------------------ #
    if len(normalized) < MIN_CDS_LENGTH:
        raise DNAPreprocessingError(
            f"Sequence '{sequence_id}' is too short to encode a single codon "
            f"(length={len(normalized)} nt; minimum={MIN_CDS_LENGTH} nt)."
        )

    # ------------------------------------------------------------------ #
    # 6. GC content                                                        #
    # ------------------------------------------------------------------ #
    gc = _compute_gc_content(normalized)

    # Biologically implausible GC extremes → warn but do not reject
    if gc < 0.20:
        warnings.append(
            f"GC content is very low ({gc:.1%}).  This may indicate a non-coding "
            "region, a highly AT-rich organism, or a data quality issue."
        )
    elif gc > 0.80:
        warnings.append(
            f"GC content is very high ({gc:.1%}).  Codon optimisation may be "
            "required for heterologous expression."
        )

    # ------------------------------------------------------------------ #
    # 7. Mutation calling (optional)                                       #
    # ------------------------------------------------------------------ #
    mutations: list[MutationRecord] = []
    if reference_sequence is not None:
        ref_normalized = _normalize_sequence(reference_sequence)
        _validate_characters(ref_normalized, f"{sequence_id}:reference")

        if len(ref_normalized) == len(normalized):
            mutations = _call_snps(ref_normalized, normalized)
        else:
            logger.info(
                "Reference length (%d) ≠ query length (%d) for '%s'. "
                "Using pairwise alignment for indel-aware mutation calling.",
                len(ref_normalized),
                len(normalized),
                sequence_id,
            )
            mutations = _call_mutations_with_alignment(
                ref_normalized, normalized, sequence_id
            )

    # ------------------------------------------------------------------ #
    # 8. Build record                                                      #
    # ------------------------------------------------------------------ #
    record = DNARecord(
        sequence_id=sequence_id,
        sequence=normalized,
        original_sequence=raw_sequence,
        length=len(normalized),
        gc_content=gc,
        in_frame=in_frame,
        has_ambiguous_bases=has_ambiguous,
        mutations=mutations,
        warnings=warnings,
    )

    logger.info(
        "Preprocessed '%s': length=%d nt, GC=%.1f%%, in_frame=%s, "
        "ambiguous=%s, mutations=%d, warnings=%d",
        sequence_id,
        record.length,
        gc * 100,
        in_frame,
        has_ambiguous,
        len(mutations),
        len(warnings),
    )
    return record


def load_fasta(
    path: str | Path,
    reference_sequence: str | None = None,
) -> list[DNARecord]:
    """
    Parse a FASTA file and preprocess each entry into a ``DNARecord``.

    Parameters
    ----------
    path : str | Path
        Path to the input FASTA file.  Must be readable.
    reference_sequence : str | None
        Optional reference DNA string used for mutation calling.  Applied
        uniformly to every sequence in the file.  Pass ``None`` to skip
        mutation calling.

    Returns
    -------
    list[DNARecord]
        One record per FASTA entry, in file order.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    FASTAParsingError
        If the file is empty, contains no parseable sequences, or cannot
        be read due to format errors.
    DNAPreprocessingError
        If any individual sequence fails validation (propagated from
        ``preprocess_sequence``).
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"FASTA file not found: {path}")
    if not path.is_file():
        raise FASTAParsingError(f"Path is not a regular file: {path}")
    if path.stat().st_size == 0:
        raise FASTAParsingError(f"FASTA file is empty: {path}")

    records: list[DNARecord] = []

    try:
        for seq_record in SeqIO.parse(str(path), "fasta"):
            raw_seq = str(seq_record.seq)
            seq_id = seq_record.id

            logger.debug("Parsing FASTA entry: %s", seq_id)
            dna_record = preprocess_sequence(
                raw_sequence=raw_seq,
                sequence_id=seq_id,
                reference_sequence=reference_sequence,
            )
            records.append(dna_record)

    except (OSError, UnicodeDecodeError) as exc:
        raise FASTAParsingError(
            f"Failed to read FASTA file '{path}': {exc}"
        ) from exc

    if not records:
        raise FASTAParsingError(
            f"No parseable sequences found in '{path}'.  "
            "Ensure the file uses standard FASTA format with '>' header lines."
        )

    logger.info("Loaded %d sequence(s) from '%s'.", len(records), path)
    return records


def write_fasta(
    record: DNARecord,
    output_path: str | Path,
    line_width: int = 60,
    overwrite: bool = False,
) -> Path:
    """
    Write a single ``DNARecord`` to a FASTA file.

    Parameters
    ----------
    record : DNARecord
    output_path : str | Path
        Destination file path.  Parent directories are created automatically.
    line_width : int
        Nucleotides per line.  Defaults to 60 (GenBank convention).
    overwrite : bool
        If False (default) and the file already exists, raises ``FileExistsError``.

    Returns
    -------
    Path
        Resolved path of the written file.

    Raises
    ------
    FileExistsError
        If the output file exists and ``overwrite=False``.
    """
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output FASTA '{output_path}' already exists.  "
            "Pass overwrite=True to replace it."
        )

    fasta_str = record.to_fasta(line_width=line_width)
    output_path.write_text(fasta_str, encoding="utf-8")

    logger.info("FASTA written to '%s' (%d nt).", output_path, record.length)
    return output_path


def write_mutation_report(
    record: DNARecord,
    output_path: str | Path,
    overwrite: bool = False,
) -> Path:
    """
    Serialise a ``DNARecord``'s metadata and mutation list to a JSON file.

    Parameters
    ----------
    record : DNARecord
    output_path : str | Path
    overwrite : bool

    Returns
    -------
    Path
        Resolved path of the written JSON file.

    Raises
    ------
    FileExistsError
        If the file exists and ``overwrite=False``.
    """
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Mutation report '{output_path}' already exists.  "
            "Pass overwrite=True to replace it."
        )

    output_path.write_text(record.to_json(), encoding="utf-8")

    logger.info(
        "Mutation report written to '%s' (%d mutations).",
        output_path,
        len(record.mutations),
    )
    return output_path


def dna_record_from_biopython(
    seq_record: SeqRecord,
    reference_sequence: str | None = None,
) -> DNARecord:
    """
    Convenience wrapper: convert a Biopython ``SeqRecord`` into a ``DNARecord``.

    Useful when the caller has already performed FASTA parsing elsewhere in
    a larger Biopython-based workflow.

    Parameters
    ----------
    seq_record : Bio.SeqRecord.SeqRecord
    reference_sequence : str | None

    Returns
    -------
    DNARecord
    """
    return preprocess_sequence(
        raw_sequence=str(seq_record.seq),
        sequence_id=seq_record.id,
        reference_sequence=reference_sequence,
    )
