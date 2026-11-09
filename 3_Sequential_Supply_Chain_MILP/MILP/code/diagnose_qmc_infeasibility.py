#!/usr/bin/env python3
"""Audit GAMS/CPLEX logs from the Sobol QMC experiment.

The script does four things:

1. Pairs ``*_cplex.log`` files with their ``*_cplex_solve_report.txt`` files.
2. Separates confirmed infeasibility from time limits, solver errors,
   infeasible-or-unbounded outcomes, optimal solves, and unknown outcomes.
3. Extracts named constraints and variable bounds from CPLEX conflict-refiner
   (IIS), presolve, and GAMS ``INFES`` diagnostics when those are present.
4. Quantifies the evidenced operability mechanisms without inferring a cause
   from a status-only log.

Place this file in ``MILP/code`` and run it without arguments, or
provide ``--project-root``/``--logs-dir`` explicitly. Output is written to
``results/stochastic_optimisation_solution_space/infeasibility_diagnostics``.

Examples
--------
From ``MILP/code``::

    python diagnose_qmc_infeasibility.py

With explicit paths::

    python diagnose_qmc_infeasibility.py \
        --logs-dir ../results/stochastic_optimisation_solution_space/gams_logs \
        --expected-infeasible 91

If existing logs contain only solve status, use the generated
``cplex_infeasibility_diagnostics.opt`` for targeted reruns, retain the GAMS
listing or conflict report, and run this script again.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Pattern


TEXT_EXTENSIONS = {".log", ".txt", ".lst", ".out"}

# These rules are deliberately applied only to named conflict/IIS members or
# explicit diagnostic lines. They are never applied to the full solver log,
# because doing so could convert a descriptive word such as "capacity" into an
# unsupported causal conclusion.
DEFAULT_CAUSE_RULES: dict[str, tuple[str, ...]] = {
    "plastic treatment or recycling capacity": (
        r"(?:plastic|polymer|recycl|tpe|ldpe|lldpe|evoh|polyolefin).*(?:cap|capacity)",
        r"(?:cap|capacity).*(?:plastic|polymer|recycl|tpe|ldpe|lldpe|evoh|polyolefin)",
        r"(?:recycling_capacity|route_capacity|treatment_capacity|cap_recycl)",
    ),
    "plastic material balance or compulsory routing": (
        r"(?:plastic|polymer|sut).*(?:balance|generation|routing|flow|demand)",
        r"(?:balance|routing|flow).*(?:plastic|polymer|sut)",
        r"(?:polymer_balance|plastic_balance|mass_balance)",
    ),
    "route eligibility or treatment availability": (
        r"(?:route|treatment).*(?:elig|avail|access|allowed|feasible)",
        r"(?:elig|avail|access|allowed).*(?:route|treatment)",
        r"(?:feasible_arc|route_eligibility|route_available)",
    ),
    "wastewater treatment or reuse capacity": (
        r"(?:wastewater|waste_water|ww|reuse|treatment_unit).*(?:cap|capacity|limit)",
        r"(?:cap|capacity|limit).*(?:wastewater|waste_water|reuse|treatment_unit)",
        r"(?:ww_capacity|reuse_capacity|wastewater_capacity)",
    ),
    "water availability or water balance": (
        r"(?:water|wfi|mains).*(?:avail|balance|demand|supply|shortfall)",
        r"(?:avail|balance|demand|supply|shortfall).*(?:water|wfi|mains)",
        r"(?:water_balance|water_availability|mains_capacity)",
    ),
    "electricity supply or renewable availability": (
        r"(?:electric|power|grid|wind|renew).*(?:avail|balance|demand|supply|cap|limit)",
        r"(?:avail|balance|demand|supply|cap|limit).*(?:electric|power|grid|wind|renew)",
        r"(?:electricity_balance|renewable_capacity|power_balance)",
    ),
    "transport payload, journeys, or route access": (
        r"(?:transport|hgv|payload|journey|journeys|trip|vehicle).*(?:cap|limit|avail|balance|required)",
        r"(?:cap|limit|avail|balance|required).*(?:transport|hgv|payload|journey|trip|vehicle)",
        r"(?:payload_constraint|journey_balance|transport_capacity)",
    ),
    "manufacturing production or batch allocation": (
        r"(?:batch|production|manufactur|site).*(?:cap|balance|demand|allocation|target|limit)",
        r"(?:cap|balance|demand|allocation|target|limit).*(?:batch|production|manufactur|site)",
        r"(?:production_balance|batch_allocation|demand_satisfaction)",
    ),
    "biomass treatment availability or capacity": (
        r"(?:biomass|anaerobic|digestion|ad_route).*(?:cap|avail|balance|limit)",
        r"(?:cap|avail|balance|limit).*(?:biomass|anaerobic|digestion|ad_route)",
    ),
}

MODEL_STATUS_RE = re.compile(
    r"\bMODEL\s+STATUS\s*(?::|=)?\s*(?P<value>[^\r\n]+)", re.IGNORECASE
)
SOLVER_STATUS_RE = re.compile(
    r"\bSOLVER\s+STATUS\s*(?::|=)?\s*(?P<value>[^\r\n]+)", re.IGNORECASE
)
TERMINATION_RE = re.compile(
    r"\btermination(?:_condition|\s+condition)?\s*(?::|=)\s*(?P<value>[^\r\n,;}]+)",
    re.IGNORECASE,
)

EXPLICIT_INFEASIBLE_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bmodel\s+status\b[^\r\n]*\binfeasible\b",
        r"\btermination(?:_condition|\s+condition)?\b[^\r\n]*\binfeasible\b",
        r"\binteger\s+infeasible\b",
        r"\bmip\s+status\s*\([^)]*\)\s*:\s*[^\r\n]*\binfeasible\b",
        r"\bproblem\s+(?:is|was|has\s+been)\s+(?:integer\s+)?infeasible\b",
        r"\bmodel\s+(?:is|was)\s+(?:integer\s+)?infeasible\b",
        r"\binfeasible\s+no\s+solution\b",
    )
)

AMBIGUOUS_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\binfeasible\s+or\s+unbounded\b",
        r"\binfeasible[- ]or[- ]unbounded\b",
    )
)

TIME_LIMIT_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\btime\s+limit\b[^\r\n]*(?:reached|exceeded|termination|interrupt)",
        r"\btimelimit\b[^\r\n]*(?:reached|exceeded|termination|interrupt)",
        r"\bresource\s+interrupt\b",
        r"\btermination[^\r\n]*(?:max(?:imum)?\s+time|time)",
    )
)

SOLVER_ERROR_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bsolver\s+error\b",
        r"\blicen[cs]e\s+(?:error|failure|not\s+found)\b",
        r"\bexecution\s+error\b",
        r"\btraceback\b",
        r"\bfatal\s+error\b",
    )
)

OPTIMAL_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bmodel\s+status\b[^\r\n]*\b(?:optimal|integer\s+solution)\b",
        r"\btermination(?:_condition|\s+condition)?\b[^\r\n]*\boptimal\b",
        r"\bmip\s*-?\s*integer\s+optimal\s+solution\b",
        r"\binteger\s+optimal\s+solution\b",
    )
)

DIAGNOSTIC_LINE_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bconflict\s+refiner\b",
        r"\bconflict\s+report\b",
        r"\birreducibly\s+inconsistent\b",
        r"\biis\b",
        r"\binfeasible\s+(?:row|constraint|equation|bound)\b",
        r"\b(?:row|constraint|equation|bound)\b[^\r\n]*\binfeasible\b",
        r"\bpossible\s+member\b",
        r"\bmember\b[^\r\n]*\bconflict\b",
        r"\bINFES\b",
    )
)

MEMBER_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b(?:row|equation|constraint)\s+(?:name\s*)?['\"]?(?P<name>[A-Za-z_][A-Za-z0-9_.$]*(?:\([^\r\n)]*\)|\[[^\r\n]]*\])?)",
        r"\b(?:lower|upper)\s+bound\s+(?:on|of)\s+(?:variable\s+)?['\"]?(?P<name>[A-Za-z_][A-Za-z0-9_.$]*(?:\([^\r\n)]*\)|\[[^\r\n]]*\])?)",
        r"\b(?:member|possible\s+member)\s*(?::|=)\s*['\"]?(?P<name>[A-Za-z_][A-Za-z0-9_.$]*(?:\([^\r\n)]*\)|\[[^\r\n]]*\])?)",
        r"^\s*(?P<name>[A-Za-z_][A-Za-z0-9_.$]*(?:\([^\r\n)]*\)|\[[^\r\n]]*\])?)\s+.*\bINFES\b",
    )
)

NAME_BLACKLIST = {
    "a",
    "all",
    "bound",
    "conflict",
    "constraint",
    "cplex",
    "equation",
    "iis",
    "infeasible",
    "member",
    "model",
    "of",
    "is",
    "possible",
    "problem",
    "row",
    "status",
    "the",
    "variable",
}


@dataclass
class CaseDiagnostic:
    case_id: str
    sample_id: str
    sequence_number: int | None
    files: list[str]
    status: str
    model_status: str
    solver_status: str
    termination_condition: str
    evidence_level: str
    primary_mechanism: str
    mechanism_families: list[str] = field(default_factory=list)
    conflict_members: list[str] = field(default_factory=list)
    status_evidence: list[str] = field(default_factory=list)
    diagnostic_evidence: list[str] = field(default_factory=list)
    truncated_files: list[str] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Identify and quantify evidenced causes of QMC infeasibility."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        help="Path to MILP. Defaults to the parent of the script's code directory.",
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        help="Directory containing GAMS/CPLEX logs and solve reports.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for diagnostic CSV, JSON, Markdown, and rerun files.",
    )
    parser.add_argument(
        "--expected-infeasible",
        type=int,
        default=91,
        help="Expected confirmed-infeasible count; set below zero to disable the check (default: 91).",
    )
    parser.add_argument(
        "--strict-count",
        action="store_true",
        help="Exit with code 2 if the confirmed-infeasible count differs from the expected count.",
    )
    parser.add_argument(
        "--rules-json",
        type=Path,
        help="Optional JSON mapping of mechanism labels to regex pattern lists.",
    )
    parser.add_argument(
        "--max-bytes-per-file",
        type=int,
        default=20_000_000,
        help="Maximum bytes read per file; oversized files are read from both ends (default: 20 MB).",
    )
    parser.add_argument(
        "--include-preflight",
        action="store_true",
        help="Include files whose names contain 'preflight'.",
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    script_path = Path(__file__).resolve()
    project_root = (
        args.project_root.expanduser().resolve()
        if args.project_root
        else script_path.parent.parent.resolve()
    )
    logs_dir = (
        args.logs_dir.expanduser().resolve()
        if args.logs_dir
        else project_root
        / "results"
        / "stochastic_optimisation_solution_space"
        / "gams_logs"
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else project_root
        / "results"
        / "stochastic_optimisation_solution_space"
        / "infeasibility_diagnostics"
    )
    return project_root, logs_dir, output_dir


def natural_key(value: str) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def normalise_case_id(path: Path) -> str:
    name = path.name
    for extension in sorted(TEXT_EXTENSIONS, key=len, reverse=True):
        if name.lower().endswith(extension):
            name = name[: -len(extension)]
            break
    suffixes = (
        "_cplex_solve_report",
        "_solve_report",
        "_cplex_report",
        "_cplex",
        "_gams_listing",
        "_listing",
    )
    lowered = name.lower()
    for suffix in suffixes:
        if lowered.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name


def discover_case_files(logs_dir: Path, include_preflight: bool) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in logs_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_EXTENSIONS:
            continue
        if not include_preflight and "preflight" in path.name.lower():
            continue
        case_id = normalise_case_id(path)
        grouped[case_id].append(path)
    return {
        case_id: sorted(paths, key=lambda item: natural_key(item.name))
        for case_id, paths in sorted(grouped.items(), key=lambda item: natural_key(item[0]))
    }


def read_text_head_tail(path: Path, max_bytes: int) -> tuple[str, bool]:
    size = path.stat().st_size
    if size <= max_bytes:
        return path.read_text(encoding="utf-8", errors="replace"), False
    half = max_bytes // 2
    with path.open("rb") as handle:
        head = handle.read(half)
        handle.seek(max(0, size - half))
        tail = handle.read(half)
    marker = b"\n\n--- MIDDLE OF OVERSIZED LOG OMITTED ---\n\n"
    return (head + marker + tail).decode("utf-8", errors="replace"), True


def unique_preserving_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        cleaned = " ".join(value.strip().split())
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            output.append(cleaned)
    return output


def matching_lines(text: str, patterns: Iterable[Pattern[str]]) -> list[str]:
    return unique_preserving_order(
        line for line in text.splitlines() if any(pattern.search(line) for pattern in patterns)
    )


def last_group_match(pattern: Pattern[str], text: str) -> str:
    matches = list(pattern.finditer(text))
    return " ".join(matches[-1].group("value").strip().split()) if matches else ""


def determine_status(text: str) -> tuple[str, list[str]]:
    lines = text.splitlines()
    ambiguous_lines = [
        line for line in lines if any(pattern.search(line) for pattern in AMBIGUOUS_PATTERNS)
    ]
    infeasible_lines = [
        line
        for line in lines
        if any(pattern.search(line) for pattern in EXPLICIT_INFEASIBLE_PATTERNS)
        and not any(pattern.search(line) for pattern in AMBIGUOUS_PATTERNS)
    ]
    time_lines = [line for line in lines if any(pattern.search(line) for pattern in TIME_LIMIT_PATTERNS)]
    error_lines = [line for line in lines if any(pattern.search(line) for pattern in SOLVER_ERROR_PATTERNS)]
    optimal_lines = [line for line in lines if any(pattern.search(line) for pattern in OPTIMAL_PATTERNS)]

    if infeasible_lines:
        return "infeasible", unique_preserving_order(infeasible_lines)
    if ambiguous_lines:
        return "infeasible_or_unbounded", unique_preserving_order(ambiguous_lines)
    if error_lines:
        return "solver_error", unique_preserving_order(error_lines)
    if time_lines:
        return "time_limit_or_interrupt", unique_preserving_order(time_lines)
    if optimal_lines:
        return "optimal_or_feasible", unique_preserving_order(optimal_lines)
    return "unknown", []


def clean_member_name(value: str) -> str:
    return value.strip().strip("'\"`:,;.")


def extract_diagnostics(text: str) -> tuple[list[str], list[str], str]:
    evidence_lines = matching_lines(text, DIAGNOSTIC_LINE_PATTERNS)
    members: list[str] = []
    for line in evidence_lines:
        for pattern in MEMBER_PATTERNS:
            match = pattern.search(line)
            if not match:
                continue
            name = clean_member_name(match.group("name"))
            if name.lower() not in NAME_BLACKLIST and len(name) > 1:
                members.append(name)

    members = unique_preserving_order(members)
    lowered_evidence = "\n".join(evidence_lines).lower()
    if members and ("conflict" in lowered_evidence or "iis" in lowered_evidence):
        level = "conflict_refiner_or_iis"
    elif members:
        level = "named_presolve_or_infes_row"
    elif evidence_lines:
        level = "diagnostic_text_without_named_member"
    else:
        level = "status_only"
    return members, evidence_lines, level


def load_rules(path: Path | None) -> dict[str, tuple[Pattern[str], ...]]:
    raw_rules: dict[str, Iterable[str]]
    if path is None:
        raw_rules = DEFAULT_CAUSE_RULES
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("--rules-json must contain a JSON object")
        raw_rules = {}
        for label, patterns in payload.items():
            if not isinstance(label, str) or not isinstance(patterns, list) or not all(
                isinstance(item, str) for item in patterns
            ):
                raise ValueError("Each rules JSON value must be a list of regex strings")
            raw_rules[label] = patterns
    return {
        label: tuple(re.compile(pattern, re.IGNORECASE) for pattern in patterns)
        for label, patterns in raw_rules.items()
    }


def classify_mechanisms(
    members: list[str], diagnostic_lines: list[str], rules: dict[str, tuple[Pattern[str], ...]]
) -> list[str]:
    # Classification is restricted to diagnostic evidence. Status-only cases
    # therefore remain unresolved by construction.
    evidence = "\n".join([*members, *diagnostic_lines])
    if not evidence:
        return []
    return [
        label
        for label, patterns in rules.items()
        if any(pattern.search(evidence) for pattern in patterns)
    ]


def infer_sample_fields(case_id: str) -> tuple[str, int | None]:
    sample_match = re.search(r"(SO_[A-Za-z]+_\d+)", case_id, flags=re.IGNORECASE)
    sample_id = sample_match.group(1) if sample_match else case_id
    sequence_match = re.match(r"(\d+)_", case_id)
    sequence_number = int(sequence_match.group(1)) if sequence_match else None
    return sample_id, sequence_number


def diagnose_case(
    case_id: str,
    paths: list[Path],
    max_bytes: int,
    rules: dict[str, tuple[Pattern[str], ...]],
) -> CaseDiagnostic:
    texts: list[str] = []
    truncated: list[str] = []
    for path in paths:
        text, was_truncated = read_text_head_tail(path, max_bytes)
        texts.append(f"\n===== FILE: {path.name} =====\n{text}")
        if was_truncated:
            truncated.append(str(path))
    combined = "\n".join(texts)

    status, status_evidence = determine_status(combined)
    members, diagnostic_evidence, evidence_level = extract_diagnostics(combined)
    mechanisms = classify_mechanisms(members, diagnostic_evidence, rules)

    if status != "infeasible":
        primary = "not a confirmed infeasible solve"
    elif len(mechanisms) == 1:
        primary = mechanisms[0]
    elif len(mechanisms) > 1:
        primary = "joint conflict: " + " + ".join(mechanisms)
    elif evidence_level == "status_only":
        primary = "unresolved: infeasible status only; rerun conflict refiner"
    else:
        primary = "unresolved: diagnostic present but no configured family matched"

    sample_id, sequence_number = infer_sample_fields(case_id)
    return CaseDiagnostic(
        case_id=case_id,
        sample_id=sample_id,
        sequence_number=sequence_number,
        files=[str(path) for path in paths],
        status=status,
        model_status=last_group_match(MODEL_STATUS_RE, combined),
        solver_status=last_group_match(SOLVER_STATUS_RE, combined),
        termination_condition=last_group_match(TERMINATION_RE, combined),
        evidence_level=evidence_level,
        primary_mechanism=primary,
        mechanism_families=mechanisms,
        conflict_members=members,
        status_evidence=status_evidence,
        diagnostic_evidence=diagnostic_evidence,
        truncated_files=truncated,
    )


def serialise_list(values: Iterable[str]) -> str:
    return " | ".join(values)


def write_case_csv(path: Path, diagnostics: list[CaseDiagnostic]) -> None:
    columns = (
        "case_id",
        "sample_id",
        "sequence_number",
        "status",
        "model_status",
        "solver_status",
        "termination_condition",
        "evidence_level",
        "primary_mechanism",
        "mechanism_families",
        "conflict_member_count",
        "conflict_members",
        "status_evidence",
        "diagnostic_evidence",
        "files",
        "truncated_files",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for item in diagnostics:
            writer.writerow(
                {
                    "case_id": item.case_id,
                    "sample_id": item.sample_id,
                    "sequence_number": "" if item.sequence_number is None else item.sequence_number,
                    "status": item.status,
                    "model_status": item.model_status,
                    "solver_status": item.solver_status,
                    "termination_condition": item.termination_condition,
                    "evidence_level": item.evidence_level,
                    "primary_mechanism": item.primary_mechanism,
                    "mechanism_families": serialise_list(item.mechanism_families),
                    "conflict_member_count": len(item.conflict_members),
                    "conflict_members": serialise_list(item.conflict_members),
                    "status_evidence": serialise_list(item.status_evidence),
                    "diagnostic_evidence": serialise_list(item.diagnostic_evidence),
                    "files": serialise_list(item.files),
                    "truncated_files": serialise_list(item.truncated_files),
                }
            )


def write_count_csv(path: Path, label_name: str, counter: Counter[str], denominator: int) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=(label_name, "case_count", "percent_of_infeasible"))
        writer.writeheader()
        for label, count in sorted(counter.items(), key=lambda item: (-item[1], item[0])):
            writer.writerow(
                {
                    label_name: label,
                    "case_count": count,
                    "percent_of_infeasible": round(100.0 * count / denominator, 3)
                    if denominator
                    else 0.0,
                }
            )




def build_summary(diagnostics: list[CaseDiagnostic], expected: int) -> dict[str, object]:
    status_counts = Counter(item.status for item in diagnostics)
    infeasible = [item for item in diagnostics if item.status == "infeasible"]
    unresolved = [item for item in infeasible if item.primary_mechanism.startswith("unresolved:")]
    primary_counts = Counter(item.primary_mechanism for item in infeasible)
    family_counts = Counter(
        family for item in infeasible for family in set(item.mechanism_families)
    )
    return {
        "case_bundle_count": len(diagnostics),
        "confirmed_infeasible_count": len(infeasible),
        "expected_infeasible_count": expected,
        "expected_count_matched": expected < 0 or len(infeasible) == expected,
        "unresolved_infeasible_count": len(unresolved),
        "status_counts": dict(status_counts),
        "primary_mechanism_counts": dict(primary_counts),
        "conflict_family_incidence": dict(family_counts),
    }


def write_outputs(
    output_dir: Path,
    logs_dir: Path,
    diagnostics: list[CaseDiagnostic],
    expected: int,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    infeasible = [item for item in diagnostics if item.status == "infeasible"]
    unresolved = [item for item in infeasible if item.primary_mechanism.startswith("unresolved:")]

    write_case_csv(output_dir / "all_case_diagnostics.csv", diagnostics)
    write_case_csv(output_dir / "infeasible_case_diagnostics.csv", infeasible)
    write_case_csv(output_dir / "unresolved_infeasible_cases.csv", unresolved)

    primary_counts = Counter(item.primary_mechanism for item in infeasible)
    family_counts = Counter(
        family for item in infeasible for family in set(item.mechanism_families)
    )
    member_counts = Counter(
        member for item in infeasible for member in set(item.conflict_members)
    )
    write_count_csv(
        output_dir / "primary_mechanism_counts.csv",
        "primary_mechanism",
        primary_counts,
        len(infeasible),
    )
    write_count_csv(
        output_dir / "conflict_family_incidence.csv",
        "constraint_family",
        family_counts,
        len(infeasible),
    )
    write_count_csv(
        output_dir / "conflict_member_counts.csv",
        "conflict_member",
        member_counts,
        len(infeasible),
    )

    summary = build_summary(diagnostics, expected)
    (output_dir / "diagnostic_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "rerun_case_ids.txt").write_text(
        "\n".join(item.case_id for item in unresolved) + ("\n" if unresolved else ""),
        encoding="utf-8",
    )
    (output_dir / "cplex_infeasibility_diagnostics.opt").write_text(
        "* CPLEX conflict diagnostics for a targeted infeasible-case rerun\n"
        "iis 1\n"
        "conflictdisplay 2\n"
        "names yes\n"
        "rerun yes\n",
        encoding="utf-8",
    )
    (output_dir / "cause_rules_template.json").write_text(
        json.dumps(DEFAULT_CAUSE_RULES, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "case_diagnostics.json").write_text(
        json.dumps([asdict(item) for item in diagnostics], indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    args = parse_args()
    _, logs_dir, output_dir = resolve_paths(args)
    if not logs_dir.is_dir():
        print(f"ERROR: logs directory does not exist: {logs_dir}", file=sys.stderr)
        return 1
    if args.max_bytes_per_file < 1_000:
        print("ERROR: --max-bytes-per-file must be at least 1000", file=sys.stderr)
        return 1

    try:
        rules = load_rules(args.rules_json)
    except (OSError, ValueError, json.JSONDecodeError, re.error) as exc:
        print(f"ERROR: invalid cause rules: {exc}", file=sys.stderr)
        return 1

    grouped = discover_case_files(logs_dir, args.include_preflight)
    if not grouped:
        print(f"ERROR: no .log/.txt/.lst/.out files found under {logs_dir}", file=sys.stderr)
        return 1

    diagnostics = [
        diagnose_case(case_id, paths, args.max_bytes_per_file, rules)
        for case_id, paths in grouped.items()
    ]
    summary = write_outputs(
        output_dir,
        logs_dir,
        diagnostics,
        args.expected_infeasible,
    )

    confirmed = int(summary["confirmed_infeasible_count"])
    unresolved = int(summary["unresolved_infeasible_count"])
    print(f"Inspected case bundles: {len(diagnostics)}")
    print(f"Confirmed infeasible: {confirmed}")
    print(f"Unresolved mechanisms: {unresolved}")
    print(f"Case details: {output_dir / 'infeasible_case_diagnostics.csv'}")

    expected_ok = bool(summary["expected_count_matched"])
    if args.strict_count and not expected_ok:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
