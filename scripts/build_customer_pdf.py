#!/usr/bin/env python3
"""Build and verify the customer field-guide PDF from canonical HTML.

The PDF is a derivative, not benchmark evidence.  This helper stamps the exact
source commit and HTML digest into the rendered pages, writes a hash sidecar,
and can later prove that the checked PDF still corresponds to the current HTML.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "docs/customer/benchmark-your-own-endpoint.html"
DEFAULT_OUTPUT = ROOT / "docs/customer/output/benchmark-your-own-endpoint.pdf"
STAMP = (
    "UNSTAMPED SOURCE - build the distributable PDF with "
    "scripts/build_customer_pdf.py")
COMMIT_RE = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")
HASH_RE = re.compile(r"[0-9a-f]{64}")
EXPECTED_TITLE = "Benchmark your own endpoint"
LETTER_SIZE_RE = re.compile(
    r"612(?:\.0+)?\s+x\s+792(?:\.0+)?\s+pts\s+\(letter\)", re.I)
SEMANTIC_QA_VERSION = 3
GEOMETRY_QA_VERSION = 1
MIN_FOOTER_CLEARANCE_POINTS = 8.0
SEMANTIC_PAGE_REQUIREMENTS = (
    (
        # The page-one kicker is positioned between the two visual h1 lines in
        # Poppler's reading order, so verify both visible title fragments.
        "Turn a traffic profile into evidence you can",
        "defend.",
        "BENCHMARK YOUR OWN ENDPOINT",
        "The evidence contract",
    ),
    (
        "Prove mechanics with minimal, quota-planned traffic.",
        "Published Enterprise P2T defaults: tier and",
        "headroom not verified",
        "not measured customer demand",
    ),
    (
        "Exercise the endpoint contract without making a timing claim.",
        "correctness smoke only; no performance or capacity verdict.",
        "non-refusal visible",
        "content or a valid non-refusal tool call",
    ),
    (
        "Run the customer contract, then separate SLA from capacity.",
        "GUARDED RATE SWEEP",
        "capacity stays inconclusive",
        "Interchunk is a stream-gap diagnostic",
    ),
    (
        "Read independent gates before quoting any number.",
        "Read five decisions, not one green badge",
        "Create a separate run-verification receipt",
        "RUN_DIR=results/customer-fixed-rate/RUN-DIRECTORY",
        "${RUN_DIR}-verification",
    ),
)
SEMANTIC_REQUIREMENTS_SHA256 = hashlib.sha256(json.dumps(
    {
        "version": SEMANTIC_QA_VERSION,
        "title": EXPECTED_TITLE,
        "pages": SEMANTIC_PAGE_REQUIREMENTS,
    },
    ensure_ascii=True,
    separators=(",", ":"),
).encode("utf-8")).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str, text: bool = True) -> str | bytes:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, check=False)
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", "replace").strip()
        raise RuntimeError(
            f"git {' '.join(args)} failed: {detail or result.returncode}")
    return (result.stdout.decode("utf-8", "strict") if text
            else result.stdout)


def _metadata_path(output: Path) -> Path:
    return output.with_suffix(output.suffix + ".metadata.json")


def _atomic_bytes(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _source_commit_time(commit: str) -> str:
    epoch = int(str(_git("show", "-s", "--format=%ct", commit)).strip())
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat().replace(
        "+00:00", "Z")


def _required_tool(name: str) -> str:
    executable = shutil.which(name)
    if executable is None:
        raise RuntimeError(f"{name} is required to build and verify the PDF")
    return executable


def _tool_version(executable: str, argument: str) -> str:
    result = subprocess.run(
        [executable, argument], capture_output=True, text=True, check=False)
    combined = "\n".join(
        value.strip() for value in (result.stdout, result.stderr)
        if value.strip())
    if result.returncode != 0 or not combined:
        raise RuntimeError(
            f"could not identify {Path(executable).name}: "
            f"{combined or f'exit {result.returncode}'}")
    return combined.splitlines()[0]


def _run_text_tool(command: list[str], name: str) -> str:
    result = subprocess.run(
        command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(
            f"{name} failed while inspecting the PDF: "
            f"{detail or f'exit {result.returncode}'}")
    return result.stdout


def _normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _pdfinfo_fields(raw: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in raw.splitlines():
        key, separator, value = line.partition(":")
        if separator:
            fields[key.strip()] = value.strip()
    return fields


def _inspect_bbox_geometry(raw: str, expected_count: int) -> dict:
    """Reject clipped text and body/footer collisions from Poppler geometry."""
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as exc:
        raise RuntimeError("pdftotext returned invalid bbox XHTML") from exc

    def local(element) -> str:
        return element.tag.rsplit("}", 1)[-1]

    pages = [element for element in root.iter() if local(element) == "page"]
    if len(pages) != expected_count:
        raise RuntimeError(
            f"PDF bbox extraction found {len(pages)} pages; expected exactly "
            f"{expected_count}")
    clearances = []
    tolerance = 0.25
    for index, page in enumerate(pages, 1):
        try:
            width = float(page.attrib["width"])
            height = float(page.attrib["height"])
        except (KeyError, ValueError) as exc:
            raise RuntimeError(
                f"PDF page {index} has invalid bbox dimensions") from exc
        if not all(math.isfinite(value) and value > 0
                   for value in (width, height)):
            raise RuntimeError(
                f"PDF page {index} has non-finite bbox dimensions")

        lines = []
        words_seen = 0
        for line in (item for item in page.iter() if local(item) == "line"):
            words = [item for item in line.iter() if local(item) == "word"]
            if not words:
                continue
            words_seen += len(words)
            text = " ".join("".join(word.itertext()).strip()
                            for word in words).strip()
            try:
                bounds = tuple(float(line.attrib[name]) for name in (
                    "xMin", "yMin", "xMax", "yMax"))
            except (KeyError, ValueError) as exc:
                raise RuntimeError(
                    f"PDF page {index} has an invalid text bbox") from exc
            x_min, y_min, x_max, y_max = bounds
            if not all(math.isfinite(value) for value in bounds) \
                    or x_min < -tolerance or y_min < -tolerance \
                    or x_max > width + tolerance \
                    or y_max > height + tolerance \
                    or x_min >= x_max or y_min >= y_max:
                raise RuntimeError(
                    f"PDF page {index} contains clipped or invalid text "
                    f"geometry: {text!r}")
            lines.append((text, y_min, y_max))
        if not words_seen:
            raise RuntimeError(f"PDF page {index} has no bounded visible text")

        marker = f"{index:02d} / {expected_count:02d}"
        footer_lines = [line for line in lines if marker in line[0]]
        if len(footer_lines) != 1:
            raise RuntimeError(
                f"PDF page {index} bbox needs exactly one footer marker "
                f"{marker!r}")
        footer_y = footer_lines[0][1]
        if footer_y < height * 0.90:
            raise RuntimeError(
                f"PDF page {index} footer is not in the bottom page region")
        body_lines = [line for line in lines
                      if abs(line[1] - footer_y) > tolerance]
        if not body_lines:
            raise RuntimeError(f"PDF page {index} has no body text")
        clearance = footer_y - max(line[2] for line in body_lines)
        if clearance < MIN_FOOTER_CLEARANCE_POINTS:
            raise RuntimeError(
                f"PDF page {index} body/footer clearance is {clearance:.2f} "
                f"points; expected at least "
                f"{MIN_FOOTER_CLEARANCE_POINTS:.2f}")
        clearances.append(clearance)
    return {
        "geometry_qa_version": GEOMETRY_QA_VERSION,
        "minimum_footer_clearance_points": round(min(clearances), 3),
    }


def _inspect_pdf(
        pdf: Path, *, source_sha: str, source_commit: str) -> dict:
    """Fail closed unless the rendered PDF retains the five-page contract."""
    if not pdf.read_bytes().startswith(b"%PDF-"):
        raise RuntimeError("rendered output is not a PDF")

    pdfinfo = _required_tool("pdfinfo")
    pdftotext = _required_tool("pdftotext")
    info = _pdfinfo_fields(_run_text_tool(
        [pdfinfo, str(pdf)], "pdfinfo"))
    try:
        page_count = int(info.get("Pages", ""))
    except ValueError as exc:
        raise RuntimeError("pdfinfo returned an invalid page count") from exc
    expected_count = len(SEMANTIC_PAGE_REQUIREMENTS)
    if page_count != expected_count:
        raise RuntimeError(
            f"PDF has {page_count} pages; expected exactly {expected_count}")
    if info.get("Title") != EXPECTED_TITLE:
        raise RuntimeError(
            f"PDF title is {info.get('Title')!r}; expected {EXPECTED_TITLE!r}")
    if not LETTER_SIZE_RE.fullmatch(info.get("Page size", "")):
        raise RuntimeError(
            f"PDF page size is not US Letter: {info.get('Page size')!r}")
    if info.get("Encrypted") != "no" or info.get("JavaScript") != "no":
        raise RuntimeError("PDF must be unencrypted and contain no JavaScript")
    if not info.get("Creator") or not info.get("Producer"):
        raise RuntimeError("PDF creator/producer provenance is missing")

    extracted = _run_text_tool(
        [pdftotext, "-layout", str(pdf), "-"], "pdftotext")
    pages = extracted.split("\f")
    if pages and not pages[-1].strip():
        pages.pop()
    if len(pages) != expected_count:
        raise RuntimeError(
            "PDF text extraction found "
            f"{len(pages)} pages; expected exactly {expected_count}")

    for index, (page, requirements) in enumerate(zip(
            pages, SEMANTIC_PAGE_REQUIREMENTS, strict=True), 1):
        visible = _normalized(page)
        marker = f"{index:02d} / {expected_count:02d}"
        if marker not in visible:
            raise RuntimeError(
                f"PDF page {index} is missing page marker {marker!r}")
        missing = [phrase for phrase in requirements if phrase not in visible]
        if missing:
            raise RuntimeError(
                f"PDF page {index} is missing required visible text: "
                + "; ".join(repr(value) for value in missing))

    visible_all = _normalized(extracted)
    if "UNSTAMPED" in visible_all:
        raise RuntimeError("PDF still contains the UNSTAMPED source marker")
    if re.search(r"\b0[1-3]\s*/\s*03\b", visible_all):
        raise RuntimeError("PDF retains a stale three-page footer marker")
    visible_compact = re.sub(r"\s+", "", extracted)
    for label, token in (
            ("source HTML digest", source_sha),
            ("source Git commit", source_commit)):
        if token not in visible_compact:
            raise RuntimeError(f"PDF does not visibly retain its {label}")

    bbox = _run_text_tool(
        [pdftotext, "-bbox-layout", str(pdf), "-"],
        "pdftotext bbox geometry")
    geometry = _inspect_bbox_geometry(bbox, expected_count)

    return {
        "pdf_page_count": page_count,
        "pdf_title": info["Title"],
        "pdf_creator": info.get("Creator", ""),
        "pdf_producer": info.get("Producer", ""),
        "pdf_page_size": info["Page size"],
        "pdf_encrypted": False,
        "pdf_javascript": False,
        "pdfinfo_version": _tool_version(pdfinfo, "-v"),
        "pdftotext_version": _tool_version(pdftotext, "-v"),
        "semantic_qa_version": SEMANTIC_QA_VERSION,
        "semantic_requirements_sha256": SEMANTIC_REQUIREMENTS_SHA256,
        **geometry,
    }


def _rendered_stamp(
        commit: str, source_sha: str, commit_time: str, *, dirty: bool) -> str:
    return (
        f"Source commit {commit}{' (DIRTY QA RENDER)' if dirty else ''} · "
        f"canonical HTML SHA-256 {source_sha} · "
        f"source commit time {commit_time}")


def build(source: Path, output: Path, *, allow_dirty: bool = False) -> dict:
    source = source.resolve(strict=True)
    if source != DEFAULT_SOURCE.resolve(strict=True):
        raise ValueError(
            "the distributable field guide must use the canonical repo HTML")
    commit = str(_git("rev-parse", "HEAD")).strip().lower()
    if not COMMIT_RE.fullmatch(commit) or set(commit) == {"0"}:
        raise RuntimeError("Git HEAD is not a valid commit digest")
    status = str(_git(
        "status", "--porcelain=v1", "--untracked-files=all")).strip()
    dirty = bool(status)
    if dirty and not allow_dirty:
        raise RuntimeError(
            "refusing to publish a PDF from a dirty tree; commit the canonical "
            "source first or use --allow-dirty for a non-distributable QA render")

    source_raw = source.read_bytes()
    source_sha = hashlib.sha256(source_raw).hexdigest()
    source_text = source_raw.decode("utf-8")
    if source_text.count(STAMP) != 1:
        raise RuntimeError("canonical HTML has no unique PDF source-stamp marker")
    commit_time = _source_commit_time(commit)
    rendered_stamp = _rendered_stamp(
        commit, source_sha, commit_time, dirty=dirty)
    rendered = source_text.replace(STAMP, rendered_stamp)

    playwright = _required_tool("playwright")
    playwright_version = _tool_version(playwright, "--version")

    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="customer-pdf-build-") as temp:
        temp_dir = Path(temp)
        stamped_html = temp_dir / source.name
        stamped_html.write_text(rendered, encoding="utf-8")
        temp_pdf = temp_dir / output.name
        subprocess.run(
            [playwright, "pdf", stamped_html.as_uri(), str(temp_pdf)],
            cwd=ROOT, check=True)
        pdf_raw = temp_pdf.read_bytes()
        semantic = _inspect_pdf(
            temp_pdf, source_sha=source_sha, source_commit=commit)
        _atomic_bytes(output, pdf_raw)

    generated = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    metadata = {
        "metadata_schema_version": 2,
        "derivative_type": "customer_field_guide_pdf",
        "evidence_status": "unsealed_navigation_derivative",
        "source_html": source.relative_to(ROOT).as_posix(),
        "source_html_sha256": source_sha,
        "source_git_commit": commit,
        "source_git_dirty": dirty,
        "source_commit_time_utc": commit_time,
        "generated_at_utc": generated,
        "renderer": playwright_version,
        "renderer_command": "playwright pdf",
        "pdf_sha256": hashlib.sha256(pdf_raw).hexdigest(),
        "pdf_bytes": len(pdf_raw),
        "stamp": rendered_stamp,
        **semantic,
    }
    sidecar = _metadata_path(output)
    _atomic_bytes(
        sidecar,
        (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"))
    return {**metadata, "pdf": str(output), "metadata": str(sidecar)}


def check(source: Path, output: Path) -> dict:
    source = source.resolve(strict=True)
    if source != DEFAULT_SOURCE.resolve(strict=True):
        raise ValueError(
            "the distributable field guide must use the canonical repo HTML")
    output = output.resolve(strict=True)
    sidecar = _metadata_path(output).resolve(strict=True)
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    required = {
        "metadata_schema_version", "derivative_type", "evidence_status",
        "source_html", "source_html_sha256", "source_git_commit",
        "source_git_dirty", "source_commit_time_utc", "generated_at_utc",
        "renderer", "renderer_command", "pdf_sha256", "pdf_bytes", "stamp",
        "pdf_page_count", "pdf_title", "pdf_creator", "pdf_producer",
        "pdf_page_size", "pdf_encrypted", "pdf_javascript",
        "pdfinfo_version", "pdftotext_version", "semantic_qa_version",
        "semantic_requirements_sha256", "geometry_qa_version",
        "minimum_footer_clearance_points",
    }
    if not isinstance(metadata, dict) or set(metadata) != required:
        raise RuntimeError("PDF metadata sidecar has unknown or missing fields")
    if metadata["metadata_schema_version"] != 2 \
            or metadata["source_git_dirty"] is not False:
        raise RuntimeError("PDF is not a clean-source distributable build")
    if metadata["derivative_type"] != "customer_field_guide_pdf" \
            or metadata["evidence_status"] != \
            "unsealed_navigation_derivative":
        raise RuntimeError("PDF metadata has an invalid derivative identity")
    if metadata["renderer_command"] != "playwright pdf" \
            or not all(isinstance(metadata[field], str) and metadata[field]
                       for field in (
                           "renderer", "pdfinfo_version", "pdftotext_version")):
        raise RuntimeError("PDF build-tool provenance is missing or invalid")
    if metadata["source_html"] != source.relative_to(ROOT).as_posix() \
            or metadata["source_html_sha256"] != _sha256(source):
        raise RuntimeError("PDF source HTML is stale")
    if metadata["pdf_sha256"] != _sha256(output) \
            or metadata["pdf_bytes"] != output.stat().st_size:
        raise RuntimeError("PDF bytes disagree with the metadata sidecar")
    commit = metadata["source_git_commit"]
    if not isinstance(commit, str) or not COMMIT_RE.fullmatch(commit):
        raise RuntimeError("PDF metadata has an invalid source commit")
    historic = _git(
        "show", f"{commit}:{metadata['source_html']}", text=False)
    if hashlib.sha256(historic).hexdigest() != metadata["source_html_sha256"]:
        raise RuntimeError("recorded Git commit does not contain the PDF source")
    commit_time = _source_commit_time(commit)
    expected_stamp = _rendered_stamp(
        commit, metadata["source_html_sha256"], commit_time, dirty=False)
    if metadata["source_commit_time_utc"] != commit_time \
            or metadata["stamp"] != expected_stamp:
        raise RuntimeError("PDF source stamp metadata is invalid")
    if not isinstance(metadata["pdf_sha256"], str) \
            or not HASH_RE.fullmatch(metadata["pdf_sha256"]) \
            or not isinstance(metadata["pdf_bytes"], int) \
            or metadata["pdf_bytes"] <= 0:
        raise RuntimeError("PDF byte provenance is invalid")

    semantic = _inspect_pdf(
        output,
        source_sha=metadata["source_html_sha256"],
        source_commit=commit,
    )
    semantic_fields = {
        "pdf_page_count", "pdf_title", "pdf_creator", "pdf_producer",
        "pdf_page_size", "pdf_encrypted", "pdf_javascript",
        "semantic_qa_version", "semantic_requirements_sha256",
        "geometry_qa_version", "minimum_footer_clearance_points",
    }
    if any(metadata[field] != semantic[field] for field in semantic_fields):
        raise RuntimeError("PDF semantic QA metadata disagrees with the PDF")
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build or verify the stamped customer field-guide PDF")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    result = (check(args.source, args.output) if args.check else
              build(args.source, args.output, allow_dirty=args.allow_dirty))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
