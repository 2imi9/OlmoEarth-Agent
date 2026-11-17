# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""The label view's sheet: which files it opens, what it reads, what it writes."""

from __future__ import annotations

import base64
import csv
import hashlib
import hmac
import importlib
import io
import json
import os
import re
from pathlib import Path
from typing import Any

import pytest

from olmoearth_agent.mcp_app import sheet

ONE_MAP = [
    "order",
    "window_index",
    "row",
    "col",
    "map_class",
    "lon",
    "lat",
    "stratum",
    "wrong",
    "reference_class",
]
TWO_MAP = [
    "order",
    "window_index",
    "row",
    "col",
    "class_a",
    "class_b",
    "reference_class",
]


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    out = tmp_path / "outputs"
    out.mkdir()
    monkeypatch.setenv("OLMOEARTH_OUTPUT_ROOT", str(out))
    monkeypatch.delenv("OLMOEARTH_SCORES_ROOT", raising=False)
    return out


def _write(path: Path, header: list[str], rows: list[list[str]]) -> Path:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def _one_map(root: Path, design: str = "random") -> Path:
    (root / "plan.json").write_text(
        json.dumps(
            {
                "design": design,
                "population": {
                    "score_kind": "class_map",
                    "classes": ["water", "trees", "crops"],
                },
            }
        ),
        encoding="utf-8",
    )
    return _write(
        root / "plan_to_label.csv",
        ONE_MAP,
        [
            ["1", "7", "0", "7", "water", "10.5", "50.25", "0", "", ""],
            ["2", "3", "0", "3", "trees", "10.6", "50.26", "0", "", ""],
        ],
    )


def _rows(path: Path) -> list[list[str]]:
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.reader(fh))


def test_resolve_refuses_files_that_are_not_a_plans_sheet(
    root: Path, tmp_path: Path
) -> None:
    other = _write(tmp_path / "elsewhere_to_label.csv", ONE_MAP, [])
    named = _write(root / "plan.csv", ONE_MAP, [])
    for path in (
        str(other),
        str(named),
        "",
        "a\x00b_to_label.csv",
        str(root / "missing_to_label.csv"),
    ):
        with pytest.raises(ValueError):
            sheet.resolve_sheet(path)


def test_resolve_refuses_a_link_out_of_the_root(root: Path, tmp_path: Path) -> None:
    target = _write(tmp_path / "secret_to_label.csv", ONE_MAP, [])
    link = root / "link_to_label.csv"
    link.symlink_to(target)
    with pytest.raises(ValueError):
        sheet.resolve_sheet(str(link))


def _by_window(path: Path) -> dict[int, str]:
    """Each window's handle by its index: the server's own map, never sent."""
    _, handles = sheet.window_handles(str(path))
    return {index: handle for handle, index in handles.items()}


def _answer_values(read: dict[str, Any]) -> list[str]:
    """The values of a page copy's answers, in order."""
    return [a["value"] for a in read["answers"]]


def test_read_one_map_sheet(root: Path) -> None:
    path = _one_map(root, design="class")
    data = sheet.read_sheet(str(path))
    assert (data["kind"], data["face"], data["open"]) == ("one_map", "class", True)
    # The class face takes Cannot tell after its classes (the last round's S4).
    assert _answer_values(data) == ["water", "trees", "crops", "?"]
    assert data["faces"] == ["class", "wrong"]
    # No face switch on the panel (the owner's O1 of 14 Nov 2026).
    assert "face_choices" not in data and "has_stratum" not in data
    assert data["n_rows"] == 2 and data["n_labelled"] == 0
    assert sorted(w["handle"] for w in data["windows"]) == sorted(
        _by_window(path).values()
    )
    for gone in ("rows", "columns", "classes", "class_names", "design", "has_wrong"):
        assert gone not in data


def test_save_labels_writes_only_the_label_cells(root: Path) -> None:
    path = _one_map(root)
    before = _rows(path)
    out = sheet.save_labels(
        str(path), [{"window_index": 3, "reference_class": "crops"}]
    )
    assert out["cells_changed"] == 1 and out["n_labelled"] == 1
    after = _rows(path)
    assert after[0] == before[0] and after[1] == before[1]
    assert after[2][:-1] == before[2][:-1] and after[2][-1] == "crops"
    assert not (root / "plan_to_label.csv.tmp").exists()
    again = sheet.save_labels(
        str(path), [{"window_index": 3, "reference_class": "crops"}]
    )
    assert again["cells_changed"] == 0


def test_save_labels_wrong_column(root: Path) -> None:
    path = _one_map(root)
    sheet.save_labels(
        str(path),
        [{"window_index": 7, "wrong": "1"}, {"window_index": 3, "wrong": "0"}],
    )
    rows = _rows(path)
    assert [r[ONE_MAP.index("wrong")] for r in rows[1:]] == ["1", "0"]


def test_a_one_map_sheet_takes_cannot_tell_in_wrong(root: Path) -> None:
    """'?' in wrong: Cannot tell, which the estimate bounds both ways."""
    path = _one_map(root)
    out = sheet.save_labels(str(path), [{"window_index": 7, "wrong": "?"}])
    assert out["cells_changed"] == 1
    assert _rows(path)[1][ONE_MAP.index("wrong")] == "?"
    read = sheet.read_sheet(str(path))
    assert (read["n_labelled"], read["n_unjudged"]) == (1, 1)
    # A class map's class seen takes '?' too (the last round's S4) ...
    sheet.save_labels(str(path), [{"window_index": 3, "reference_class": "?"}])
    read = sheet.read_sheet(str(path))
    assert (read["n_labelled"], read["n_unjudged"]) == (2, 2)
    # ... a scores map's one-map sheet does not, in reference_class.
    (root / "conf.json").write_text(
        json.dumps({"design": "random", "population": {"score_kind": "x"}}),
        encoding="utf-8",
    )
    scores = _write(
        root / "conf_to_label.csv",
        ONE_MAP,
        [["1", "7", "0", "7", "1", "10.5", "50.25", "0", "", ""]],
    )
    before = scores.read_bytes()
    with pytest.raises(ValueError) as refused:
        sheet.save_labels(str(scores), [{"window_index": 7, "reference_class": "?"}])
    assert str(refused.value) == sheet.UNJUDGED_NOT_HERE
    assert scores.read_bytes() == before


def test_a_certificate_plans_sheet_takes_no_cannot_tell(root: Path) -> None:
    """A seed_from_map plan's certificate needs 1 or 0 on every window."""
    (root / "cert.json").write_text(
        json.dumps(
            {
                "design": "random",
                "population": {"score_kind": "window_confidence"},
                "certificate": {"seed_rule": "from_map"},
            }
        ),
        encoding="utf-8",
    )
    path = _write(
        root / "cert_to_label.csv",
        ONE_MAP,
        [["1", "4", "0", "4", "1", "10.5", "50.2", "0", "", ""]],
    )
    assert sheet.read_sheet(str(path))["takes_unjudged"] is False
    before = path.read_bytes()
    with pytest.raises(ValueError, match="certificate plan"):
        sheet.save_labels(str(path), [{"window_index": 4, "wrong": "?"}])
    assert path.read_bytes() == before
    sheet.save_labels(str(path), [{"window_index": 4, "wrong": "1"}])
    assert sheet.read_sheet(str(_one_map(root)))["takes_unjudged"] is True


def test_a_scores_maps_class_names_come_from_its_design(root: Path) -> None:
    """Its design names the classes but holds no map_class, so the server
    cannot work out wrong from a class seen: the sheet stays right or wrong."""
    (root / "kelp.json").write_text(
        json.dumps(
            {
                "design": "confidence",
                "population": {
                    "score_kind": "window_confidence",
                    "class_names": {"0": "absent", "1": "present", "x": "bad", "2": ""},
                },
            }
        ),
        encoding="utf-8",
    )
    path = _write(
        root / "kelp_to_label.csv",
        ONE_MAP,
        [["1", "4", "0", "4", "1", "10.5", "50.2", "0", "", ""]],
    )
    read = sheet.read_sheet(str(path))
    assert read["label_rule"] == "wrong" and read["face"] == "wrong"
    assert [w["map_class"] for w in read["windows"]] == ["present"]
    assert _answer_values(read) == ["0", "1", "?"]
    assert "rule_note" not in read


@pytest.mark.parametrize(
    "label",
    [
        {"window_index": 99, "reference_class": "water"},
        {"window_index": 7, "wrong": "2"},
        {"window_index": 7, "reference_class": "a\nb"},
        {"reference_class": "water"},
    ],
)
def test_save_labels_refuses(root: Path, label: dict[str, object]) -> None:
    path = _one_map(root)
    before = path.read_bytes()
    with pytest.raises(ValueError):
        sheet.save_labels(str(path), [label])
    assert path.read_bytes() == before


def test_two_map_sheet_takes_unjudged_and_has_no_wrong(root: Path) -> None:
    path = _write(
        root / "pair_to_label.csv",
        TWO_MAP,
        [["1", "4", "0", "4", "water", "trees", ""]],
    )
    data = sheet.read_sheet(str(path))
    assert data["kind"] == "two_map" and data["face"] == "pair"
    # No design file: the classes the two maps show, sorted by name, then ?.
    assert _answer_values(data) == ["trees", "water", "?"]
    sheet.save_labels(str(path), [{"window_index": 4, "reference_class": "?"}])
    assert _rows(path)[1][-1] == "?"
    with pytest.raises(ValueError, match="no wrong column"):
        sheet.save_labels(str(path), [{"window_index": 4, "wrong": "1"}])


def test_a_semicolon_sheet_is_refused_with_how_to_fix(root: Path) -> None:
    path = root / "plan_to_label.csv"
    path.write_text("order;window_index;reference_class\n1;7;\n", encoding="utf-8")
    with pytest.raises(ValueError, match="comma-separated"):
        sheet.read_sheet(str(path))


def test_the_label_rule_follows_the_design(root: Path) -> None:
    path = _one_map(root)
    data = sheet.read_sheet(str(path))
    assert data["label_rule"] == "class"
    assert data["design_path"] == str((root / "plan.json").resolve())
    (root / "plan.json").write_text(
        json.dumps(
            {"design": "confidence", "population": {"score_kind": "class_scores"}}
        ),
        encoding="utf-8",
    )
    assert sheet.read_sheet(str(path))["label_rule"] == "wrong"
    (root / "plan.json").unlink()
    data = sheet.read_sheet(str(path))
    assert data["label_rule"] is None and data["design_path"] is None


def test_a_two_map_sheet_offers_the_designs_own_classes_first(root: Path) -> None:
    (root / "pair.json").write_text(
        json.dumps({"design": "disagreement", "classes": ["crop", "forest", "water"]}),
        encoding="utf-8",
    )
    path = _write(
        root / "pair_to_label.csv",
        TWO_MAP,
        [
            ["1", "4", "0", "4", "water", "forest", ""],
            ["2", "5", "0", "5", "wetland", "water", ""],
        ],
    )
    data = sheet.read_sheet(str(path))
    assert data["label_rule"] == "class"
    assert _answer_values(data) == ["crop", "forest", "water", "wetland", "?"]


def test_a_save_leaves_no_staged_file_and_keeps_the_mode(root: Path) -> None:
    path = _one_map(root)
    path.chmod(0o640)
    sheet.save_labels(str(path), [{"window_index": 7, "reference_class": "crops"}])
    leftovers = sorted(
        p.name for p in root.iterdir() if p.name.startswith(".plan_to_label.csv.")
    )
    # The lock, and the sheet's own handles' secret (a save reads the sheet's
    # version to keep a graded mark true, the last round's Q1): no staged file.
    assert leftovers == [".plan_to_label.csv.handles", ".plan_to_label.csv.lock"]
    assert (path.stat().st_mode & 0o777) == 0o640


def test_a_link_at_the_staging_place_is_not_written_through(
    root: Path, tmp_path: Path
) -> None:
    """The old fixed staging name (<sheet>.tmp) as a link: a save no longer uses it."""
    path = _one_map(root)
    target = tmp_path / "victim.txt"
    target.write_text("keep", encoding="utf-8")
    (root / "plan_to_label.csv.tmp").symlink_to(target)
    sheet.save_labels(str(path), [{"window_index": 7, "reference_class": "crops"}])
    assert target.read_text(encoding="utf-8") == "keep"


def test_the_finish_button_names_the_sheets_job(root: Path) -> None:
    """The finish button and its message name the tool the sheet's job needs:
    the estimate for a plan, the certification at the plan's own alpha for a
    certificate plan (never the estimate), which map for a two-map sheet."""
    path = _one_map(root)
    labels = str(path.resolve())
    design = str((root / "plan.json").resolve())
    plain = sheet.read_sheet(str(path))
    call = (
        f"olmoearth_estimate_map_error with design_path {design} and "
        f"labels_path {labels}"
    )
    assert (plain["job"], plain["finish_label"]) == (
        "estimate",
        "Estimate the error rate",
    )
    assert plain["finish_call"] == call
    # It starts in plain words (r5-a L3), then the line Claude needs, as it
    # was; no "window" (the owner's ruling 6 of 13 Nov 2026).
    assert plain["finish_message"] == (
        "I have answered every square of this sheet. Estimate the error rate. "
        f"Estimate the map's error rate from it: {call}."
    )

    (root / "cert.json").write_text(
        json.dumps(
            {
                "design": "random",
                "population": {"score_kind": "window_confidence"},
                "certificate": {"seed_rule": "from_map", "alpha": 0.05},
            }
        ),
        encoding="utf-8",
    )
    cert_path = _write(
        root / "cert_to_label.csv",
        ONE_MAP,
        [["1", "4", "0", "4", "1", "10.5", "50.2", "0", "", ""]],
    )
    cert = sheet.read_sheet(str(cert_path))
    cert_call = (
        f"olmoearth_certify_zone with design_path {(root / 'cert.json').resolve()}, "
        f"labels_path {cert_path.resolve()} and alpha 0.05 (fixed by the plan)"
    )
    assert (cert["job"], cert["finish_label"]) == ("certify", "Certify the zone")
    assert cert["finish_call"] == cert_call
    assert cert["finish_message"].startswith(
        "I have answered every square of this sheet. Certify the zone. "
        f"Certify the zone from it: {cert_call}. If a zone is certified, offer"
    )
    assert "olmoearth_write_certificate with the same design_path" in (
        cert["finish_message"]
    )
    assert "estimate_map_error" not in cert["finish_message"]

    (root / "pair.json").write_text(
        json.dumps({"design": "disagreement", "classes": ["crop", "water"]}),
        encoding="utf-8",
    )
    pair_path = _write(
        root / "pair_to_label.csv",
        TWO_MAP,
        [["1", "4", "0", "4", "water", "crop", ""]],
    )
    pair = sheet.read_sheet(str(pair_path))
    assert (pair["job"], pair["finish_label"]) == (
        "which_map",
        "Say which map is more accurate",
    )
    assert pair["finish_message"] == (
        "I have answered every square of this sheet. Say which map is more "
        "accurate. Say which "
        "map is more accurate from it: olmoearth_estimate_map_error with "
        f"design_path {(root / 'pair.json').resolve()} and labels_path "
        f"{pair_path.resolve()}."
    )

    # No design file beside the sheet: the panel's older words, no job.
    (root / "plan.json").unlink()
    bare = sheet.read_sheet(str(path))
    assert (bare["job"], bare["finish_label"]) == (None, "Ask Claude to grade it")
    assert bare["finish_call"] == (
        f"olmoearth_estimate_map_error with labels_path {labels}"
    )
    assert bare["finish_message"] == (
        "I have answered every square of this sheet. Estimate "
        f"the map error from it: labels_path {labels}."
    )
    for message in (plain, cert, pair, bare):
        assert "window" not in message["finish_message"].split(":")[0]
    (root / "pair.json").unlink()
    assert sheet.read_sheet(str(pair_path))["job"] == "which_map"


@pytest.mark.parametrize(
    ("design", "job"),
    [
        ({"design": "random"}, "estimate"),
        ({"design": "class", "certificate": {"seed_rule": "other"}}, "estimate"),
        ({"design": "random", "certificate": {"seed_rule": "from_map"}}, "certify"),
        ({"design": "disagreement"}, "which_map"),
        (None, None),
    ],
)
def test_the_sheets_job_follows_its_design(
    design: dict[str, object] | None, job: str | None
) -> None:
    assert sheet.sheet_job(design, "one_map") == job
    assert sheet.sheet_job(design, "two_map") == "which_map"


@pytest.mark.parametrize("alpha", [None, True, "0.05"])
def test_a_certify_call_without_the_plans_alpha_names_none(alpha: object) -> None:
    """A design file whose certificate holds no number for alpha: the call
    points at the plan's alpha rather than naming one."""
    block: dict[str, object] = {"seed_rule": "from_map"}
    if alpha is not None:
        block["alpha"] = alpha
    call = sheet.finish_call(
        "certify", labels_path="L", design_path="D", design={"certificate": block}
    )
    assert call == (
        "olmoearth_certify_zone with design_path D, labels_path L and the alpha "
        "the plan fixed (certificate.alpha in its design file)"
    )


#: A named scores map's class of 20 windows, as its design holds it (-1:
#: no class there).
SEEN_MAP = [1 if w in (4, 12) else -1 if w == 15 else 0 for w in range(20)]

#: Why a value is no answer on a seen sheet, with and without Cannot tell.
NO_ANSWER = (
    "reference_class here is one of the codes 1, 0, or ? (cannot tell), or blank."
)
NO_ANSWER_CERT = "reference_class here is one of the codes 1, 0, or blank."


def _seen(
    root: Path,
    *,
    certificate: bool = False,
    on_map: list[int] | None = None,
    windows: tuple[int, ...] = (4, 9, 12),
) -> Path:
    """A named scores map's plan, as a scores_from_file plan writes one (the
    kelp trial's): the design with class names and the map's classes, and
    its sheet with every label blank."""
    on_map = SEEN_MAP if on_map is None else on_map
    design: dict[str, Any] = {
        "design": "confidence",
        "population": {
            "score_kind": "window_confidence",
            "class_names": {"0": "absent", "1": "present"},
            "map_class": on_map,
        },
    }
    if certificate:
        design["design"] = "random"
        design["certificate"] = {"seed_rule": "from_map", "alpha": 0.05}
    (root / "kelp.json").write_text(json.dumps(design), encoding="utf-8")
    rows = [
        [
            str(order),
            str(w),
            "0",
            str(w),
            str(on_map[w]) if w < len(on_map) else "0",
            "10.5",
            "50.2",
            str(order % 2),
            "",
            "",
        ]
        for order, w in enumerate(windows, start=1)
    ]
    return _write(root / "kelp_to_label.csv", ONE_MAP, rows)


def _cells(path: Path, column: str) -> dict[str, str]:
    """One column of the sheet on disk, by window."""
    rows = _rows(path)
    at, at_window = rows[0].index(column), rows[0].index("window_index")
    return {row[at_window]: row[at] for row in rows[1:]}


def _set_cell(path: Path, window: str, column: str, value: str) -> None:
    """Edit one cell of the sheet, as a spreadsheet would."""
    rows = _rows(path)
    at, at_window = rows[0].index(column), rows[0].index("window_index")
    for row in rows[1:]:
        if row[at_window] == window:
            row[at] = value
    _write(path, rows[0], rows[1:])


def test_a_named_scores_map_asks_for_the_class_seen(root: Path) -> None:
    """A scores map whose design names its classes and holds the map's class:
    the panel picks the class seen (Present, Absent, Cannot tell) and gets no
    map class, stratum, wrong or order."""
    path = _seen(root)
    read = sheet.read_sheet(str(path))
    assert read["label_rule"] == "seen" and read["kind"] == "one_map"
    assert (read["face"], read["faces"], read["open"]) == ("seen", ["seen"], False)
    assert read["answers"] == [
        {"value": "1", "label": "Present", "keys": ["P", "1"]},
        {"value": "0", "label": "Absent", "keys": ["A", "2"]},
        {"value": "?", "label": "Cannot tell", "keys": ["?", "3"]},
    ]
    assert all(set(w) == {"handle", "answer"} for w in read["windows"])
    assert sorted(_by_window(path)) == [4, 9, 12] and len(read["windows"]) == 3
    dumped = json.dumps(read)
    for hidden in ("map_class", "wrong", "stratum", "order", "window_index"):
        assert f'"{hidden}"' not in dumped
    assert (read["n_rows"], read["n_labelled"], read["job"]) == (3, 0, "estimate")
    assert "rule_note" not in read
    assert sheet.sheet_rule(str(root / "kelp_to_label.csv")) == "seen"
    # A sample the map calls absent everywhere still offers both answers.
    path = _seen(root, on_map=[0] * 20)
    assert _answer_values(sheet.read_sheet(str(path))) == ["1", "0", "?"]


def test_a_certificate_plans_seen_sheet_offers_no_cannot_tell(root: Path) -> None:
    read = sheet.read_sheet(str(_seen(root, certificate=True)))
    assert read["label_rule"] == "seen" and read["job"] == "certify"
    assert [a["label"] for a in read["answers"]] == ["Present", "Absent"]
    assert read["takes_unjudged"] is False


def test_a_seen_label_writes_the_code_and_wrong_from_the_designs_map(
    root: Path,
) -> None:
    """The server writes the code seen and works out wrong from the design's
    map class (never the sheet's map_class cell); cells_changed counts the
    reference_class cells."""
    path = _seen(root)
    out = sheet.save_labels(
        str(path),
        [
            {"window_index": 4, "reference_class": "1"},
            {"window_index": 9, "reference_class": "1"},
            {"window_index": 12, "reference_class": "?"},
        ],
    )
    assert (out["cells_changed"], out["n_rows"], out["n_labelled"]) == (3, 3, 3)
    assert _cells(path, "reference_class") == {"4": "1", "9": "1", "12": "?"}
    assert _cells(path, "wrong") == {"4": "0", "9": "1", "12": "?"}
    assert sheet.read_sheet(str(path))["n_unjudged"] == 1
    # A changed answer works wrong out again; a cleared one clears both.
    again = sheet.save_labels(
        str(path),
        [
            {"window_index": 4, "reference_class": "0"},
            {"window_index": 12, "reference_class": ""},
        ],
    )
    assert again["cells_changed"] == 2
    assert _cells(path, "wrong") == {"4": "1", "9": "1", "12": ""}
    assert _cells(path, "reference_class")["12"] == ""
    # The sheet's map_class cell is never read: edited, it changes nothing.
    _set_cell(path, "9", "map_class", "1")
    sheet.save_labels(str(path), [{"window_index": 9, "reference_class": "0"}])
    assert _cells(path, "wrong")["9"] == "0"
    # A wrong cell out of step with its answer is written again from the
    # answer, though no reference_class cell changed.
    _set_cell(path, "9", "wrong", "1")
    same = sheet.save_labels(str(path), [{"window_index": 9, "reference_class": "0"}])
    assert same["cells_changed"] == 0
    assert _cells(path, "wrong")["9"] == "0"
    assert sheet.read_sheet(str(path))["label_rule"] == "seen"


@pytest.mark.parametrize(
    ("label", "certificate", "message"),
    [
        ({"window_index": 4, "wrong": "1"}, False, sheet.SEEN_NO_WRONG),
        (
            {"window_index": 4, "reference_class": "1", "wrong": "0"},
            False,
            sheet.SEEN_NO_WRONG,
        ),
        ({"window_index": 4, "reference_class": "present"}, False, NO_ANSWER),
        ({"window_index": 4, "reference_class": "2"}, False, NO_ANSWER),
        ({"window_index": 4, "reference_class": "present"}, True, NO_ANSWER_CERT),
        ({"window_index": 4, "reference_class": "?"}, True, "certificate plan"),
        (
            {"window_index": 30, "reference_class": "1"},
            False,
            "window 30 has no class on this design's map",
        ),
        (
            {"window_index": 15, "reference_class": "0"},
            False,
            "window 15 has no class on this design's map",
        ),
    ],
)
def test_a_seen_sheet_refuses_what_is_no_answer(
    root: Path, label: dict[str, object], certificate: bool, message: str
) -> None:
    path = _seen(root, certificate=certificate, windows=(4, 9, 12, 15, 30))
    assert sheet.read_sheet(str(path))["label_rule"] == "seen"
    before = path.read_bytes()
    with pytest.raises(ValueError) as caught:
        sheet.save_labels(
            str(path), [{"window_index": 9, "reference_class": "0"}, label]
        )
    assert message in str(caught.value)
    assert path.read_bytes() == before


def test_a_named_sheet_started_right_or_wrong_stays_so(root: Path) -> None:
    """A row with wrong filled and reference_class blank: the sheet keeps
    right or wrong, with the map's class shown and a note why."""
    path = _seen(root)
    _set_cell(path, "4", "wrong", "1")
    read = sheet.read_sheet(str(path))
    assert read["label_rule"] == "wrong" and read["rule_note"] == sheet.RULE_NOTE
    assert read["face"] == "wrong" and _answer_values(read) == ["0", "1", "?"]
    at = _by_window(path)
    assert {w["handle"]: w["map_class"] for w in read["windows"]} == {
        at[4]: "present",
        at[9]: "absent",
        at[12]: "present",
    }
    assert sheet.sheet_rule(str(path)) == "wrong"
    sheet.save_labels(str(path), [{"window_index": 9, "wrong": "0"}])
    assert _cells(path, "wrong") == {"4": "1", "9": "0", "12": ""}


def test_a_panels_save_is_refused_once_the_sheets_rule_changed(root: Path) -> None:
    """The panel sends the rule it drew; a sheet read by another rule now (its
    design file moved, or wrong filled in a spreadsheet) is left as it is."""
    path = _seen(root)
    labels = [{"window_index": 4, "reference_class": "1"}]
    saved = sheet.save_labels(str(path), labels, rule="seen", check_rule=True)
    assert saved["cells_changed"] == 1
    design = root / "kelp.json"
    kept = design.read_bytes()
    design.unlink()
    before = path.read_bytes()
    with pytest.raises(ValueError, match="changed since the panel opened it"):
        sheet.save_labels(
            str(path),
            [{"window_index": 9, "reference_class": "0"}],
            rule="seen",
            check_rule=True,
        )
    assert path.read_bytes() == before
    design.write_bytes(kept)
    _set_cell(path, "9", "wrong", "1")
    before = path.read_bytes()
    with pytest.raises(ValueError, match="changed since the panel opened it"):
        sheet.save_labels(
            str(path),
            [{"window_index": 12, "reference_class": "1"}],
            rule="seen",
            check_rule=True,
        )
    assert path.read_bytes() == before
    # The rule it has now is taken.
    sheet.save_labels(
        str(path), [{"window_index": 12, "wrong": "0"}], rule="wrong", check_rule=True
    )
    assert _cells(path, "wrong")["12"] == "0"


@pytest.mark.parametrize(
    ("population", "codes"),
    [
        (
            {"class_names": {"0": "absent", "1": "present"}, "map_class": [0, 1, -1]},
            ["1", "0"],
        ),
        (
            {
                "class_names": {"0": "water", "1": "Absence", "2": "Presence"},
                "map_class": [0, 2],
            },
            ["2", "1", "0"],
        ),
        (
            {"class_names": ["crop", "water", "trees"], "map_class": [2, 0]},
            ["0", "1", "2"],
        ),
        ({"class_names": {"1": "present"}, "map_class": [1]}, None),
        ({"class_names": {"0": "absent", "1": "present"}}, None),
        ({"class_names": {"0": "absent", "1": "present"}, "map_class": [0, 2]}, None),
        (
            {"class_names": {"0": "absent", "1": "present"}, "map_class": [0, True]},
            None,
        ),
        ({"class_names": {"0": "absent", "1": "present"}, "map_class": [0, 1.0]}, None),
        ({"class_names": {"0": "absent", "01": "present"}, "map_class": [0]}, None),
        # A scores file's placeholder names (no classes in its manifest) name
        # no class: such a map stays right or wrong (owner ruling 1).
        ({"class_names": {"0": "class_0", "1": "class_1"}, "map_class": [0, 1]}, None),
        ({"class_names": ["CLASS_0", "Class_1", "class_2"], "map_class": [2]}, None),
        (
            {"class_names": {"0": "class_0", "1": "kelp"}, "map_class": [0, 1]},
            ["0", "1"],
        ),
        (
            {
                "score_kind": "class_map",
                "class_names": {"0": "absent", "1": "present"},
                "map_class": [0],
            },
            None,
        ),
        (
            {
                "score_kind": "topk_score",
                "class_names": {"0": "absent", "1": "present"},
                "map_class": [0],
            },
            None,
        ),
    ],
)
def test_the_answer_codes_follow_the_design(
    population: dict[str, Any], codes: list[str] | None
) -> None:
    population = {"score_kind": "window_confidence", **population}
    assert sheet._seen_codes({"design": "random", "population": population}) == codes
    assert sheet._seen_codes(None) is None


def test_a_scores_map_with_placeholder_names_stays_right_or_wrong(root: Path) -> None:
    """A scores run whose manifest names no class gets class_0, class_1 from
    its scores file: the sheet asks right or wrong, with the map's class."""
    path = _seen(root)
    design = json.loads((root / "kelp.json").read_text(encoding="utf-8"))
    design["population"]["class_names"] = {"0": "class_0", "1": "class_1"}
    (root / "kelp.json").write_text(json.dumps(design), encoding="utf-8")
    read = sheet.read_sheet(str(path))
    assert read["label_rule"] == "wrong" and read["face"] == "wrong"
    assert "rule_note" not in read and _answer_values(read) == ["0", "1", "?"]
    assert all("map_class" in w for w in read["windows"])
    assert sheet.sheet_progress(str(path))["blind"] is False
    sheet.save_labels(str(path), [{"window_index": 4, "wrong": "0"}])
    assert _cells(path, "wrong")["4"] == "0"


def test_a_seen_row_counts_once_its_wrong_is_worked_out(root: Path) -> None:
    """Codes typed in a spreadsheet with wrong blank are not yet labelled: the
    page shows no answer there and the counts leave them out, so the panel
    never offers the finish for a sheet the estimate refuses. The next save
    works wrong out on every row holding an answer, sent or not."""
    path = _seen(root)
    for window, code in (("4", "1"), ("9", "1"), ("12", "?")):
        _set_cell(path, window, "reference_class", code)
    read = sheet.read_sheet(str(path))
    assert read["label_rule"] == "seen"
    assert [w["answer"] for w in read["windows"]] == ["", "", ""]
    assert (read["n_labelled"], read["n_unjudged"]) == (0, 0)
    progress = sheet.sheet_progress(str(path))
    assert (progress["n_labelled"], progress["open_label"]) == (0, sheet.OPEN_FIRST)
    # One answer from the panel: wrong is worked out on every row with an answer.
    out = sheet.save_labels(
        str(path),
        [{"window_index": 4, "reference_class": "1"}],
        rule="seen",
        check_rule=True,
    )
    assert (out["cells_changed"], out["n_labelled"]) == (0, 3)
    assert _cells(path, "wrong") == {"4": "0", "9": "1", "12": "?"}
    read = sheet.read_sheet(str(path))
    at = _by_window(path)
    assert {w["handle"]: w["answer"] for w in read["windows"]} == {
        at[4]: "1",
        at[9]: "1",
        at[12]: "?",
    }
    assert (read["n_labelled"], read["n_unjudged"]) == (3, 1)
    assert sheet.sheet_progress(str(path))["open_label"] == sheet.OPEN_DONE
    # A wrong edited by hand is worked out again at the next save, from the
    # design's map class, though its row was not sent.
    _set_cell(path, "9", "wrong", "0")
    sheet.save_labels(str(path), [{"window_index": 12, "reference_class": "0"}])
    assert _cells(path, "wrong") == {"4": "0", "9": "1", "12": "1"}
    # A cell that holds no answer (a name, here) is neither shown nor counted.
    _set_cell(path, "4", "reference_class", "present")
    read = sheet.read_sheet(str(path))
    assert {w["handle"]: w["answer"] for w in read["windows"]}[at[4]] == ""
    assert read["n_labelled"] == 2
    sheet.save_labels(str(path), [])
    assert _cells(path, "wrong")["4"] == "0"  # left as it was


def test_a_panels_save_that_works_out_rows_not_sent_says_rederived(root: Path) -> None:
    """A seen sheet with a code typed in a spreadsheet, wrong left blank: the
    panel's page shows that window unlabelled. A panel's save of another
    window works the typed row out too, so its answer says rederived (naming
    no handle or row) and the panel reads the sheet again: the version the
    save answers names that row labelled. A save that changes only the rows
    sent does not say it."""
    path = _seen(root)
    _set_cell(path, "9", "reference_class", "1")
    read = sheet.read_sheet(str(path))
    at = _by_window(path)
    assert {w["handle"]: w["answer"] for w in read["windows"]}[at[9]] == ""
    good = {"sheet_id": read["sheet_id"], "face": "seen", "rule": "seen"}
    out = sheet.save_answers(
        str(path),
        **good,
        answers=[{"handle": at[4], "answer": "1"}],
        version=read["version"],
    )
    assert out["rederived"] is True
    assert (out["cells_changed"], out["n_labelled"]) == (1, 2)
    assert set(out) == {
        "path",
        "cells_changed",
        "n_rows",
        "n_labelled",
        "mix_problem",
        "clear_column",
        "version",
        "rederived",
    }
    again = sheet.read_sheet(str(path))
    assert again["version"] == out["version"]
    assert {w["handle"]: w["answer"] for w in again["windows"]}[at[9]] == "1"
    assert again["n_labelled"] == 2
    plain = sheet.save_answers(
        str(path),
        **good,
        answers=[{"handle": at[12], "answer": "0"}],
        version=out["version"],
    )
    assert plain["cells_changed"] == 1 and "rederived" not in plain
    assert plain["n_labelled"] == 3


def test_a_sheet_opens_blind_while_it_asks_for_the_class_seen(root: Path) -> None:
    """blind (the server holds the figure of the map's confidence from the plan
    card until the sheet is complete): a seen sheet, a class map's, a two-map
    sheet; not a sheet labelled right or wrong."""
    assert sheet.sheet_progress(str(_seen(root)))["blind"] is True
    class_map = _one_map(root)
    assert sheet.sheet_progress(str(class_map))["blind"] is True
    sheet.save_labels(str(class_map), [{"window_index": 7, "wrong": "1"}])
    assert sheet.sheet_progress(str(class_map))["blind"] is False  # wrong alone
    sheet.save_labels(str(class_map), [{"window_index": 3, "reference_class": "trees"}])
    assert sheet.sheet_progress(str(class_map))["blind"] is True
    started = _seen(root)
    _set_cell(started, "4", "wrong", "1")
    assert sheet.sheet_progress(str(started))["blind"] is False
    (root / "kelp.json").unlink()
    assert sheet.sheet_progress(str(started))["blind"] is False  # no design file
    pair = _write(
        root / "pair_to_label.csv", TWO_MAP, [["1", "4", "0", "4", "a", "b", ""]]
    )
    assert sheet.sheet_progress(str(pair))["blind"] is True


@pytest.mark.parametrize(
    ("progress", "shown"),
    [
        (None, False),
        ({"blind": False, "n_rows": 2, "n_labelled": 0}, True),
        ({"blind": True, "n_rows": 2, "n_labelled": 0}, False),
        ({"blind": True, "n_rows": 2, "n_labelled": 1}, False),
        ({"blind": True, "n_rows": 2, "n_labelled": 2}, True),
        ({"blind": True, "n_rows": 0, "n_labelled": 0}, False),
        ({"blind": True, "n_rows": "2", "n_labelled": "2"}, False),
        ({"n_rows": 2, "n_labelled": 1}, False),
    ],
)
def test_a_plans_figure_is_shown_once_its_counts_allow_it(
    progress: dict[str, Any] | None, shown: bool
) -> None:
    """figure_shown: at once beside a sheet that is not blind; beside a blind
    sheet (or one whose blind is not said) only once every window is
    labelled; never when the counts cannot be read."""
    assert sheet.figure_shown(progress) is shown


def test_a_plans_figure_is_given_beside_its_sheet_once_the_counts_allow_it(
    root: Path,
) -> None:
    """plan_figure reads ``<stem>_figure.svg`` beside the sheet: refused with
    FIGURE_LATER while a blind sheet (a class map's, two maps') is not
    complete or its counts cannot be read, given once it is complete and at
    once beside a sheet labelled right or wrong; FIGURE_GONE when no figure a
    tool wrote is there; a path that is no plan's sheet is refused as
    resolve_sheet refuses it."""
    svg = '<svg xmlns="http://www.w3.org/2000/svg"><text>plan</text></svg>'
    class_map = _one_map(root)  # plan_to_label.csv, the class seen
    (root / "plan_figure.svg").write_text(svg, encoding="utf-8")
    with pytest.raises(ValueError) as held:
        sheet.plan_figure(str(class_map))
    assert str(held.value) == sheet.FIGURE_LATER
    sheet.save_labels(str(class_map), [{"window_index": 7, "reference_class": "water"}])
    with pytest.raises(ValueError, match="once the whole sheet is labelled"):
        sheet.plan_figure(str(class_map))  # part way
    sheet.save_labels(str(class_map), [{"window_index": 3, "reference_class": "trees"}])
    assert sheet.plan_figure(str(class_map)) == {"svg": svg}
    (root / "plan_figure.svg").unlink()
    (root / "plan_figure.svg").mkdir()  # not a file a tool wrote
    with pytest.raises(ValueError) as gone:
        sheet.plan_figure(str(class_map))
    assert str(gone.value) == sheet.FIGURE_GONE
    # Two maps: blind too, so the figure where the maps differ waits as well.
    pair = _write(
        root / "pair_to_label.csv", TWO_MAP, [["1", "4", "0", "4", "a", "b", ""]]
    )
    (root / "pair_figure.svg").write_text(svg, encoding="utf-8")
    with pytest.raises(ValueError, match="once the whole sheet is labelled"):
        sheet.plan_figure(str(pair))
    # No design file: labelled right or wrong, the map's class shown: at once.
    plain = _write(
        root / "plain_to_label.csv",
        ONE_MAP,
        [["1", "7", "0", "7", "water", "10.5", "50.25", "0", "", ""]],
    )
    (root / "plain_figure.svg").write_text(svg, encoding="utf-8")
    assert sheet.plan_figure(str(plain)) == {"svg": svg}
    # Counts that cannot be read (a sheet saved in another encoding): held.
    (root / "odd_to_label.csv").write_bytes(b"order,window_index\n\xff\xfe\n")
    (root / "odd_figure.svg").write_text(svg, encoding="utf-8")
    with pytest.raises(ValueError) as odd:
        sheet.plan_figure(str(root / "odd_to_label.csv"))
    assert str(odd.value) == sheet.FIGURE_LATER
    with pytest.raises(ValueError, match="does not exist"):
        sheet.plan_figure(str(root / "gone_to_label.csv"))
    with pytest.raises(ValueError, match="Only a plan's"):
        sheet.plan_figure(str(root / "plain_figure.svg"))


def test_the_plan_card_says_the_servers_words_for_a_held_figure() -> None:
    """result.html's sentence in a held figure's place is FIGURE_LATER, and the
    card asks the server's panel-only tool for the figure."""
    page = (Path(sheet.__file__).parent / "views" / "result.html").read_text(
        encoding="utf-8"
    )
    assert f'const FIGURE_LATER = "{sheet.FIGURE_LATER}";' in page
    assert "'olmoearth_app_plan_figure'" in page


def test_the_page_knows_the_servers_refusal_of_a_changed_sheet() -> None:
    """sheet.js offers to reopen the sheet on CHANGED_UNDER_PAGE, by its words."""
    script = (Path(sheet.__file__).parent / "views" / "sheet.js").read_text(
        encoding="utf-8"
    )
    head = sheet.CHANGED_UNDER_PAGE.split(";")[0]
    assert f"OE.CHANGED_UNDER_PAGE = '{head}';" in script


def test_the_pictures_rule_takes_present_and_absent_from_the_sheet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from olmoearth_agent.mcp_app import imagery

    assert imagery.reading_rule(["Presence", "absence"]) == imagery.READING_RULE
    monkeypatch.setattr(sheet, "PRESENT_NAMES", frozenset({"kelp"}))
    assert imagery.reading_rule(["kelp", "absent"]) == imagery.READING_RULE
    assert imagery.reading_rule(["present", "absent"]) is None


#: What sheet_progress gives, and nothing else: counts, the job and the
#: sheet's version, the labelling rule and whether it was graded, no row.
PROGRESS_KEYS = {
    "path",
    "name",
    "kind",
    "design_path",
    "label_rule",
    "blind",
    "takes_unjudged",
    "n_rows",
    "n_labelled",
    "n_unjudged",
    "progress",
    "open_label",
    "job",
    "finish_label",
    "finish_call",
    "finish_message",
    "version",
    "reading_rule",
    "rule_scope",
    "rule_source",
    # The owner's rule is for kelp canopy maps only (r4i F9), else None.
    "rule_limit",
    "graded",
}


def _progress(path: Path) -> dict[str, Any]:
    """sheet_progress of a sheet, checked to hold only its keys and to agree
    with read_sheet on every key both give."""
    got = sheet.sheet_progress(str(path))
    assert set(got) == PROGRESS_KEYS
    read = sheet.read_sheet(str(path))
    for key in PROGRESS_KEYS - {"progress", "open_label"}:
        assert got[key] == read[key], key
    return got


def test_a_sheets_progress_is_counts_only(root: Path) -> None:
    """sheet_progress: the counts, the label rule and the finish fields, never
    a row, a place or a class; its line and button follow how far the sheet
    is labelled."""
    path = _one_map(root)
    fresh = _progress(path)
    dumped = json.dumps(fresh)
    for hidden in ("rows", "columns", "classes", "map_class"):
        assert hidden not in fresh
    for value in ("50.25", "50.26", "water", "trees", "map_class"):
        assert value not in dumped
    assert (fresh["kind"], fresh["label_rule"], fresh["job"]) == (
        "one_map",
        "class",
        "estimate",
    )
    assert (fresh["progress"], fresh["open_label"]) == (
        "0 of 2 labelled",
        "Open the sheet here",
    )
    sheet.save_labels(str(path), [{"window_index": 7, "wrong": "?"}])
    part = _progress(path)
    assert (part["n_labelled"], part["n_unjudged"]) == (1, 1)
    assert (part["progress"], part["open_label"]) == (
        "1 of 2 labelled, 1 cannot tell",
        "Continue labelling",
    )
    sheet.save_labels(str(path), [{"window_index": 3, "wrong": "0"}])
    done = _progress(path)
    assert (done["progress"], done["open_label"]) == (
        "2 of 2 labelled, 1 cannot tell",
        "Open the sheet",
    )
    # A seen sheet: Cannot tell is in both of its label cells, counted once.
    seen_path = _seen(root)
    seen = _progress(seen_path)
    assert (seen["label_rule"], seen["takes_unjudged"]) == ("seen", True)
    assert "50.2" not in json.dumps(seen)
    sheet.save_labels(str(seen_path), [{"window_index": 12, "reference_class": "?"}])
    assert _progress(seen_path)["progress"] == "1 of 3 labelled, 1 cannot tell"
    # A two-map sheet says Cannot tell in reference_class.
    (root / "pair.json").write_text(
        json.dumps({"design": "disagreement", "classes": ["crop", "water"]}),
        encoding="utf-8",
    )
    pair_path = _write(
        root / "pair_to_label.csv",
        TWO_MAP,
        [
            ["1", "4", "0", "4", "water", "crop", ""],
            ["2", "5", "0", "5", "crop", "", ""],
        ],
    )
    sheet.save_labels(str(pair_path), [{"window_index": 4, "reference_class": "?"}])
    pair = _progress(pair_path)
    assert (pair["kind"], pair["job"], pair["progress"]) == (
        "two_map",
        "which_map",
        "1 of 2 labelled, 1 cannot tell",
    )
    assert "crop" not in json.dumps(pair)
    # A certificate plan's: its job and the plan's alpha, no Cannot tell.
    cert_path = _seen(root, certificate=True)
    cert = _progress(cert_path)
    assert (cert["job"], cert["finish_label"], cert["takes_unjudged"]) == (
        "certify",
        "Certify the zone",
        False,
    )
    assert "alpha 0.05 (fixed by the plan)" in cert["finish_call"]


@pytest.mark.parametrize(
    ("labelled", "rows", "unjudged", "words"),
    [
        (0, 3, 0, ("0 of 3 labelled", "Open the sheet here")),
        (2, 3, 0, ("2 of 3 labelled", "Continue labelling")),
        (3, 3, 2, ("3 of 3 labelled, 2 cannot tell", "Open the sheet")),
        (0, 0, 0, ("0 of 0 labelled", "Open the sheet here")),
    ],
)
def test_the_plan_cards_words_follow_the_counts(
    labelled: int, rows: int, unjudged: int, words: tuple[str, str]
) -> None:
    assert sheet.progress_words(labelled, rows, unjudged) == words


# ------------------------------------------------------------ the page copy

#: A handle: "w" and 16 base32 characters, 17 in all; a sheet id, "s" and 16.
HANDLE = re.compile(r"w[a-z2-7]{16}")
SHEET_ID = re.compile(r"s[a-z2-7]{16}")


def _bare(root: Path) -> Path:
    """A one-map sheet with no design file beside it: right or wrong."""
    return _write(
        root / "bare_to_label.csv",
        ONE_MAP,
        [
            ["1", "7", "0", "7", "water", "10.5", "50.25", "3", "", ""],
            ["2", "3", "0", "3", "trees", "10.6", "50.26", "4", "", ""],
        ],
    )


def _pair(root: Path) -> Path:
    """A two-map sheet whose design lists two classes; the maps show a third."""
    (root / "pair.json").write_text(
        json.dumps({"design": "disagreement", "classes": ["crop", "water"]}),
        encoding="utf-8",
    )
    return _write(
        root / "pair_to_label.csv",
        TWO_MAP,
        [
            ["1", "4", "0", "4", "water", "forest", ""],
            ["2", "5", "0", "5", "crop", "water", ""],
        ],
    )


def _cert(root: Path) -> Path:
    """A certificate plan's sheet on a scores map: right or wrong, no ?."""
    (root / "cert.json").write_text(
        json.dumps(
            {
                "design": "random",
                "population": {"score_kind": "window_confidence"},
                "certificate": {"seed_rule": "from_map", "alpha": 0.05},
            }
        ),
        encoding="utf-8",
    )
    return _write(
        root / "cert_to_label.csv",
        ONE_MAP,
        [
            ["1", "4", "0", "4", "1", "10.5", "50.25", "0", "", ""],
            ["2", "6", "0", "6", "0", "10.6", "50.26", "1", "", ""],
        ],
    )


#: The sheets a page copy is checked on: how each is made and its face.
PAGE_SHEETS = {
    "bare": (_bare, "wrong"),
    "seen": (_seen, "seen"),
    "class": (_one_map, "class"),
    "pair": (_pair, "pair"),
    "cert": (_cert, "wrong"),
}


@pytest.mark.parametrize("which", sorted(PAGE_SHEETS))
def test_the_page_copy_holds_handles_only(root: Path, which: str) -> None:
    """Each window is a handle and its answer (and the map's class on the
    right-or-wrong face alone): no window index, place, row, column, order
    or stratum anywhere in what the page gets."""
    make, face = PAGE_SHEETS[which]
    path = make(root)
    read = sheet.read_sheet(str(path))
    assert read["face"] == face and SHEET_ID.fullmatch(read["sheet_id"])
    keys = {"handle", "answer"} | ({"map_class"} if face == "wrong" else set())
    assert all(set(w) == keys for w in read["windows"])
    assert len(read["windows"]) == read["n_rows"] == len(_by_window(path))
    assert all(HANDLE.fullmatch(w["handle"]) for w in read["windows"])
    dumped = json.dumps(read)
    for value in ("10.5", "50.25", "50.26", "50.2"):
        assert value not in dumped
    for key in ("window_index", "order", "stratum", "row", "col", "lon", "lat"):
        assert f'"{key}"' not in dumped
    for gone in ("rows", "columns", "classes", "class_names", "design"):
        assert gone not in read


def test_handles_are_keyed_and_stable(root: Path) -> None:
    """A handle is 17 characters, the same at every read and after the module
    is loaded again (a server restart), other for another sheet with the
    same windows; the windows come in the order of their handles; a sheet
    drawn again with other windows refuses the old page's id."""
    path = _seen(root, windows=(4, 9, 12, 2, 7))
    first = sheet.read_sheet(str(path))
    handles = [w["handle"] for w in first["windows"]]
    assert all(HANDLE.fullmatch(h) and len(h) == 17 for h in handles)
    assert len(set(handles)) == 5 and handles == sorted(handles)
    again = sheet.read_sheet(str(path))
    assert [w["handle"] for w in again["windows"]] == handles
    assert again["sheet_id"] == first["sheet_id"]
    importlib.reload(sheet)
    reloaded = sheet.read_sheet(str(path))
    assert [w["handle"] for w in reloaded["windows"]] == handles
    assert reloaded["sheet_id"] == first["sheet_id"]
    # Another sheet of the same windows: other handles, another id.
    other = root / "other_to_label.csv"
    other.write_bytes(path.read_bytes())
    (root / "other.json").write_bytes((root / "kelp.json").read_bytes())
    elsewhere = sheet.read_sheet(str(other))
    assert not {w["handle"] for w in elsewhere["windows"]} & set(handles)
    assert elsewhere["sheet_id"] != first["sheet_id"]
    # The same windows drawn again, in another order: the same handles.
    _seen(root, windows=(7, 2, 12, 9, 4))
    assert sheet.read_sheet(str(path))["sheet_id"] == first["sheet_id"]
    # Other windows: the old page's id is refused, before any write.
    _seen(root, windows=(4, 9, 13))
    before = path.read_bytes()
    with pytest.raises(ValueError) as old:
        sheet.save_answers(
            str(path),
            sheet_id=first["sheet_id"],
            face="seen",
            rule="seen",
            answers=[],
        )
    assert str(old.value) == sheet.CHANGED_UNDER_PAGE
    with pytest.raises(ValueError, match="changed since the panel opened it"):
        sheet.window_row(str(path), first["sheet_id"], handles[0])
    assert path.read_bytes() == before


def test_the_handles_secret(root: Path, tmp_path: Path) -> None:
    """The secret is a hidden file beside the sheet, mode 0600, never written
    through a link; a malformed or deleted one is made anew, and pages read
    before then are refused."""
    path = _one_map(root)
    first = sheet.read_sheet(str(path))
    secret = root / ".plan_to_label.csv.handles"
    assert secret.is_file() and (secret.stat().st_mode & 0o777) == 0o600
    assert re.fullmatch(r"[0-9a-f]{64}", secret.read_text(encoding="ascii"))
    secret.write_text("not a secret", encoding="ascii")
    second = sheet.read_sheet(str(path))
    assert second["sheet_id"] != first["sheet_id"]
    assert re.fullmatch(r"[0-9a-f]{64}", secret.read_text(encoding="ascii"))
    assert (secret.stat().st_mode & 0o777) == 0o600
    with pytest.raises(ValueError, match="changed since the panel opened it"):
        sheet.save_answers(
            str(path),
            sheet_id=first["sheet_id"],
            face="class",
            rule="class",
            answers=[],
        )
    secret.unlink()
    assert sheet.read_sheet(str(path))["sheet_id"] != second["sheet_id"]
    for target in (tmp_path / "victim.txt", tmp_path / "nowhere.txt"):
        secret.unlink()
        if target.name == "victim.txt":
            target.write_text("keep", encoding="utf-8")
        secret.symlink_to(target)
        with pytest.raises(ValueError, match="is a link"):
            sheet.read_sheet(str(path))
        assert secret.is_symlink()
    assert (tmp_path / "victim.txt").read_text(encoding="utf-8") == "keep"
    assert not (tmp_path / "nowhere.txt").exists()


VERSION = re.compile(r"v[a-z2-7]{16}")


def test_a_sheets_version_names_its_bytes(root: Path) -> None:
    """The page copy and the counts carry one version, keyed by the sheet's
    own secret: not a plain hash of the file, and the same after the module
    is loaded again (a server restart keeps an open panel's version good). A
    save that writes changes it and answers the version the next read gives;
    a save that changes nothing keeps it; another secret gives another."""
    path = _one_map(root, design="class")
    read = sheet.read_sheet(str(path))
    first = read["version"]
    assert VERSION.fullmatch(first)
    assert sheet.sheet_progress(str(path))["version"] == first
    # Keyed by the sheet's secret, not the version's shape over an unkeyed
    # digest of the bytes (a plain hash, or the same digest under no key).
    raw = path.read_bytes()
    secret = sheet._sheet_secret(sheet.resolve_sheet(str(path)))
    assert first == sheet._version(secret, raw)
    for unkeyed in (
        hashlib.sha256(raw).digest(),
        hmac.new(b"", b"olmoearth-version\0" + raw, "sha256").digest(),
    ):
        assert first != "v" + base64.b32encode(unkeyed[:10]).decode("ascii").lower()
    importlib.reload(sheet)
    assert sheet.read_sheet(str(path))["version"] == first
    at = _by_window(path)
    good = {"sheet_id": read["sheet_id"], "face": "class", "rule": "class"}
    out = sheet.save_answers(
        str(path), **good, answers=[{"handle": at[7], "answer": "trees"}], version=first
    )
    assert out["cells_changed"] == 1 and VERSION.fullmatch(out["version"])
    assert out["version"] != first
    assert sheet.read_sheet(str(path))["version"] == out["version"]
    assert sheet.sheet_progress(str(path))["version"] == out["version"]
    same = sheet.save_answers(
        str(path),
        **good,
        answers=[{"handle": at[7], "answer": "trees"}],
        version=out["version"],
    )
    assert same["cells_changed"] == 0 and same["version"] == out["version"]
    (root / ".plan_to_label.csv.handles").write_text("0" * 64, encoding="ascii")
    assert sheet.read_sheet(str(path))["version"] != out["version"]


def test_a_save_at_an_older_version_is_refused(root: Path) -> None:
    """A save naming a version the sheet no longer has (another panel saved,
    a spreadsheet rewrote the CSV) is refused as stale, a ValueError of its
    own, with the file as it was. The sheet's id and the rule are checked
    first (CHANGED_UNDER_PAGE), then the version; a version that is not text
    is refused; a save at the version now, or with none (the file route's
    way), writes."""
    path = _one_map(root, design="class")
    read = sheet.read_sheet(str(path))
    at = _by_window(path)
    good = {"sheet_id": read["sheet_id"], "face": "class", "rule": "class"}
    first = sheet.save_answers(
        str(path),
        **good,
        answers=[{"handle": at[7], "answer": "water"}],
        version=read["version"],
    )
    before = path.read_bytes()
    with pytest.raises(sheet.StaleSheet) as old:
        sheet.save_answers(
            str(path),
            **good,
            answers=[{"handle": at[3], "answer": "crops"}],
            version=read["version"],
        )
    assert str(old.value) == sheet.STALE_SHEET and isinstance(old.value, ValueError)
    assert path.read_bytes() == before
    for change in ({"sheet_id": "s" + "a" * 16}, {"rule": "wrong"}):
        with pytest.raises(ValueError) as first_refusal:
            sheet.save_answers(
                str(path), **{**good, **change}, answers=[], version=read["version"]
            )
        assert str(first_refusal.value) == sheet.CHANGED_UNDER_PAGE, change
        assert not isinstance(first_refusal.value, sheet.StaleSheet)
    with pytest.raises(ValueError, match="version is the sheet's version"):
        sheet.save_answers(str(path), **good, answers=[], version=3)
    assert path.read_bytes() == before
    # A spreadsheet rewrites the CSV: the version the panel holds is old.
    _set_cell(path, "3", "reference_class", "trees")
    with pytest.raises(sheet.StaleSheet):
        sheet.save_answers(
            str(path),
            **good,
            answers=[{"handle": at[3], "answer": "crops"}],
            version=first["version"],
        )
    assert _cells(path, "reference_class") == {"7": "water", "3": "trees"}
    now = sheet.sheet_progress(str(path))["version"]
    done = sheet.save_answers(
        str(path), **good, answers=[{"handle": at[3], "answer": "crops"}], version=now
    )
    assert (
        done["cells_changed"] == 1 and _cells(path, "reference_class")["3"] == "crops"
    )
    loose = sheet.save_answers(
        str(path), **good, answers=[{"handle": at[3], "answer": "water"}]
    )
    assert loose["cells_changed"] == 1


@pytest.mark.parametrize("end", ["\r\n", "\n", "\r"])
def test_a_sheets_line_ends_read_as_a_text_read_did(root: Path, end: str) -> None:
    """The sheet's bytes are read once (its version names them) and parsed
    as a text read parsed them: \\r\\n, \\n and \\r line ends give the same
    rows, a line break inside a quoted cell included."""
    path = root / "plan_to_label.csv"
    lines = [
        ",".join(ONE_MAP),
        '1,7,0,7,water,10.5,50.25,0,,"a' + end + 'b"',
        "2,3,0,3,trees,10.6,50.26,0,,",
    ]
    path.write_bytes((end.join(lines) + end).encode("utf-8"))
    text = path.read_text(encoding="utf-8")
    then = [row for row in csv.reader(io.StringIO(text, newline="")) if row]
    header, body = sheet._read(path)
    assert [header, *body] == then
    assert body[0][ONE_MAP.index("reference_class")] == "a\nb"
    copy_ = sheet.read_sheet(str(path))
    assert copy_["version"] == sheet.sheet_progress(str(path))["version"]


def test_the_page_takes_in_a_sheet_saved_elsewhere() -> None:
    """common.js reads the server's stale refusal (app_error's stale), and
    sheet.js sends the version it holds, asks for the version when shown
    again, and reads the sheet again on a stale refusal."""
    views = Path(sheet.__file__).parent / "views"
    common = (views / "common.js").read_text(encoding="utf-8")
    script = (views / "sheet.js").read_text(encoding="utf-8")
    assert "err.stale = Boolean(body && body.stale);" in common
    assert "args.version = data.version" in script and "if (err.stale)" in script
    assert "'olmoearth_app_sheet_progress'" in script


def test_a_save_by_handle_lands_on_its_window(root: Path) -> None:
    """A seen sheet: the code and the wrong worked out from it land on the
    window the handle names; an unknown handle, an old sheet id, a window
    index, a value that is no answer, another rule or face, and ? on a
    certificate plan's sheet are each refused, with the file as it was."""
    path = _seen(root)
    page = sheet.read_sheet(str(path))
    at = _by_window(path)
    sid = page["sheet_id"]
    out = sheet.save_answers(
        str(path),
        sheet_id=sid,
        face="seen",
        rule="seen",
        answers=[{"handle": at[9], "answer": "1"}],
    )
    assert (out["cells_changed"], out["n_labelled"]) == (1, 1)
    assert (out["mix_problem"], out["clear_column"]) == (None, None)
    assert _cells(path, "reference_class") == {"4": "", "9": "1", "12": ""}
    assert _cells(path, "wrong") == {"4": "", "9": "1", "12": ""}
    shown = {w["handle"]: w["answer"] for w in sheet.read_sheet(str(path))["windows"]}
    assert shown == {at[4]: "", at[9]: "1", at[12]: ""}
    good = {"sheet_id": sid, "face": "seen", "rule": "seen"}
    refusals: list[tuple[dict[str, Any], str]] = [
        (
            {"answers": [{"handle": "w" + "z" * 16, "answer": "1"}]},
            sheet.UNKNOWN_HANDLE,
        ),
        ({"answers": [{"handle": 4, "answer": "1"}]}, sheet.UNKNOWN_HANDLE),
        (
            {"sheet_id": "s" + "a" * 16, "answers": [{"handle": at[4], "answer": "1"}]},
            sheet.CHANGED_UNDER_PAGE,
        ),
        ({"answers": [{"window_index": 4, "answer": "1"}]}, sheet.OLDER_PANEL),
        (
            {"answers": [{"handle": at[4], "answer": "present"}]},
            "present is not one of this sheet's answers: Present (1), Absent (0), "
            "Cannot tell (?).",
        ),
        ({"answers": [{"handle": at[4], "answer": 1}]}, "an answer is text"),
        ({"answers": "1"}, "answers must be a list"),
        (
            {"rule": "wrong", "answers": [{"handle": at[4], "answer": "1"}]},
            sheet.CHANGED_UNDER_PAGE,
        ),
        (
            {"face": "wrong", "answers": [{"handle": at[4], "answer": "1"}]},
            sheet.CHANGED_UNDER_PAGE,
        ),
        ({"clear_column": "reference_class"}, sheet.NOT_CLEARED),
    ]
    for change, said in refusals:
        before = path.read_bytes()
        with pytest.raises(ValueError) as caught:
            sheet.save_answers(str(path), **{**good, **change})
        assert said in str(caught.value), change
        assert path.read_bytes() == before
    # A certificate plan's sheet takes no Cannot tell.
    cert = _seen(root, certificate=True)
    sid = sheet.read_sheet(str(cert))["sheet_id"]
    before = cert.read_bytes()
    with pytest.raises(ValueError, match=r"\? is not one of this sheet's answers"):
        sheet.save_answers(
            str(cert),
            sheet_id=sid,
            face="seen",
            rule="seen",
            answers=[{"handle": at[4], "answer": "?"}],
        )
    assert cert.read_bytes() == before


@pytest.mark.parametrize(
    ("which", "face", "values", "keys", "open_"),
    [
        ("seen", "seen", ["1", "0", "?"], [["P", "1"], ["A", "2"], ["?", "3"]], False),
        ("seen_cert", "seen", ["1", "0"], [["P", "1"], ["A", "2"]], False),
        ("bare", "wrong", ["0", "1", "?"], [["R"], ["W"], ["?"]], False),
        ("cert", "wrong", ["0", "1"], [["R"], ["W"]], False),
        (
            "class",
            "class",
            ["water", "trees", "crops", "?"],
            [["1"], ["2"], ["3"], ["?"]],
            True,
        ),
        (
            "pair",
            "pair",
            ["crop", "water", "forest", "?"],
            [["1"], ["2"], ["3"], ["?"]],
            True,
        ),
    ],
)
def test_each_sheet_sends_its_allowed_answers(
    root: Path,
    which: str,
    face: str,
    values: list[str],
    keys: list[list[str]],
    open_: bool,
) -> None:
    """The answers a face takes, with their keys: a seen sheet's codes, right
    or wrong (no digits), a class map's classes then Cannot tell (key ? alone,
    the last round's S4), a two-map sheet's (those the maps show besides,
    sorted); Cannot tell never on a certificate plan's."""
    if which == "seen_cert":
        path = _seen(root, certificate=True)
    else:
        path = PAGE_SHEETS[which][0](root)
    read = sheet.read_sheet(str(path))
    assert read["face"] == face and read["open"] is open_
    assert _answer_values(read) == values
    assert [a["keys"] for a in read["answers"]] == keys


def test_a_class_maps_faces(root: Path) -> None:
    """A class map's sheet: the class seen first, right or wrong (with the
    map's class) only before any class seen is saved, the class seen only
    before any right or wrong; a mix made by hand is said with the column to
    clear; an unblind press adds nothing on the class face (the owner's O1 of
    14 Nov 2026: no switch on a class sheet, and no stratum on any page)."""
    path = _one_map(root)
    blank = sheet.read_sheet(str(path))
    sid = blank["sheet_id"]
    at = _by_window(path)
    assert (blank["face"], blank["faces"]) == ("class", ["class", "wrong"])
    assert "has_stratum" not in blank and blank["has_place"] is True
    assert all(set(w) == {"handle", "answer"} for w in blank["windows"])
    wrong = sheet.read_sheet(str(path), "wrong")
    assert {w["handle"]: w["map_class"] for w in wrong["windows"]} == {
        at[7]: "water",
        at[3]: "trees",
    }
    assert "has_stratum" not in wrong
    assert all("stratum" not in w for w in wrong["windows"])
    with pytest.raises(ValueError, match="face is one of"):
        sheet.read_sheet(str(path), "sideways")
    with pytest.raises(ValueError) as other:
        sheet.read_sheet(str(path), "seen")
    assert str(other.value) == sheet.CHANGED_UNDER_PAGE
    on_class = {"sheet_id": sid, "face": "class", "rule": "class"}
    sheet.save_answers(
        str(path), **on_class, answers=[{"handle": at[7], "answer": "water"}]
    )
    with pytest.raises(ValueError) as refused:
        sheet.read_sheet(str(path), "wrong")
    assert str(refused.value) == sheet.FACE_WRONG_REFUSED
    before = path.read_bytes()
    with pytest.raises(ValueError, match="offered only before any class seen"):
        sheet.save_answers(
            str(path),
            sheet_id=sid,
            face="wrong",
            rule="class",
            answers=[{"handle": at[3], "answer": "1"}],
        )
    assert path.read_bytes() == before
    # The classes seen cleared: right or wrong is offered again.
    cleared = sheet.save_answers(str(path), **on_class, clear_column="reference_class")
    assert cleared["cells_changed"] == 1
    assert _cells(path, "reference_class") == {"7": "", "3": ""}
    sheet.save_answers(
        str(path),
        sheet_id=sid,
        face="wrong",
        rule="class",
        answers=[{"handle": at[3], "answer": "1"}],
    )
    assert _cells(path, "wrong") == {"7": "", "3": "1"}
    with pytest.raises(ValueError) as refused:
        sheet.read_sheet(str(path), "class")
    assert str(refused.value) == sheet.FACE_CLASS_REFUSED
    # A class seen typed in a spreadsheet beside it: a mix, on the class face.
    _set_cell(path, "7", "reference_class", "water")
    mixed = sheet.read_sheet(str(path))
    assert (mixed["face"], mixed["faces"]) == ("class", ["class"])
    assert mixed["mix_problem"] == "it also has 1 label in its wrong column"
    assert mixed["clear_column"] == "wrong"
    # An unblind press on the class face is ignored: no map class, no stratum.
    shown = sheet.read_sheet(str(path), unblind=True)
    assert "unblinded" not in shown
    assert all(set(w) == {"handle", "answer"} for w in shown["windows"])
    with pytest.raises(ValueError, match="not cleared"):
        sheet.save_answers(str(path), **on_class, clear_column="map_class")
    out = sheet.save_answers(str(path), **on_class, clear_column="wrong")
    assert (out["mix_problem"], out["clear_column"]) == (None, None)
    assert _cells(path, "wrong") == {"7": "", "3": ""}
    # Both columns full: wrong must agree with the class seen everywhere.
    for window, ref, said in (("7", "water", "0"), ("3", "crops", "1")):
        _set_cell(path, window, "reference_class", ref)
        _set_cell(path, window, "wrong", said)
    assert "mix_problem" not in sheet.read_sheet(str(path))
    _set_cell(path, "3", "wrong", "0")
    assert sheet.read_sheet(str(path))["mix_problem"] == (
        "its wrong column disagrees with the classes seen"
    )


def test_a_two_map_sheets_pair_comes_at_the_press(root: Path) -> None:
    """A two-map sheet sends no pair until the blind switch's press, then each
    window's two classes."""
    path = _pair(root)
    at = _by_window(path)
    assert '"pair"' not in json.dumps(sheet.read_sheet(str(path))["windows"])
    shown = sheet.read_sheet(str(path), unblind=True)
    assert shown["unblinded"] is True
    assert {w["handle"]: w["pair"] for w in shown["windows"]} == {
        at[4]: ["water", "forest"],
        at[5]: ["crop", "water"],
    }
    assert "has_stratum" not in shown
    # A seen sheet has nothing to show: the press adds nothing.
    seen = sheet.read_sheet(str(_seen(root)), unblind=True)
    assert "unblinded" not in seen
    assert all(set(w) == {"handle", "answer"} for w in seen["windows"])


def test_a_windows_map_link_is_given_by_handle_at_the_press(root: Path) -> None:
    """window_link: one window's OpenStreetMap link, by its handle and the
    sheet's id; refused for an unknown handle, an old id, a row with no place."""
    path = _one_map(root)
    sid = sheet.read_sheet(str(path))["sheet_id"]
    at = _by_window(path)
    assert sheet.window_link(str(path), sid, at[7]) == {
        "url": "https://www.openstreetmap.org/?mlat=50.25&mlon=10.5#map=17/50.25/10.5"
    }
    for handle, sheet_id, said in (
        ("w" + "a" * 16, sid, sheet.UNKNOWN_HANDLE),
        (None, sid, sheet.UNKNOWN_HANDLE),
        (at[7], "s" + "a" * 16, sheet.CHANGED_UNDER_PAGE),
        (at[7], None, sheet.CHANGED_UNDER_PAGE),
    ):
        with pytest.raises(ValueError) as caught:
            sheet.window_link(str(path), sheet_id, handle)
        assert str(caught.value) == said
    _set_cell(path, "3", "lat", "")
    with pytest.raises(ValueError) as none:
        sheet.window_link(str(path), sid, at[3])
    assert str(none.value) == sheet.NO_PLACE
    _set_cell(path, "3", "lat", "95")
    with pytest.raises(ValueError, match="no place"):
        sheet.window_link(str(path), sid, at[3])
    real, row, names = sheet.window_row(str(path), sid, at[3])
    assert (real, row["window_index"]) == (path.resolve(), "3")
    assert names == ["water", "trees", "crops"]


def test_the_page_knows_the_servers_face_refusals() -> None:
    """sheet.js offers to clear the column in the way on a refused face, by
    the server's words up to their ';'."""
    script = (Path(sheet.__file__).parent / "views" / "sheet.js").read_text(
        encoding="utf-8"
    )
    wrong = sheet.FACE_WRONG_REFUSED.split(";")[0]
    assert f'OE.FACE_WRONG_REFUSED = "{wrong}";' in script
    face = sheet.FACE_CLASS_REFUSED.split(";")[0]
    assert f"OE.FACE_CLASS_REFUSED = '{face}';" in script
    for gone in ("window_index", "OE.shuffled", "OE.osmLink"):
        assert gone not in script


# ------------------------------------------------------------ in progress

#: What each row under "In progress" gives, and nothing else.
IN_PROGRESS_KEYS = {
    "path",
    "labels_csv_path",
    "name",
    "job",
    "job_label",
    "progress",
    "saved",
    "resume_message",
}

#: When a fixture's sheet was last written (seconds): 14 Nov 2023, 22:13:20 UTC.
WRITTEN = 1_700_000_000


def _sheet_at(
    folder: Path,
    stem: str,
    wrongs: list[str],
    *,
    design: bool = True,
    when: float | None = None,
) -> Path:
    """A class map's sheet ``<stem>_to_label.csv`` in folder, one window per
    entry of wrongs ("" not labelled), its design ``<stem>.json`` beside it
    unless design is False, last written at when (seconds) when given."""
    if design:
        population = {"score_kind": "class_map", "classes": ["water", "trees"]}
        (folder / f"{stem}.json").write_text(
            json.dumps({"design": "random", "population": population}),
            encoding="utf-8",
        )
    rows = [
        [str(i + 1), str(i), "0", str(i), "water", "12.3456", "48.7654", "0", w, ""]
        for i, w in enumerate(wrongs)
    ]
    path = _write(folder / f"{stem}_to_label.csv", ONE_MAP, rows)
    if when is not None:
        os.utime(path, (when, when))
    return path


def test_only_part_labelled_sheets_beside_a_design_are_in_progress(
    root: Path, tmp_path: Path
) -> None:
    """sheets_in_progress lists a sheet at the top of the root with some
    windows labelled and not all, its design beside it: not a blank or full
    sheet, one without a design, a CSV that is not a sheet, a sheet in a
    subfolder, a hidden one, a link out of the root or one that cannot be
    read. Its row is counts only, as the plan card counts them, and no file
    is made beside any sheet."""
    part = _sheet_at(root, "part", ["0", "?", ""], when=WRITTEN)
    _sheet_at(root, "blank", ["", "", ""])
    _sheet_at(root, "full", ["0", "1", "0"])
    _sheet_at(root, "lone", ["0", ""], design=False)
    (root / "notes.csv").write_bytes(part.read_bytes())
    (root / "notes.json").write_text("{}", encoding="utf-8")
    (root / "sub").mkdir()
    _sheet_at(root / "sub", "deep", ["0", ""])
    _sheet_at(root, ".hidden", ["0", ""])
    outside = _sheet_at(tmp_path, "away", ["0", ""])
    (root / "link_to_label.csv").symlink_to(outside)
    (root / "link.json").write_bytes((tmp_path / "away.json").read_bytes())
    (root / "broken_to_label.csv").write_bytes(b"\xff\xfeorder,window_index\n")
    (root / "broken.json").write_text("{}", encoding="utf-8")
    before = sorted(os.listdir(root))
    got = sheet.sheets_in_progress()
    assert sorted(os.listdir(root)) == before  # no lock file, no secret
    assert got["more"] is False
    (row,) = got["sheets"]
    assert set(row) == IN_PROGRESS_KEYS
    real = os.path.realpath(part)
    assert (row["path"], row["name"]) == (real, "part_to_label.csv")
    # The projects panel tells Claude the sheet by its whole name and this path.
    assert row["labels_csv_path"] == real
    assert (row["job"], row["job_label"]) == (
        "estimate",
        "Checking how wrong a map is",
    )
    assert row["progress"] == "2 of 3 labelled, 1 cannot tell"
    assert row["progress"] == sheet.sheet_progress(str(part))["progress"]
    assert row["saved"] == "2023-11-14T22:13:20+00:00"
    assert row["resume_message"] == sheet.resume_message("part_to_label.csv", real)
    dumped = json.dumps(got)
    for hidden in ("48.765", "12.345", "water", "trees", "map_class", "window_index"):
        assert hidden not in dumped


def test_sheets_in_progress_are_newest_first_within_their_caps(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Newest first; at most IN_PROGRESS_MAX rows, a design read for each row
    listed and no other; at most IN_PROGRESS_READ sheets read; more says
    when sheets in progress may be left out."""
    for i in range(7):
        _sheet_at(root, f"s{i}", ["0", ""], when=WRITTEN + i)
    reads: list[str] = []
    designs: list[str] = []
    read, design_of = sheet._read, sheet._design_of

    def spy_read(real: Path) -> tuple[list[str], list[list[str]]]:
        """Note each sheet read."""
        reads.append(real.name.removesuffix(sheet.SHEET_SUFFIX))
        return read(real)

    def spy_design(real: Path) -> tuple[Path | None, dict[str, Any] | None]:
        """Note each design read."""
        designs.append(real.name.removesuffix(sheet.SHEET_SUFFIX))
        return design_of(real)

    monkeypatch.setattr(sheet, "_read", spy_read)
    monkeypatch.setattr(sheet, "_design_of", spy_design)

    def listed() -> tuple[list[str], bool]:
        """The stems listed, and more."""
        reads.clear()
        designs.clear()
        got = sheet.sheets_in_progress()
        stems = [r["name"].removesuffix(sheet.SHEET_SUFFIX) for r in got["sheets"]]
        return stems, got["more"]

    assert (sheet.IN_PROGRESS_MAX, sheet.IN_PROGRESS_READ) == (5, 25)
    assert listed() == (["s6", "s5", "s4", "s3", "s2"], True)
    assert designs == ["s6", "s5", "s4", "s3", "s2"]
    monkeypatch.setattr(sheet, "IN_PROGRESS_MAX", 3)
    assert listed() == (["s6", "s5", "s4"], True)
    assert designs == ["s6", "s5", "s4"] and reads == ["s6", "s5", "s4", "s3"]
    monkeypatch.setattr(sheet, "IN_PROGRESS_MAX", 10)
    monkeypatch.setattr(sheet, "IN_PROGRESS_READ", 2)
    assert listed() == (["s6", "s5"], True)
    assert reads == ["s6", "s5"] and designs == ["s6", "s5"]
    monkeypatch.setattr(sheet, "IN_PROGRESS_READ", 25)
    assert listed() == ([f"s{i}" for i in range(6, -1, -1)], False)


def test_a_seen_sheet_in_progress_is_counted_as_its_plan_card_counts(
    root: Path,
) -> None:
    """A class code typed in a spreadsheet, wrong not yet worked out, passes
    the filter but labels nothing: the sheet is left out. Once a save works
    wrong out, it is listed with its plan card's counts."""
    path = _seen(root)
    _set_cell(path, "4", "reference_class", "1")
    assert sheet.sheets_in_progress() == {"sheets": [], "more": False}
    sheet.save_labels(str(path), [{"window_index": 12, "reference_class": "?"}])
    (row,) = sheet.sheets_in_progress()["sheets"]
    assert row["progress"] == "2 of 3 labelled, 1 cannot tell"
    assert row["progress"] == sheet.sheet_progress(str(path))["progress"]
    assert "50.2" not in json.dumps(row) and "present" not in json.dumps(row)


def test_each_sheet_in_progress_says_what_its_labels_are_for(root: Path) -> None:
    """Each listed sheet's job in words: a two-map sheet's, a certificate
    plan's and a plan's estimate."""
    pair = _pair(root)
    sheet.save_labels(str(pair), [{"window_index": 4, "reference_class": "?"}])
    cert = _cert(root)
    sheet.save_labels(str(cert), [{"window_index": 4, "wrong": "0"}])
    _sheet_at(root, "plan", ["1", ""])
    rows = sheet.sheets_in_progress()["sheets"]
    jobs = {r["name"]: (r["job"], r["job_label"]) for r in rows}
    assert jobs == {
        "pair_to_label.csv": ("which_map", "Saying which map is more accurate"),
        "cert_to_label.csv": (
            "certify",
            "Labelling for a zone test (it may certify nothing)",
        ),
        "plan_to_label.csv": ("estimate", "Checking how wrong a map is"),
    }
    assert set(sheet.IN_PROGRESS_JOBS) == set(sheet.JOB_LABELS)


def test_sheets_in_progress_are_found_in_both_roots_once(
    root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The write root and the workspace root are each read once, a sheet
    listed once when the two are one folder; a root that is missing gives
    none."""
    _sheet_at(root, "here", ["0", ""], when=WRITTEN)
    scores = tmp_path / "scores"
    scores.mkdir()
    _sheet_at(scores, "there", ["0", ""], when=WRITTEN + 1)

    def names() -> list[str]:
        """The names listed."""
        return [r["name"] for r in sheet.sheets_in_progress()["sheets"]]

    assert names() == ["here_to_label.csv"]
    monkeypatch.setenv("OLMOEARTH_SCORES_ROOT", str(scores))
    assert names() == ["there_to_label.csv", "here_to_label.csv"]
    monkeypatch.setenv("OLMOEARTH_SCORES_ROOT", str(root))
    assert names() == ["here_to_label.csv"]
    monkeypatch.setenv("OLMOEARTH_SCORES_ROOT", str(tmp_path / "missing"))
    assert names() == ["here_to_label.csv"]


def test_the_resume_message_opens_the_sheet_again() -> None:
    """Resume asks Claude to open the sheet again with the sheet tool, by path."""
    assert sheet.resume_message("kelp_to_label.csv", "/out/kelp_to_label.csv") == (
        "Continue labelling my label sheet kelp_to_label.csv: open it with "
        "olmoearth_label_sheet, labels_csv_path /out/kelp_to_label.csv."
    )
