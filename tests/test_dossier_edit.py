"""Offline dry check of the assembler harness (no LLM, no network): render/IDs, edit ops,
the source gate, and a scripted fake-client session driving run_assembler_session end to end.

Run: uv run pytest tests/test_dossier_edit.py
"""

import json
from types import SimpleNamespace

import pytest

from dossier_edit import (
    ABSENCE_SECTIONS, SECTIONS, AssemblerState, ChangeLog, apply_absent, apply_append,
    apply_patch, entry_id, find_collision, run_assembler_session, section_size,
)
from dossier_response import (
    ConfidenceScore, Dossier, Grant, GrantCandidate, Publication, PublicationCandidate, Status,
)
from source_filter import Observations, check_entry, source_verdict

def _pub(title, doi):
    return PublicationCandidate(
        data=Publication(title=title, doi=doi),
        reasoning=f"source for {title}", confidence=ConfidenceScore.HIGH, status=Status.UNVERIFIED)

def _grant(title, award):
    return GrantCandidate(
        data=Grant(title=title, award_number=award, proposal_number=None),
        reasoning="CRIS record", confidence=ConfidenceScore.HIGH, status=Status.CONFIRMED)

def _media(url, title="Coverage", year=2024):
    return {"reasoning": "found in search", "confidence": "medium", "status": "unverified",
            "title": title, "url": url, "source": "Outlet", "snippet": "s", "year": year}

@pytest.fixture
def dossier():
    d = Dossier(net_id="t", first_name="A", last_name="B", last_known_college="COM",
                last_known_department="Med", degree="MD", initial_research_title="X", affiliation="UIC")
    d.publications = [_pub("Alpha study of kidneys", "10.1000/alpha"),
                      _pub("Beta trial of livers", "10.1000/beta")]
    d.grants.items = [_grant("Kidney grant", "E2103"), _grant("Liver grant", "E0965")]
    return d

@pytest.fixture
def obs():
    """Grounding evidence: one search observed the alpha DOI page; nothing else."""
    o = Observations()
    o.add_tool_call({"function": "search", "arguments": {"query": "q"},
                     "result_raw": {"results": [{"url": "https://doi.org/10.1000/alpha",
                                                 "content": "the alpha study"}]},
                     "result_distilled": {}})
    return o

GATE_OK = None

def _gate(o, cris=frozenset()):
    return lambda sec, e: check_entry(sec, e, o, cris)

def test_render_deterministic_and_ids_stable(dossier):
    s1, s2 = AssemblerState(dossier), AssemblerState(dossier.model_copy(deep=True))
    assert s1.render() == s2.render()
    assert s1.ids_of("publications") == s2.ids_of("publications")
    view = s1.render()
    assert "\n  " not in view.split("\n", 1)[1]
    for eid in s1.ids_of("publications"):
        assert eid in view

def test_ids_survive_patch(dossier):
    state = AssemblerState(dossier)
    eid = state.ids_of("publications")[0]
    r = apply_patch(state, "publications", eid, json.dumps({"reasoning": "updated"}))
    assert r.ok
    assert eid in state.ids_of("publications")

def test_append_success(dossier):
    state = AssemblerState(dossier)
    r = apply_append(state, "grants",
                     json.dumps({"reasoning": "new", "confidence": "medium", "status": "unverified",
                                 "data": {"title": "Lung grant", "award_number": "G9999"}}))
    assert r.ok and r.entry_id and "appended" in r.message
    assert section_size(state.dossier, "grants") == 3

def test_append_collision_names_existing_entry(dossier):
    state = AssemblerState(dossier)
    existing = state.ids_of("grants")[0]
    r = apply_append(state, "grants",
                     json.dumps({"reasoning": "dup", "confidence": "low", "status": "unverified",
                                 "data": {"title": "Kidney copy", "award_number": "E2103"}}))
    assert not r.ok and r.error_kind == "collision"
    assert existing in r.message and "patch that entry" in r.message
    assert section_size(state.dossier, "grants") == 2

def test_append_bad_json_and_unknown_section(dossier):
    state = AssemblerState(dossier)
    assert not apply_append(state, "publications", "{not json").ok
    assert not apply_append(state, "nope", "{}").ok

def test_append_keep_rule_drop_is_loud(dossier):

    state = AssemblerState(dossier)
    r = apply_append(state, "key_profiles.orcid",
                     json.dumps({"reasoning": "r", "confidence": "high", "status": "unverified",
                                 "url": ""}))
    assert not r.ok and r.error_kind == "schema"
    assert section_size(state.dossier, "key_profiles") == 0

def test_append_strips_model_set_source_flag(dossier):
    state = AssemblerState(dossier)
    e = _media("https://doi.org/10.1000/alpha")
    e["source_flag"] = "pre-flagged by the model"
    r = apply_append(state, "media_mentions", json.dumps(e))
    assert r.ok
    assert state.dossier.media_mentions.items[0].source_flag == ""

def test_gate_grounded_append_passes(dossier, obs):
    state = AssemblerState(dossier)
    r = apply_append(state, "media_mentions", json.dumps(_media("https://doi.org/10.1000/alpha")),
                     gate=_gate(obs))
    assert r.ok and not r.flagged
    assert state.dossier.media_mentions.items[0].source_flag == ""

def test_gate_rejects_then_accepts_flagged_on_exhaustion(dossier, obs):
    state = AssemblerState(dossier)
    bad = json.dumps(_media("https://nowhere.example/page9"))
    r1 = apply_append(state, "media_mentions", bad, gate=_gate(obs), max_attempts=3)
    r2 = apply_append(state, "media_mentions", bad, gate=_gate(obs), max_attempts=3)
    assert not r1.ok and not r2.ok
    assert r1.error_kind == "gate" and r1.gate_verdict == "ungrounded"
    assert "attempt 1/3" in r1.message and "attempt 2/3" in r2.message
    assert section_size(state.dossier, "media_mentions") == 0
    r3 = apply_append(state, "media_mentions", bad, gate=_gate(obs), max_attempts=3)
    assert r3.ok and r3.flagged and "flagged for the human reviewer" in r3.message
    items = state.dossier.media_mentions.items
    assert len(items) == 1 and items[0].source_flag.startswith("ungrounded")

    assert "source_flag" not in state.render() and "source_flag" not in r3.message

def test_gate_patch_cannot_break_grounded_source(dossier, obs):
    state = AssemblerState(dossier)
    apply_append(state, "media_mentions", json.dumps(_media("https://doi.org/10.1000/alpha")),
                 gate=_gate(obs))
    eid = state.ids_of("media_mentions")[0]
    r = apply_patch(state, "media_mentions", eid,
                    json.dumps({"url": "https://nowhere.example/x"}), gate=_gate(obs))
    assert not r.ok and r.error_kind == "gate"
    assert state.dossier.media_mentions.items[0].url == "https://doi.org/10.1000/alpha"

def test_gate_patch_grounding_clears_flag(dossier, obs):
    state = AssemblerState(dossier)
    bad = json.dumps(_media("https://nowhere.example/page9"))
    for _ in range(3):
        r = apply_append(state, "media_mentions", bad, gate=_gate(obs), max_attempts=3)
    assert r.ok and r.flagged
    eid = state.ids_of("media_mentions")[0]
    r = apply_patch(state, "media_mentions", eid,
                    json.dumps({"url": "https://doi.org/10.1000/alpha"}), gate=_gate(obs))
    assert r.ok and not r.flagged
    assert state.dossier.media_mentions.items[0].source_flag == ""

def test_gate_publications_exempt(dossier, obs):
    state = AssemblerState(dossier)
    r = apply_append(state, "publications",
                     json.dumps({"reasoning": "r", "confidence": "high", "status": "unverified",
                                 "data": {"title": "Gamma", "doi": "10.1000/gamma"}}),
                     gate=_gate(obs))
    assert r.ok and not r.flagged

def test_gate_cris_grant_grounds_via_export(dossier, obs):
    state = AssemblerState(dossier)
    r = apply_append(state, "grants",
                     json.dumps({"reasoning": "internal", "confidence": "high", "status": "confirmed",
                                 "data": {"title": "Internal", "proposal_number": "P777"}}),
                     gate=_gate(obs, cris={"P777"}))
    assert r.ok and not r.flagged

def test_patch_data_field_directly(dossier):
    state = AssemblerState(dossier)
    eid = state.ids_of("grants")[0]
    r = apply_patch(state, "grants", eid, json.dumps({"nih_project_num": "R01MH091811"}))
    assert r.ok
    assert any(g.data.nih_project_num == "R01MH091811" for g in state.dossier.grants.items)

def test_patch_unknown_id_lists_current(dossier):
    state = AssemblerState(dossier)
    r = apply_patch(state, "grants", "F-doesnotexist", json.dumps({"role": "PI"}))
    assert not r.ok and r.error_kind == "unknown_id"
    for eid in state.ids_of("grants"):
        assert eid in r.message

def test_patch_unknown_field_and_source_flag_handling(dossier):
    state = AssemblerState(dossier)
    eid = state.ids_of("publications")[0]
    assert apply_patch(state, "publications", eid, json.dumps({"nope": 1})).error_kind == "unknown_field"

    assert apply_patch(state, "publications", eid,
                       json.dumps({"source_flag": ""})).error_kind == "usage"
    r = apply_patch(state, "publications", eid,
                    json.dumps({"source_flag": "", "reasoning": "regrounded"}))
    assert r.ok and "harness-managed and was ignored" in r.message
    assert state.dossier.publications[0].reasoning == "regrounded" or \
        any(p.reasoning == "regrounded" for p in state.dossier.publications)

def test_patch_collision_rejected(dossier):
    state = AssemblerState(dossier)
    eid = state.ids_of("publications")[0]
    r = apply_patch(state, "publications", eid, json.dumps({"doi": "10.1000/beta"}))
    assert not r.ok and r.error_kind == "collision"
    assert {p.data.doi for p in state.dossier.publications} == {"10.1000/alpha", "10.1000/beta"}

def test_confirm_absent_lifecycle(dossier):
    state = AssemblerState(dossier)
    r = apply_absent(state, "economic_impacts", "searched patents + company registries; none")
    assert r.ok and state.dossier.economic_impacts.confirmed_absent

    assert not apply_absent(state, "grants", "reason").ok

    r = apply_append(state, "economic_impacts",
                     json.dumps({"reasoning": "r", "confidence": "low", "status": "unverified",
                                 "impact_type": "patent", "name": "US1", "summary": "s",
                                 "url": "https://patents.google.com/patent/US1"}))
    assert r.ok and not state.dossier.economic_impacts.confirmed_absent

    assert not apply_absent(state, "policy_impacts", "  ").ok
    assert not apply_absent(state, "publications", "reason").ok

def test_no_op_sequence_can_shrink_a_section(dossier, obs):
    state = AssemblerState(dossier)
    before = {s: len(state.pairs[s]) for s in SECTIONS}
    ops = [
        lambda: apply_append(state, "media_mentions", json.dumps(_media("https://nowhere.example/z")),
                             gate=_gate(obs)),
        lambda: apply_patch(state, "grants", state.ids_of("grants")[0], json.dumps({"role": "PI"}),
                            gate=_gate(obs)),
        lambda: apply_patch(state, "publications", "F-bogus", "{}"),
        lambda: apply_absent(state, "grants", "reason"),
    ]
    for op in ops:
        op()
    after = {s: len(state.pairs[s]) for s in SECTIONS}
    assert all(after[s] >= before[s] for s in SECTIONS)

def test_observation_keys_roundtrip_grounds_next_run(obs):
    keys = obs.export_keys()
    fresh = Observations()
    fresh.load_keys(json.loads(json.dumps(keys)))
    verdict, _ = source_verdict("https://doi.org/10.1000/alpha", fresh, set())
    assert verdict == "ok"
    verdict, _ = source_verdict("https://nowhere.example/page9", fresh, set())
    assert verdict == "ungrounded"

def test_changelog_rows(tmp_path, dossier, obs):
    cl = ChangeLog(tmp_path / "t.changes.jsonl", "t", source_run=2)
    state = AssemblerState(dossier)
    apply_append(state, "media_mentions", json.dumps(_media("https://nowhere.example/p")),
                 gate=_gate(obs), max_attempts=1, changelog=cl)
    eid = state.ids_of("media_mentions")[0]
    apply_patch(state, "media_mentions", eid, json.dumps({"year": 2020}), changelog=cl)
    rows = [json.loads(l) for l in (tmp_path / "t.changes.jsonl").read_text().splitlines()]
    ops = [r["op"] for r in rows]
    assert ops == ["append", "flag", "patch"]
    assert all(r["net_id"] == "t" and r["source_run"] == 2 for r in rows)
    assert rows[2]["key_or_anchor"] == eid and "2020" in rows[2]["after"]

class FakeClient:
    """responses.create returns scripted tool-call batches; [] ends the session."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    @property
    def responses(self):
        return self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        batch = self.script.pop(0) if self.script else []
        out = [SimpleNamespace(type="function_call", name=n, arguments=json.dumps(a),
                               call_id=f"c{len(self.calls)}_{i}") for i, (n, a) in enumerate(batch)]
        usage = SimpleNamespace(input_tokens=1000, output_tokens=50,
                                input_tokens_details=SimpleNamespace(cached_tokens=200))
        return SimpleNamespace(id=f"r{len(self.calls)}", usage=usage, output=out)

def test_assembler_session_end_to_end(dossier, obs):
    good = _media("https://doi.org/10.1000/alpha")
    bad = _media("https://nowhere.example/page9", title="Other")
    eid = entry_id("media_mentions", good)
    client = FakeClient([
        [("append_entry", {"section": "media_mentions", "entry": json.dumps(bad)})],
        [("append_entry", {"section": "media_mentions", "entry": json.dumps(good)})],
        [("patch_entry", {"section": "media_mentions", "id": eid,
                          "fields": json.dumps({"year": 2023})}),
         ("confirm_absent", {"section": "economic_impacts", "reason": "searched; none"})],
        [],
    ])
    out, rec = run_assembler_session(
        dossier, "### Turn 1\n(findings)", [{"type": "gap", "section": "media_mentions",
                                             "what": "w", "fix": "f"}],
        client=client, model="fake", net_id="t", obs=obs, cris=set(), max_turns=10)
    assert [m.year for m in out.media_mentions.items] == [2023]
    assert out.economic_impacts.confirmed_absent
    stats = rec["stats"]
    assert stats["append"] == 1 and stats["patch"] == 1 and stats["absent"] == 1
    assert stats["reject_gate"] == 1 and stats["gate_ungrounded"] == 1

    prompt = client.calls[0]["input"]
    assert prompt.index("RESEARCH TRANSCRIPT") < prompt.index("ISSUES TO RESOLVE") \
        < prompt.index("CURRENT DOSSIER")
    assert eid not in prompt

    assert all("previous_response_id" in c for c in client.calls[1:])

    assert all(c["tools"] == client.calls[0]["tools"] for c in client.calls)

def test_assembler_session_no_tool_calls_is_clean_stop(dossier):
    out, rec = run_assembler_session(dossier, "(no findings this run)", None,
                                     client=FakeClient([[]]), model="fake", net_id="t")
    assert rec["stats"] == {} or not any(k.startswith("reject") for k in rec["stats"])
    assert section_size(out, "publications") == 2

def test_render_headers_carry_entry_fields(dossier):
    view = AssemblerState(dossier).render()

    assert ("## key_profiles.orcid (0)  entry fields: reasoning, confidence(high|medium|low), "
            "status(unverified|confirmed|rejected), url") in view
    assert "data{pi_net_id" in view
    assert "impact_type(guideline|" in view
    assert "source_flag" not in view

def test_schema_error_repeats_entry_fields(dossier):
    state = AssemblerState(dossier)
    r = apply_append(state, "key_profiles.orcid",
                     json.dumps({"reasoning": "r", "confidence": "high", "status": "unverified",
                                 "data": {"url": "https://orcid.org/x"}}))
    assert not r.ok and "Entry fields for 'key_profiles.orcid'" in r.message

def test_missing_pubs_diff(dossier):
    from dossier_edit import _missing_pubs
    state = AssemblerState(dossier)

    expected = [
        {"title": "Alpha study of kidneys", "doi": "10.1000/alpha"},
        {"title": "BETA TRIAL OF LIVERS"},
        {"title": "Gamma paper", "doi": "10.1000/gamma", "year": 2020},
    ]
    missing = _missing_pubs(state, expected)
    assert [m["title"] for m in missing] == ["Gamma paper"]

def test_completeness_nudge_fires_once(dossier):
    gamma = {"reasoning": "from openalex", "confidence": "high", "status": "unverified",
             "data": {"title": "Gamma paper", "doi": "10.1000/gamma", "year": 2020}}
    client = FakeClient([
        [],
        [("append_entry", {"section": "publications", "entry": json.dumps(gamma)})],
        [],
    ])
    out, rec = run_assembler_session(
        dossier, "t", None, client=client, model="fake", net_id="t",
        expected_pubs=[{"title": "Gamma paper", "doi": "10.1000/gamma", "year": 2020}])
    assert rec["stats"]["completeness_nudge"] == 1
    assert any(p.data.doi == "10.1000/gamma" for p in out.publications)

    nudge_input = client.calls[1]["input"]
    assert isinstance(nudge_input, list) and "Gamma paper" in nudge_input[0]["content"]
    assert len(client.calls) == 3

class DyingClient(FakeClient):
    """First call raises (a deadline kill / dropped connection mid-build)."""

    def create(self, **kwargs):
        self.calls.append(kwargs)
        raise RuntimeError("model call exceeded wall-clock deadline")

def test_dead_session_records_missing_pubs(dossier):
    out, rec = run_assembler_session(
        dossier, "t", None, client=DyingClient([]), model="fake", net_id="t",
        expected_pubs=[{"title": "Gamma paper", "doi": "10.1000/gamma"},
                       {"title": "Delta paper", "doi": "10.1000/delta"}])

    assert rec["stats"]["call_failed"] == 1
    assert rec["stats"]["pubs_missing_at_end"] == 2
    assert len(out.publications) == 2

def test_post_nudge_leftover_becomes_durable_auto_skip(dossier):

    disp = {"skipped_records": []}
    client = FakeClient([[], []])
    _out, rec = run_assembler_session(
        dossier, "t", None, client=client, model="fake", net_id="t", dispositions=disp,
        expected_pubs=[{"title": "Gamma paper", "doi": "10.1000/gamma"}])
    assert rec["stats"]["completeness_nudge"] == 1
    assert "pubs_missing_at_end" not in rec["stats"]
    assert rec["stats"]["auto_skipped"] == 1
    assert disp["skipped_records"][0]["auto"] is True

    client2 = FakeClient([[]])
    _out, rec2 = run_assembler_session(
        dossier, "t", None, client=client2, model="fake", net_id="t", dispositions=disp,
        expected_pubs=[{"title": "Gamma paper", "doi": "10.1000/gamma"}])
    assert "completeness_nudge" not in rec2["stats"] and len(client2.calls) == 1

def test_skip_record_op(dossier):
    from dossier_edit import apply_skip, _missing_pubs, _disposed_keys, AssemblerState
    disp = {"skipped_records": []}
    offered = [{"title": "Ground failures paper", "doi": "10.9999/geology"}]
    assert apply_skip(disp, offered, "10.1/unknown", "not hers").error_kind == "usage"
    assert apply_skip(disp, offered, "10.9999/geology", "  ").error_kind == "usage"
    r = apply_skip(disp, offered, "10.9999/geology", "a geologist namesake, not this scholar")
    assert r.ok and disp["skipped_records"][0]["auto"] is False
    assert apply_skip(disp, offered, "Ground failures paper", "dup call").ok
    assert len(disp["skipped_records"]) == 1
    state = AssemblerState(dossier)
    assert _missing_pubs(state, offered, frozenset(_disposed_keys(disp))) == []

def test_skip_record_via_session_settles_the_record(dossier):
    disp = {"skipped_records": []}
    client = FakeClient([
        [],
        [("skip_record", {"record": "10.9999/geology",
                          "reason": "a geologist namesake, not this scholar"})],
        [],
    ])
    _out, rec = run_assembler_session(
        dossier, "t", None, client=client, model="fake", net_id="t", dispositions=disp,
        expected_pubs=[{"title": "Ground failures paper", "doi": "10.9999/geology"}])
    assert rec["stats"]["skip"] == 1 and "auto_skipped" not in rec["stats"]
    assert disp["skipped_records"][0]["reason"].startswith("a geologist namesake")

def test_missing_doi_filter_and_patch_clears_it(dossier):
    from dossier_edit import _run_filters
    state = AssemblerState(dossier)
    apply_append(state, "publications",
                 json.dumps({"reasoning": "r", "confidence": "high", "status": "unverified",
                             "data": {"title": "Gamma paper"}}))
    offered = [{"title": "Gamma paper", "doi": "10.1000/gamma"}]
    items = _run_filters(state, offered, None)
    assert len(items["missing_doi"]) == 1
    it = items["missing_doi"][0]
    assert it["doi"] == "10.1000/gamma"
    r = apply_patch(state, "publications", it["id"], json.dumps({"doi": it["doi"]}))
    assert r.ok
    assert "missing_doi" not in _run_filters(state, offered, None)

def test_nudge_cycle_progress_gate(dossier):
    gamma = {"reasoning": "r", "confidence": "high", "status": "unverified",
             "data": {"title": "Gamma paper", "doi": "10.1000/gamma"}}
    disp = {"skipped_records": []}
    client = FakeClient([
        [],
        [("append_entry", {"section": "publications", "entry": json.dumps(gamma)})],
        [],
        [],
    ])
    _out, rec = run_assembler_session(
        dossier, "t", None, client=client, model="fake", net_id="t", dispositions=disp,
        expected_pubs=[{"title": "Gamma paper", "doi": "10.1000/gamma"},
                       {"title": "Delta paper", "doi": "10.1000/delta"}])
    assert rec["nudge_cycles"] == 2
    assert rec["stats"]["auto_skipped"] == 1
    assert len(client.calls) == 4

    assert rec["end_reason"] == "stalled"
    assert rec["auto_skipped_labels"] == ["Delta paper"]
    assert all("turn" in op for op in rec["ops"])
    assert len(rec["filter_log"]) == 3 and rec["filter_log"][0]["turn"] == 1
    assert len(rec["nudges"]) == 2 and "Delta paper" in rec["nudges"][0]

def test_end_reason_clean_and_died(dossier, obs):
    _out, rec = run_assembler_session(dossier, "t", None, client=FakeClient([[]]),
                                      model="fake", net_id="t")
    assert rec["end_reason"] == "clean" and rec["label"] == "final"
    _out, rec = run_assembler_session(dossier, "t", None, client=DyingClient([]),
                                      model="fake", net_id="t", label="compaction", run=2)
    assert rec["end_reason"] == "call_failed" and rec["label"] == "compaction" and rec["run"] == 2

def test_completeness_nudge_silent_when_covered(dossier):
    client = FakeClient([[]])
    _out, rec = run_assembler_session(
        dossier, "t", None, client=client, model="fake", net_id="t",
        expected_pubs=[{"title": "Alpha study of kidneys", "doi": "10.1000/alpha"}])
    assert "completeness_nudge" not in rec["stats"] and len(client.calls) == 1

def test_empty_section_issues_deterministic_gap_check(dossier):
    from collect_dossier import _empty_section_issues
    gaps = {g["section"] for g in _empty_section_issues(dossier)}

    assert "publications" not in gaps and "grants" not in gaps
    assert {"key_profiles", "career_trajectory", "media_mentions", "clinical_impacts",
            "community_impacts", "economic_impacts", "policy_impacts"} == gaps

    dossier.economic_impacts.confirmed_absent = True
    dossier.economic_impacts.confirmed_absent_reason = "searched; none"
    assert "economic_impacts" not in {g["section"] for g in _empty_section_issues(dossier)}

def test_absence_sections_constant():

    assert SECTIONS == tuple(
        [f"key_profiles.{s}" for s in
         ("institutional_homepage", "google_scholar", "linkedin", "orcid", "personal_website")]
        + ["career_trajectory", "publications"] + list(ABSENCE_SECTIONS))
