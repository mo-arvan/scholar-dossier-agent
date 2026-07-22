# Scholar Dossier Agent

A human-in-the-loop AI agent that assembles an evidence dossier for a scholar and drafts
one-sentence impact summaries from it, framed by the Translational Science Benefits Model (TSBM),
for a person to accept, edit, or reject.

The agent works in research runs. A *gather* stage plans and runs its own searches across five
research databases (OpenAlex, PubMed, ORCID, ClinicalTrials.gov, NIH RePORTER) and the open web.
An *assemble* stage writes the findings into the dossier through entry-scoped operations (append,
patch, confirm-absent, skip-with-reason), each validated against the schema, checked for keyed
duplicates, and source-verified against what the tools actually observed before it lands; an entry
whose source cannot be verified is kept but flagged for the reviewer, never silently dropped, and
deterministic completeness checks make the session account for every publication record the
databases returned before it may end. Later runs edit the dossier in place rather than rewriting
it, and a record the agent explicitly skipped stays settled across runs. An isolated *critique*
stage then flags what needs judgment or new research (gaps, duplicates, contradictions, weak
sources) for the next run, and a *register-revise* stage tightens impact summaries that drift from
the target style. The agent is tuned for recall: it proposes generously so a reviewer prunes
rather than fills gaps.

This is the system described in "Real-World Evaluation of an AI Agent Drafting Translational Impact
Summaries."

## Requirements

Nothing here is optional unless marked so. The pipeline will not run without all of it.

**Azure OpenAI, not the OpenAI API.** The agent talks to an Azure OpenAI resource, addressed by
endpoint and key. An `api.openai.com` key will not work without changing the client code. You need
**two model deployments in the same region**:

| Deployment name | Set by | Role |
| --- | --- | --- |
| `gpt-5.1` | `conf/model/gpt5-1-east2.yaml` (`name`) | the loop: plans searches, reasons, drafts summaries |
| `gpt-5.4-mini` | `conf/dossier.yaml` (`cheap_model_name`) | the two high-volume reading subtasks: page distillation and abstract trimming |

The deployment names must match those strings, or point them at your own by editing the config. The
region comes from the selected model config's `azure_region` (`EASTUS2` by default) and decides
which environment variables are read: `AZURE_OPENAI_ENDPOINT_<REGION>` and
`AZURE_OPENAI_API_KEY_<REGION>`. Change the region and the variable names change with it.

**Tavily API key.** Web search, and the page fetch behind `structured_extract`. This is a paid
service and the agent is search-heavy: budget real credits per scholar, and try one scholar before
a batch.

**A contact email.** `CONTACT_EMAIL`, required, no default. OpenAlex and NCBI ask callers to
identify themselves so they can reach you about usage; the adapters raise rather than send your
traffic out under someone else's address. Use one you monitor.

**Python 3.11-3.13 and [uv](https://docs.astral.sh/uv/).**

**A scholars CSV**, at `dossier.scholars_path`. Every row runs; filter the file to the scholars
you want before pointing at it. Only `NetID`, `First Name`, and `Last Name` are required (`NetID`
is just the per-scholar key; any stable identifier works). `College`, `Department`, `Degree`,
`Research Project Title`, and `Affiliation` are optional and narrow the search when present.

Optional:

- `NCBI_API_KEY` raises the PubMed rate limit from 3 to 10 requests/second. Free. Works without it.
- `data/grant_data.csv` seeds the run with grants your institution already knows about, so the agent
  starts from them instead of rediscovering them. It expects the column names of the current
  research information system this was built against (`Proposal PI Net ID` and friends), so it is
  unlikely to match yours without edits. Absent, the run logs a warning and proceeds on public
  sources alone.

## Install

```bash
uv sync
cp .env.example .env    # then fill in the values above
```

## Run

```bash
uv run python src/collect_dossier.py hydra.run.dir=outputs/$(date +%Y%m%d_%H%M)
```

Configuration is Hydra (`conf/dossier.yaml`): `dossier.only=<netid>` restricts the run to one
scholar, which is the right way to try it first. Each run writes, per scholar, a JSON dossier, a
full trace (per-turn tool calls, every assembly operation with its outcome, and why each session
ended), a change log (`<netid>.changes.jsonl`), and two small sidecars the next run reads: the
grounding keyset (`<netid>.obs.json`) and the skip ledger (`<netid>.disp.json`, each skipped
record with its reason). A token and wall-clock summary covers the run.

```bash
uv run pytest
```

The tests are offline and mock the network; they need no keys.

## What is not here

This is a snapshot of the agent itself. The evaluation reported in the paper is not reproducible
from this repository, and deliberately so: it ran against one institution's internal grant records
and named scholars, which are not ours to publish. The analysis code, the review workbooks, the
dossiers, and the reference data all stay private. What ships is the agent, its prompts, and its
configuration, so the system can be read, run against your own inputs, and checked against the
paper's description.

The one data file included is `data/tsbm_style/tsbm_impact_statements.json`: the aggregate style
target the register-revise stage measures against, nine (mean, standard deviation) pairs derived
from 76 published benefit statements. It carries no personal data, and the stage cannot run without
it.

Anything you put under `data/` is yours and is gitignored, including the cache of everything a run
fetches about real people.

## Citation

If you use this work, please cite the preprint:

```bibtex
@misc{arvan2026realworldevaluationaiagent,
      title={Real-World Evaluation of an AI Agent Drafting Translational Impact Summaries},
      author={Mohammad Arvan and Amber E. Osterholt and Bailee Rue and Yuvaneswaren R. Sureshbabu and Krishna R. Patel and Rebecca T. Feinstein and Bethany C. Bray and Niranjan S. Karnik},
      year={2026},
      eprint={2607.16989},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2607.16989},
}
```

## License

MIT. See [LICENSE](LICENSE).
