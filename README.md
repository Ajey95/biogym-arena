<p align="center">
  <img src="assets/biogym-arena-banner.svg" alt="BioGym Arena: a scientific agent moving through connected laboratory rooms" width="100%" />
</p>

<h1 align="center">BioGym Arena</h1>

<p align="center"><strong>A training and evaluation laboratory for bioinformatics AI agents.</strong></p>

<p align="center">
  <a href="#project-status"><img alt="Status: active build" src="https://img.shields.io/badge/status-active%20build-5d866c" /></a>
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-5b6d7d" /></a>
  <a href="#scientific-boundaries"><img alt="Scientific use: computational research" src="https://img.shields.io/badge/scientific%20use-computational%20research-7557a6" /></a>
</p>

<p align="center">
  <a href="#the-problem">The problem</a> ·
  <a href="#who-it-is-for">Who it is for</a> ·
  <a href="#what-we-plan-to-build">The plan</a> ·
  <a href="#evaluation">Evaluation</a> ·
  <a href="#related-research">Research</a>
</p>

## Project status

BioGym Arena is under active development. The repository contains a runnable three-room episode API, fixture cases, an OpenEnv WebSocket adapter, an ImmunoGraph MCP integration, a pixel laboratory frontend, a trainable starter policy, paired evaluation, and run exports. The current source-backed **LLM RL suite v2** has 80 training, 20 held-out test, and 20 transfer episodes over cached UniProt/RCSB records, including unsupported-claim audits and paired sequence/structure tasks. Its [training protocol](docs/rl-training.md) adds a train-only tool-use warm start before GRPO, followed by fresh-process test and transfer evaluation. A [measured Qwen3-0.6B LoRA run](results/README.md) completed 62 supervised and 32 GRPO optimizer steps; verified success changed from 5/40 to 9/40 on held-out test and from 3/40 to 6/40 in TransferArena. Both uncertainty intervals include zero, and neither suite establishes broad biological validity. A Sarvam 105B hosted trajectory collector and verified-success curator are implemented but have not made an authenticated model call. The starter policy remains a separate small neural action selector.

The [Product Requirements Document](docs/PRD.md) is the technical build reference. The [selected top-down frontend design](docs/design/top-down-flow/README.md) defines the visual journey.

### Run locally with the NVIDIA GPU

On the verified RTX 4060 Laptop GPU, the NVIDIA driver supports CUDA and the project uses a CUDA enabled PyTorch wheel in `.venv`. A separate CUDA Toolkit installation is not required for this PyTorch training path. On another Windows machine, install the wheel version appropriate for its driver from [PyTorch's official installer](https://pytorch.org/get-started/locally/), then install the backend:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install "torch==2.6.0" --index-url https://download.pytorch.org/whl/cu124
.\.venv\Scripts\python.exe -m pip install -e "backend[dev]" numpy
$env:BIOGYM_DEVICE = "cuda"
.\.venv\Scripts\python.exe -m uvicorn biogym.api:app --host 127.0.0.1 --port 8000
```

In a second terminal, run `npm install` and `npm run dev` in `frontend`. `BIOGYM_DEVICE=cuda` fails clearly if CUDA is unavailable; the default `auto` selects CUDA when present and CPU otherwise. Every trained policy records the selected device, GPU name, PyTorch version, CUDA runtime, optimizer steps, and checkpoint hash. The local 8 GB GPU runs the starter policy; Sarvam 30B adapter training needs substantially more model memory or a separately validated offload/cloud setup.

The OpenEnv server uses the same episode engine and durable event log as the app API:

```powershell
.\.venv\Scripts\python.exe -m uvicorn biogym.openenv_server:app --host 127.0.0.1 --port 8001
```

Its training service accepts authored `train` cases through OpenEnv `reset()`, `step()`, and `state()`. The hosted Sarvam trajectory collector requires `SARVAM_API_KEY` and consumes Sarvam credits when run: `python -m biogym.collect --cases seq-01 struct-01 --episodes 4`. It records model requests, actions, environment events, reward, source status, and a curated train-only JSONL with a hash manifest. No Sarvam trajectories or LLM policy updates are claimed until they are actually run.

## The problem

An MCP server can give an AI agent scientific tools. Tool access alone does not tell us whether the agent chooses appropriate tools, recovers from failures, preserves provenance, or draws conclusions supported by the returned evidence. An agent may complete a workflow yet still mishandle a corrupted input or misleading result.

[BioAgent Bench](https://proceedings.mlr.press/v306/fa26a.html) reports failures under controlled bioinformatics perturbations despite agents' ability to construct multi-step pipelines. [BixBench](https://arxiv.org/abs/2503.00096) evaluates open-ended computational biology analysis, while [BiomniBench](https://www.biorxiv.org/content/10.64898/2026.05.12.724604v2) studies evaluation of agents' analytical processes. These works motivate careful agent evaluation; BioGym Arena does not claim to have invented bioinformatics agent benchmarks or reinforcement-learning environments.

**Our question:** can repeatable scientific episodes, explicit evidence checks, and held-out transfer tasks help us measure and improve the reliability of bioinformatics agents?

## Who it is for

| User | Job BioGym Arena should help with |
| --- | --- |
| Scientific agent developers | Test agent policies and tool integrations before supervised research use; compare versions under the same conditions. |
| Bioinformatics and computational biology teams | Inspect an agent's decisions, evidence trail, failures, and reproducible run package. |
| AI evaluation and training researchers | Develop episodes and reward rubrics, test generalization, and study whether policy updates improve held-out performance. |
| Educators and learners | Practice valid scientific tool use and evidence interpretation in controlled scenarios. |

The initial scope is **protein sequence, protein structure, and epitope analysis**. The wider set of scientific domains is a roadmap, not current coverage.

## What we plan to build

### Scientific rooms and training modes

The frontend is conceived as a connected laboratory. A moving floor guides visitors between **scientific rooms**; each room contains sessions made up of replayable episodes. Direct navigation remains available for inspection.

| Initial room | Example tasks |
| --- | --- |
| Protein Sequence | Validate FASTA input, maintain sequence identity, and select appropriate sequence operations. |
| Protein Structure | Retrieve and validate structures and chains; distinguish predicted from experimental structural evidence. |
| Epitope Analysis | Execute bounded immunoinformatics workflows and qualify claims against the available evidence. |

Each room can host reusable **training modes**: ToolGym (tool selection), PipelineGym (multi-step execution), ChaosGym (fault recovery), EvidenceGym (evidence assessment), and, later, ResearchGym (adaptive planning). These are modes within rooms, not separate scientific datasets. The longer-term room roadmap also includes DNA, RNA-seq, variants, gene expression, metagenomics, docking, and drug-discovery workflow validation.

Every episode can measure two complementary skills:

- **LabOps:** valid execution, workflow planning, recovery, reproducibility, and resource use.
- **ResearchSense:** provenance assessment, uncertainty, evidence-supported conclusions, and justified abstention.

They share an episode and may both be tested by one challenge. Evaluation reports keep their measurements separate so operational success cannot conceal poor scientific judgment.

### Shared environment and scientific tools

The runtime implements [OpenEnv](https://huggingface.co/docs/openenv/index) `reset()`, `step()`, and `state()` over a WebSocket server: an agent receives observations, takes actions, and gets environment-side rewards. A **harness** controls permitted actions and budgets; a **loop** advances actions, observations, and rewards; an **execution graph** records dependencies and evidence lineage.

[ImmunoGraph MCP](https://github.com/Ajey95/immuno-graph) is the first scientific tool layer. Its [tool catalog](https://github.com/Ajey95/immuno-graph/blob/main/docs/TOOL_CATALOG.md) covers sequence validation, immune prediction, evidence and ranking, structure, chemistry, docking, and exports. The current BioGym adapter executes its released `validate_sequence` and `fetch_structure` tools, including a live RCSB PDB coordinate path. BioGym adds episodes, independent grading, agent trajectories, and transfer evaluation around those calls. The 46 MCP tools are **not** 46 independent live data providers; availability depends on configuration and external runtimes. See ImmunoGraph's [execution reality matrix](https://github.com/Ajey95/immuno-graph/blob/main/docs/TOOL_REALITY_MATRIX.md).

### TransferArena and visual replay

**TransferArena is present in the initial app.** It offers an unseen fixture case and a live RCSB structure task after recorded room attempts, with source status visible. The current policy comparison includes the fixture transfer case; automated before/after measurement on the live task remains work. Recorded, versioned inputs support repeatable comparisons when external records or availability change; replayed data must not be labelled live.

The interface uses a **pixel-art laboratory** with a small robot representing the agent as it moves through cells. Its position follows recorded episode events. The interface synchronizes:

- a 3D molecular view using coordinates relevant to the active episode;
- a live tool and evidence graph;
- a timeline that can pause, scrub, and replay decisions and failures;
- step-level rewards with explanations and LabOps / ResearchSense scorecards;
- clickable provenance from source record through tool result to agent claim.

See the [selected top-down frontend design](docs/design/top-down-flow/README.md) for the outer lab map, room interiors, episode interaction, TransferArena, and Evaluation Hall. [Earlier visual explorations](docs/design/frontend-visual-reference.md) remain available for comparison.

The **Evaluation Hall** compares a baseline with an updated starter policy on matched held-out fixture tasks. Attempt and evaluation ZIPs include inputs, actions, observations, tool outputs, reward breakdowns, policies, molecular coordinate artifacts where applicable, and an integrity manifest. Secrets and private credentials are excluded from exports.

Sarvam 105B has a hosted trajectory adapter awaiting an API key and a measured run. Smallest.ai spoken replay and Devin integration remain planned.

## Evaluation

We intend to publish **measured outcomes with their denominators and conditions**, not a single opaque score. An episode rubric should distinguish checkable task completion from the scientific interpretation of that completion.

| Measure | What it asks |
| --- | --- |
| Task success | Did the agent produce the specified, verifiable artifact or decision? |
| LabOps reliability | Were tool inputs valid, dependencies respected, and failures recovered without invented output? |
| Evidence integrity | Were sources and `LIVE`, `FIXTURE`, or `SYNTHETIC` statuses represented correctly? |
| ResearchSense judgment | Were claims supported, uncertainty stated, and abstention used when warranted? |
| Efficiency | How many tool calls, how much elapsed time, and what service cost did the task require? |
| Transfer performance | How did the agent perform on held-out and unseen tasks relative to its controlled-room results? |

For a before/after claim, the comparison should use held-out tasks, the same task definitions, tool access, budgets, and scoring rules, repeated episodes, and an identified baseline. The report should include the number of cases, per-task outcomes, reward definitions, model or configuration versions, episode seeds, source snapshots, and uncertainty appropriate to the sample size. The run package should preserve the evidence required to inspect a disputed result. Live sources can change; a saved snapshot supports replay, while a fresh live rerun may produce different data.

**“After training” is reserved for an actual policy update.** OpenEnv episodes supply actions, observations, and rewards; they do not themselves change model weights. The starter policy has its own policy-gradient update and held-out fixture comparison. Separately, the [Qwen3-0.6B LLM adapter pilot](results/README.md) used stateful tool episodes through TRL's `environment_factory`, changed adapter tensors over 16 optimizer steps, and was compared in fresh processes on 16 held-out cases. That adapter is not yet the app's serving policy. [TRL's environment integration](https://huggingface.co/docs/trl/openenv) describes the training contract.

## Scientific boundaries

BioGym Arena is intended for computational research and agent evaluation. A tool call, epitope prediction, docking pose, confidence score, or successful episode does not establish experimental binding, vaccine efficacy, safety, or clinical validity. Fixtures and synthetic exercises must remain visibly distinct from live scientific results. Expert review and experimental validation are outside an automated episode score.

## Related research

| Work | Relevance to this project |
| --- | --- |
| [BioAgent Bench (ICML 2026)](https://proceedings.mlr.press/v306/fa26a.html) | End-to-end bioinformatics tasks and robustness under controlled perturbations. |
| [BixBench (2025)](https://arxiv.org/abs/2503.00096) | Open-ended analysis of real computational biology scenarios. |
| [BixBench3 (2026 preprint)](https://arxiv.org/abs/2608.25286) | Research-study-scale tasks using raw biological data and multi-step analyses. |
| [BiomniBench (2026 preprint)](https://www.biorxiv.org/content/10.64898/2026.05.12.724604v2) | Process-level assessment of biomedical research agents. |

These are related efforts and research context, not BioGym Arena results or endorsements.

## Architecture diagrams

<!-- Reserved for diagrams grounded in an implemented and reviewed architecture. -->

## Results

<!-- Reserved for measured runs and reproducible evaluation artifacts. -->

## Contributing

The repository contains the early app and research plan. Contributions should include source-status honesty and reproducible evidence; see [CONTRIBUTING.md](CONTRIBUTING.md). Please use [SECURITY.md](SECURITY.md) for security reports.

## License

MIT. See [LICENSE](LICENSE).
