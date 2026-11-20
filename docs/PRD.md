# BioGym Arena — Product Requirements Document

**Version:** 1.0

**Date:** 20 November 2026

**Status:** Approved product direction; implementation has not started

**Scope:** Complete product and technical build specification
**Authority:** This document is the build reference for BioGym Arena. The selected [top-down laboratory design](design/top-down-flow/README.md) governs the visual journey. If implementation exposes a conflict, record the decision and update this PRD before changing product behavior.

> **Implementation hold:** The owner has explicitly asked that product implementation wait until they say to start. Documentation, design, and planning do not constitute authorization to build the application.

## 1. Product definition

BioGym Arena is an instrumented training and evaluation laboratory for bioinformatics agents. An LLM agent enters a scientific room, attempts versioned episodes against real or clearly marked replayed scientific inputs, uses tools through a controlled interface, receives independently computed feedback, and undergoes an actual model-policy update. It then faces unseen work in TransferArena. Scientists and developers inspect the full action and evidence trail in a 2D pixel laboratory and download a reproducible run package.

The product asks a measurable question: **does training improve an agent's ability to execute scientific workflows and make evidence-supported judgments on held-out tasks?** The answer must be demonstrated by a frozen evaluation protocol and run artifacts, not inferred from a visually convincing replay.

### 1.1 Users and jobs

| User | Primary job | Required outcome |
| --- | --- | --- |
| Scientific agent developer | Train or compare tool-using policies | Reproduce an episode, identify failures, inspect real policy changes, compare held-out performance |
| Computational biologist or immunoinformatics researcher | Audit an agent's scientific work | Follow source → tool result → claim, distinguish computational inference from validation, export evidence |
| Evaluation researcher | Study reliability and transfer | Version tasks and rubrics, run paired evaluations, measure uncertainty and failure categories |
| Educator or learner | Explore scientific agent behavior | Replay an episode and understand valid tool use, recovery, and justified abstention |
| Hackathon judge | See a truthful, compelling demonstration | Navigate rooms, watch the robot execute/replay, enter TransferArena, inspect before/after evidence and ZIP |

The first production-quality scientific domains are **Protein Sequence, Protein Structure, and Epitope Analysis**. The full product roadmap has ten domains (§3.3). Visual coverage of a domain never implies that its scientific environment or grader exists.

### 1.2 Product promises

1. Every scored action has a versioned observation, permitted action, tool response, verifier finding, and durable event.
2. A training claim requires an actual trainable model, optimizer step, changed policy artifact, and a separate held-out evaluation. Hosted prompting or tool configuration changes receive different labels.
3. LabOps and ResearchSense are independently reported dimensions inside the same episodes.
4. TransferArena uses unseen cases and the same action contract as training. Its reveal path cannot expose hidden answers to the policy.
5. Provenance and source status are visible in the UI and machine-readable in exports.
6. The pixel game is an inspection interface over recorded execution. Animation cannot invent actions, outcomes, or biological data.

### 1.3 Non-goals and scientific boundary

BioGym Arena does not establish experimental binding, vaccine efficacy, drug safety, clinical validity, or biological truth from a computational score. It is not a clinical decision system. An ImmunoGraph tool response is an input to an independently checked episode, not an automatic correct answer. A completed pipeline can still receive a poor ResearchSense score or fail the task.

## 2. Release strategy and definition of “built”

**Version 1** includes the complete platform, the first three validated rooms, all five training modes where their tasks are meaningful, LLM trajectory collection, reward-driven LLM policy training, TransferArena, Evaluation Hall, provenance and ZIP. The **full domain vision** includes all ten rooms listed in §3.3. The remaining seven are subsequent implementation work under M7, not present capabilities. A room is “built” only when it has executable episode contracts, meaningful verifiers, reviewed scientific cases, replay, provenance, and evaluation coverage. A small custom neural action selector may prove that the runtime can update parameters, but it does **not** satisfy the LLM post-training requirement.

The **hackathon target** is an honest vertical slice of the same architecture: one or more high-quality episodes in each initial room, a real ImmunoGraph-backed task, at least one live source call where available, a fault/recovery case, TransferArena, and the selected visual journey. **A genuine RL-driven weight or adapter update and a separately scored held-out comparison are mandatory for this target.** If the time budget forces a tradeoff, narrow the number of episodes or visual polish before dropping the policy update or TransferArena. A configuration-only comparison is an explicitly incomplete fallback demonstration, never a substitute for this requirement or a “trained agent” claim.

### 2.1 Completion gates

| Gate | Evidence required |
| --- | --- |
| Executable environment | OpenEnv-compatible reset/step/state contracts, action validation, deterministic fixture replay, bounded live mode |
| Scientific episodes | Three initial rooms with reviewed case manifests, hidden graders, negative cases, failure injection, and source labels |
| Agent execution | Real tool calls through a gateway with budgets, structured errors, immutable traces, and safe retries |
| Policy training | Trainable policy, recorded optimizer updates, pre/post artifact hashes, checkpoint, reproducible training recipe |
| Evaluation | Frozen held-out split, paired baseline/updated-policy runs, task-level outcomes, uncertainty and stratified error report |
| TransferArena | Unseen task with independent score, source status, and no answer leakage |
| UI | Selected top-down map, room interiors, event-driven robot, graph/timeline sync, Evaluation Hall, 3D structure viewer when coordinates exist |
| Export | Complete and validated ZIP manifest with authorized artifacts, no credentials |
| Scientific review | Case and rubric review, provenance audit, limitation statements |

### 2.2 Requirement registry

The IDs below are stable traceability anchors for issues, implementation PRs, and acceptance evidence. Every P0 item is required for a release described as “BioGym Arena with trained-agent evaluation.” P1 items remain required for the full product but can follow the hackathon vertical slice.

| ID | Priority | Requirement | Acceptance evidence |
| --- | --- | --- | --- |
| ENV-01 | P0 | Versioned OpenEnv-compatible episode lifecycle | Reset/step/state contract and a replayed multi-step attempt |
| SCI-01 | P0 | Real scientific tools with source-status envelope | Pinned ImmunoGraph call and a source trace with hashes |
| SCI-02 | P0 | Reviewed cases in the three initial rooms | Case cards, hidden verifiers, positive/negative/fault examples |
| GRD-01 | P0 | Independent outcome, LabOps and ResearchSense grading | Per-event findings and separate score components |
| TRAJ-01 | P0 | Full LLM trajectories for successful and failed attempts, with a provenance-preserving curated subset for post-training | Conversation, tool/action/observation log, reward and filter decision per attempt |
| RL-01 | P0 | Actual reward-driven **LLM** policy update in the hackathon target and full product | Pre/post LLM or adapter hashes, optimizer steps, loadable checkpoint |
| EVAL-01 | P0 | Frozen paired held-out comparison | Per-case baseline/updated-policy results under matched conditions |
| XFER-01 | P0 | Unseen TransferArena task | Sealed case, independent score, no training exposure |
| XFER-02 | P0 | Sim-to-real loop from live failures into versioned training cases | Captured failure, reviewed case revision, retraining and re-evaluation lineage |
| PROV-01 | P0 | Durable event and evidence lineage | Append-only events and source → claim DAG |
| UX-01 | P0 | Selected 2D laboratory journey | Doors, room interiors, event-driven robot, truthful replay |
| EXP-01 | P0 | Downloadable integrity-checked run ZIP | Validated manifest, replay inputs, no secrets |
| UX-02 | P0 | Synchronized scientific viewers | Event-linked graph/timeline and molecular coordinates when applicable |
| OPS-01 | P0 | Recoverable, observable multi-run operation | Queue recovery, quotas, errors and trace correlation |
| DOM-01 | P1 | Expansion to other scientific domains | Each later room passes the same case, source and verifier gate |

## 3. Product experience

### 3.1 Navigation

The outer world is a **2D top-down pixel laboratory with the feel of a game garage**. A conveyor or moving-floor motif provides visual continuity. A small robot embodies the agent. Visitors may click a door or use direct navigation; movement never blocks access. They enter Protein Sequence, Protein Structure, and Epitope Analysis rooms, complete or inspect their required attempts, pass through the final TransferArena door, and arrive at Evaluation Hall.

A required episode is considered traversed when its attempt is durably recorded, **whether it passed or failed**. Failure remains visible. The TransferArena door explains missing attempts if locked. A recorded TransferArena attempt opens Evaluation Hall. Replays use the same event-based gate state.

### 3.2 Room grammar

Each room contains a Mission Desk for sessions and episodes; domain-specific tool workstations; a Fault Simulator for injected failures; an Evidence Board for source/result/claim lineage; a scientific artifact viewer; a Provenance Archive; and an Exit Gate. Tool calls move the robot to a workstation only after the run emits the corresponding action event. Clicking a station inspects information and never silently invokes a scientific tool.

The visual language must make four states distinct: not started, running, recorded attempt, and verified success/failure. The 3D molecular viewer is a focused overlay opened from a structure artifact with actual coordinates; it is not decorative molecule imagery. The graph, timeline, robot, reward event, and evidence panel seek to the same event ID during live playback or replay.

### 3.3 Scientific room roadmap

| Room | Initial or later | Candidate capabilities and sources | Scientific gate |
| --- | --- | --- | --- |
| Protein Sequence | Initial | FASTA validation, identifiers, sequence operations; UniProt and caller-supplied records | Identity, alphabet, length, provenance and expected transformation checks |
| Protein Structure | Initial | RCSB PDB, AlphaFold DB, ImmunoGraph structure tools | Chain/coordinate validity; experimental vs predicted origin; model confidence limits |
| Epitope Analysis | Initial | ImmunoGraph, IEDB-supported prediction and population coverage | Input validity, allele/method provenance, bounded claim language, population assumptions |
| DNA | Later | Curated public sequence records and variant-ready references | Reference build, strand, coordinate and identifier checks |
| RNA-seq | Later | Curated public expression data and reproducible analysis tools | QC, design matrix, batch effects and statistical interpretation |
| Variants | Later | Curated variant records and reference annotations | Genome assembly, allele normalization and evidence class |
| Gene Expression | Later | Public expression matrices and annotation sources | Normalization, contrasts, uncertainty and replication |
| Metagenomics | Later | Curated public sequence datasets and analysis tools | Contamination, taxonomic ambiguity and compositional limits |
| Docking | Later | Structures, ligands and configured local computational tools | Preparation, pose/protocol provenance, no experimental-affinity claim |
| Drug Discovery | Later | Public compound records and computational pipelines | Assay vs predicted distinction, chemistry checks and scope limits |

Candidate sources in later rooms are design candidates, not integrated providers. Each expansion needs a separate source, license, verifier, and expert-review decision. ImmunoGraph does **not** by itself cover all ten domains.

### 3.4 Training modes and evaluation dimensions

| Mode | Episode work | Mandatory verifier behavior |
| --- | --- | --- |
| ToolGym | Select and invoke a suitable tool with valid inputs | Check method suitability, schema, source choice, and artifact |
| PipelineGym | Build a dependent multi-step workflow | Check ordering, identity preservation, intermediate artifacts, and final result |
| ChaosGym | Recover from timeouts, malformed records, decoys, unavailable tools, or conflicting evidence | Check detection, bounded retry/fallback, no invented output, and correct failure report |
| EvidenceGym | Interpret evidence and provenance | Check claim support, uncertainty, source status, method limits, and abstention |
| ResearchGym | Plan and revise an investigation under budget | Check information gain, hypothesis revision, source triangulation, and defensible stopping |

**LabOps** scores valid execution, planning, recovery, reproducibility, and resource discipline. **ResearchSense** scores provenance understanding, evidence quality, uncertainty, unsupported claims, and justified abstention. They are score dimensions within episodes, not separate rooms or pages. Graders retain raw component scores and findings; they do not bury them in one composite.

## 4. Scientific data and tool integration

### 4.1 ImmunoGraph boundary

ImmunoGraph MCP is the first scientific tool gateway. Its tool catalog includes sequence, epitope, structural, chemistry, docking, provenance and export capabilities. The 46 tools are callable capabilities, **not** 46 independent live data providers. Some paths use configured live services or local scientific runtimes; others transform supplied data; some are fixtures or synthetic demonstrations. BioGym Arena must inspect the tool's actual response and execution mode for each call, not infer provenance from a tool name.

An ImmunoGraph workflow planner is not equivalent to an executed workflow. A plan-only result remains a plan. Computational epitope and docking results remain predictions. The integration adapter must be pinned to a tested ImmunoGraph version or commit and record tool schema/version, runtime configuration, upstream status, and output hashes.

### 4.2 Source-status vocabulary

Every data-bearing artifact uses one of **LIVE, CACHED, FIXTURE, SYNTHETIC** plus source ID, retrieval time, license/redistribution policy, transformation lineage, and content digest. LIVE means fetched during that run from the named upstream; CACHED means previously fetched content with capture metadata; FIXTURE means a reviewed, versioned test case; SYNTHETIC means generated exercise content. A live-service error or local simulation cannot be relabeled LIVE. Unknown status is an error that prevents a scientific-integrity pass.

Initial source candidates are [UniProt](https://www.uniprot.org/help/api), [RCSB PDB](https://data.rcsb.org/index.html), [AlphaFold Protein Structure Database](https://alphafold.ebi.ac.uk/), and [IEDB next-generation tools](https://nextgen-tools.iedb.org/docs/api/endpoints/), with ImmunoGraph as the adapter where appropriate. The population-coverage endpoint has its own biological assumptions and input rules. At implementation time, validate current endpoint behavior, terms, quotas, redistribution rights, method versions, and availability.

### 4.3 Source ingestion and lineage

Each source adapter returns a typed envelope: request, response status, upstream reference, capture timestamp, content digest, source-status enum, method/version, license metadata, error class, and an artifact reference. Store immutable raw responses where license permits, plus normalized data with transformation code version. If raw redistribution is barred, the ZIP includes a permitted citation, retrieval instructions, hash, and explicit “not bundled” reason. No silent substitutions.

## 5. Episode engine and environment contract

### 5.1 Core objects

| Object | Meaning |
| --- | --- |
| Room | Scientific domain and its authorized tool set |
| Session | Ordered or selectable collection of episode definitions |
| Episode definition | Versioned scenario, allowed actions, input/source policy, budgets, faults, hidden verifiers, reward rubric |
| Episode attempt | One reset-to-terminal interaction with a specific policy, seed, and source snapshot/live state |
| Run | Related training/evaluation attempts with complete configuration and lineage |
| Policy artifact | Model weights or adapter checkpoint plus tokenizer, chat template, decoding and tool schema |
| Evaluation suite | Frozen split, cases, rubrics, budget and metric definitions |

An OpenEnv-compatible environment exposes reset(), step(action), and state() around typed observations and actions. The environment owns task state and scoring. The policy never reads hidden expected answers, verifier code, reward internals beyond the documented feedback contract, or held-out case manifests. API contracts and episode manifests are schema-versioned and validated at load time.

### 5.2 Episode state machine

An attempt transitions through CREATED → READY → RUNNING → TERMINAL_SUCCESS, TERMINAL_FAILURE, TERMINAL_ABORTED, or TERMINAL_BUDGET_EXHAUSTED. Reset binds a case version, policy version, source snapshot or live policy, seed, fault schedule, and budgets. Step validates action schema and authorization, executes at most one approved tool action or environment operation, records an append-only event, updates observable state, invokes independent verifiers, computes reward, and returns observation/reward/done. Duplicate step requests use an idempotency key and cannot double-execute a chargeable or stateful tool.

Action types include inspect artifact, call tool, submit intermediate artifact, submit conclusion with evidence references, request clarification/abstain, and stop. The exact allowed set is episode-specific. A tool result is a structured observation with success/error code and provenance. Natural-language output cannot substitute for a missing required artifact.

### 5.2.1 Minimum wire contract

The following fields are required regardless of programming language. The API schema must give every enum, identifier, and artifact an explicit version. Large scientific payloads are stored as content-addressed artifacts rather than embedded in a model observation.

| Message | Required fields |
| --- | --- |
| Reset request | episode_version, case assignment token, policy_id, seed, source_mode, budget profile |
| Reset response | attempt_id, observation, allowed_action_schemas, remaining budgets, public task statement, done=false |
| Action | action_id/idempotency_key, attempt_id, action_type, typed arguments, evidence/artifact references where relevant |
| Step response | event_id, observation, reward_total, reward_components visible to the agent, done, terminal_reason, remaining budgets, artifact references |
| State response | attempt_id, public state, event cursor, terminal status, budgets and source statuses; no hidden answer or grader internals |
| Tool envelope | invocation_id, schema version, input/output hashes, source status, upstream/method versions, started/ended times, error class, artifact references |

The actual policy-visible reward fields are fixed per suite before training. A full verifier explanation may be delayed until a training episode ends and is withheld during sealed evaluation if it could reveal answers. Case assignment tokens are opaque; the agent cannot enumerate held-out cases.

### 5.3 Case design and difficulty

Each initial room needs positive, negative, ambiguous, and recovery cases. Cases must require actual interpretation or sequencing of tools, not just a scripted sequence of exact tool names. Difficulty comes from realistic constraints: malformed FASTA, mismatched accession/sequence, wrong structure chain, predicted-versus-experimental confusion, expired or unavailable service, conflicting prediction methods, decoy evidence, missing allele context, and resource limits. Faults are seeded and versioned; a fault injection is visible in the trace and never secretly edits the scientific “ground truth.”

Case authors write a case card with question, acceptable outputs, disallowed claims, verifiable artifacts, source provenance, expected failure modes, rubric, and expert-review sign-off. The scorer must accept scientifically equivalent valid routes where possible. Do not reward memorized wording or a single hard-coded call path. Include adversarial cases that tempt reward hacking, unsupported certainty, and provenance laundering.

### 5.4 Reward and verifier design

Rewards are computed by the environment after the action, never self-reported by the agent. The rubric includes:

- **Outcome:** independently verified artifact/decision and scientific constraints.
- **Process:** valid tool inputs, dependencies, recovery, and stopping behavior.
- **Evidence:** traceable source-to-claim links, correct source status, qualified conclusions, appropriate abstention.
- **Cost:** tool-call, elapsed-time, token and paid-service budgets.
- **Integrity gates:** fabricated results, unlicensed source use, hidden-answer access, or source-status misrepresentation can cap or invalidate a score.

Outcome and scientific-integrity checks dominate superficial format/efficiency rewards. Rewards and weights are published with version IDs; hidden answers are not. Capture both scalar training reward and decomposed grader findings. Evaluate reward quality manually with competent and deliberately weak trajectories before using it for RL. Test for shortcuts by changing labels, source order, distractors, and valid alternative methods. Rubric revisions create a new suite version and trigger reruns; historical results are not silently recomputed.

For each episode, specify a rubric vector rather than a universal hard-coded number: verified outcome, LabOps, ResearchSense, cost, and integrity findings. The scalar reward used by the trainer is a versioned weighted function of that vector with documented range and terminal bonuses/penalties. Critical integrity failures impose a cap regardless of successful tool execution. Step rewards may guide valid intermediate actions, but terminal success requires the independent outcome verifier. Never grant success because an LLM grader finds an answer plausible; any semantic judge is an auxiliary, versioned component with calibration examples and disagreement review. Multiple acceptable scientific paths are represented as invariants and evidence requirements, not one canonical command transcript.

### 5.5 Execution graph and audit log

The event log is append-only and authoritative. Each event contains run/attempt IDs, sequence number, wall and monotonic times, action/observation hashes, policy/tool/source/scorer versions, source status, budget deltas, parent event IDs, artifact references, and error details. A materialized execution DAG links inputs → actions → tool outputs → intermediate artifacts → claims → verifier findings → rewards. The UI graph and timeline derive from this log. Failed and superseded actions remain present. Corrections add events.

## 6. Actual policy training

### 6.1 Two agent tracks

**Track A: hosted Sarvam agent and trajectory generator.** Sarvam's hosted tool-capable API can attempt BioGym episodes, provide a strong baseline or teacher, and generate complete trajectories. Calls, prompts, model identifier, decoding, tool schemas, outputs, cost and rewards are logged. A hosted API call does not by itself update the hosted model. A changed prompt/tool set is labeled “revised configuration.”

**Track B: trainable Sarvam LLM.** [Sarvam 30B and 105B weights are published openly](https://www.sarvam.ai/blogs/sarvam-30b-105b); the [Sarvam 30B model card](https://huggingface.co/sarvamai/sarvam-30b) lists Apache-2.0 and is the preferred Sarvam adaptation target, subject to a compatibility and GPU pilot. Sarvam 105B is a strong hosted teacher/baseline but has a much larger checkpoint and multi-GPU burden. Use a frozen base/reference, parameter-efficient adapters where appropriate, and [TRL's OpenEnv integration](https://huggingface.co/docs/trl/openenv) for multi-turn GRPO-style optimization, or a documented PPO/verL path if compatibility requires it. Test the released model's custom architecture, chat/tool template, gradient flow, LoRA target modules, and trainer version before claiming support. The currently verified local RTX 4060 has 8 GB VRAM; this supports the starter neural policy but cannot hold Sarvam 30B weights entirely in GPU memory, even at 4-bit precision before training overhead. Sarvam 30B adaptation requires a separately validated offload or larger GPU setup. A smaller open-weight LLM may validate the LLM training path while Sarvam remains a hosted baseline; that is clearly a different model result. A hand-built small neural action policy is only a smoke test.

**2026-11-20 compute decision:** The connected Colab account currently enables T4 but disables A100, H100, and L4. The first source-backed LLM RL pilot therefore uses Qwen3-0.6B LoRA on Colab T4 with explicit model labeling. Its 24 cached UniProt/RCSB source records create 64 train, 16 held-out test, and 16 transfer episodes, with entity-disjoint splits, provenance and failure-recovery checks. See [the concrete training protocol](rl-training.md). This pilot validates LLM RL plumbing; it does not count as Sarvam 30B training or the complete scientific case suite. Sarvam 30B remains a separate larger-GPU compatibility and training run.

### 6.1.1 End-to-end post-training loop

1. **Trajectory generation:** Run a frozen base LLM (Sarvam hosted or open-weight where available) through thousands of varied, versioned OpenEnv episodes. Persist every user prompt, model response, tool call, environment observation, reward component, source envelope, terminal state and model/version settings. Preserve failures as first-class data.
2. **Curation:** Derive a quality-controlled success subset for optional supervised warm-start, while retaining failed, ambiguous, abstention and adversarial trajectories for analysis and future curriculum design. Filtering is versioned and auditable; do not train on the held-out suite or select only “easy successes.”
3. **Reward-driven post-training:** Use live multi-turn rollouts from the train split with GRPO/PPO or another justified policy optimizer. The environment independently grades each trajectory; the trainer updates LLM weights/adapters and saves checkpoints plus optimizer evidence.
4. **Benchmarking:** Freeze the new policy and compare it with its own pre-update baseline on sealed held-out cases, matched budgets, source snapshots and grader. Report per-case outcomes and error classes.
5. **Sim-to-real:** Run the frozen policy against unseen live scientific APIs in TransferArena. Capture failures, have a domain reviewer convert eligible failures into **new versioned training cases**, retrain the next policy version, and rerun held-out and live transfer evaluations. Never silently move an active transfer case into the training set and continue to call it held out.

### 6.2 Training sequence

1. Freeze environment, tool gateway, reward rubric, task splits, token/tool budgets, and baseline LLM policy.
2. Validate the model's structured tool-call competence. If needed, run supervised warm-start on authored or licensed training trajectories, separately recording its effect.
3. Collect and curate baseline LLM trajectories across the training and validation distributions. Confirm independent grading and inspect weak/strong examples; retain all original traces and filter decisions.
4. Train on training cases only with multi-turn rollouts. Record optimizer steps, learning rate, loss, KL/reference divergence where relevant, reward components, truncations, invalid actions, tokens, wall time, and compute cost.
5. Save immutable checkpoints and pre/post model or adapter digests. Verify at least one trainable parameter changed and that the resulting policy loads and acts.
6. Select a checkpoint using validation cases; never select on TransferArena/held-out test results.
7. Evaluate frozen baseline and selected updated policy on a held-out suite under matched conditions. Publish all per-case outcomes, including failures.
8. Retain rollback to baseline if safety, integrity, or evaluation gates regress.

Training is curriculum-based but does not collapse into easy toy tasks: begin with valid tool use, then dependency planning, corruption/recovery, evidence interpretation, and adaptive ResearchGym cases. Maintain a balanced sampling ledger so one source or easy episode cannot dominate the policy. Record data mixture and episode exposure. A run that only updates a prompt, memory, retrieval index, or hand-built rules cannot be called policy-weight training.

For GRPO, each sampled task produces multiple policy rollouts; the trainer uses relative reward within its sampled group, while the environment logs every full multi-turn trajectory. Group size, batch size, maximum context, rollout limit, clipping/KL settings, and adapter configuration are set from a measured compute pilot and frozen in the training recipe. Monitor invalid-call rate and reward-component distribution alongside aggregate reward so the model cannot improve its scalar return by abstaining universally, skipping expensive calls, or exploiting one easy case family. Stop or roll back on integrity regression, repeated reward exploitation, unstable divergence, or insufficient valid rollouts.

### 6.3 Leakage prevention

Split cases by biological entity/family, source record, task template family, and evidence pattern where feasible—not only random seeds. Keep held-out case IDs, answers, hidden verifier fixtures, and TransferArena prompts outside training storage and policy-visible observations. Deduplicate near-identical sequences and tasks across splits. The evaluation service issues sealed cases to a frozen policy and records access. Prompt injection in retrieved scientific text is treated as untrusted data and tested as an episode fault.

### 6.4 Proof of update

The training report includes baseline and updated policy IDs, weight/adapter hashes, checkpoint metadata, trainable parameter count, optimizer-step count, data/suite versions, hyperparameters, seed, code/container digest, training curves, and evaluation run IDs. A before/after visualization must resolve to this record. If the two policy artifacts are identical or no optimizer update is evidenced, the UI disables the “after training” label.

## 7. TransferArena and evaluation protocol

TransferArena is a mandatory final stage, not a bonus page. It presents unseen tasks through the same reset/step/state and tool gateway contract. Initial transfer cases should combine skills from at least two rooms, contain an authentic decision point, and include a source-status display. At least one demo path should reach an available live scientific service; versioned snapshots are used for fair comparisons and offline replay. A fresh live rerun is reported separately because upstream data can drift.

### 7.1 Before/after design

The primary comparison is **frozen baseline policy versus frozen updated policy on identical held-out cases**. Match tool access, schema, source snapshots, time/call/token budgets, scorer version, and execution environment. Use multiple sampling seeds or repeated attempts where model stochasticity matters. Pair results by base case and report each denominator. The main outcome is independently verified task success; LabOps and ResearchSense remain separate secondary outcomes. Do not compare a training-set score to a held-out score as a before/after result.

Pre-register suite, exclusions, primary endpoint, success criteria, and failure handling before running the final comparison. Report absolute difference and appropriate uncertainty (for example, case-clustered paired bootstrap intervals when sample size supports it). Small hackathon samples are descriptive, not proof of broad generalization. If an improvement claim is made, show the paired result, interval, case count, and safety/integrity regression checks; otherwise state the observed difference without a general claim.

### 7.2 Metrics

| Metric | Definition / denominator |
| --- | --- |
| Verified task success | Attempts satisfying all mandatory independent outcome checks / all issued attempts |
| LabOps reliability | Component rubric for valid actions, dependency handling, recovery and reproducibility |
| ResearchSense quality | Component rubric for evidence support, source-status accuracy, uncertainty and abstention |
| Provenance violation rate | Attempts with at least one material source or lineage misrepresentation / all attempts |
| Unsupported-claim rate | Attempts with at least one major unsupported scientific conclusion / all attempts |
| Correct abstention | Unanswerable/insufficient-evidence cases where agent stops with justified uncertainty / such cases |
| Recovery rate | Injected-fault cases reaching valid recovery or accurate failure report / injected-fault cases |
| Tool validity | Valid authorized calls / attempted calls, with invalid-call causes |
| Efficiency | Calls, tokens, wall time, retries, and service cost per attempt and per success |
| Transfer gap | Held-out TransferArena performance relative to matched in-room held-out performance, with case-mix caveat |
| Replay integrity | Reconstructed event/artifact hashes matching the recorded manifest / checked artifacts |

Report distributions and failure categories, not only averages. Stratify by room, mode, source status, fault type, and biological case family. Mark missing or unavailable upstream data explicitly rather than silently dropping an attempt.

### 7.3 Evaluation Hall

Evaluation Hall displays separate LabOps and ResearchSense scoreboards; matched baseline/updated-policy comparison; methods, budgets, denominators, suite and scorer versions; transfer results; source-status breakdown; and synchronized per-case replay. Its evidence wall traces source → tool result → claim → verifier finding → reward. If no valid comparison exists, show an empty state. A configuration comparison receives its actual label. Never pre-fill a scoreboard with illustration numbers.

## 8. System architecture

The following architecture is proposed, not implemented:

~~~mermaid
flowchart LR
    UI["2D pixel frontend<br/>map, rooms, replay, 3D viewer"] --> API["API and authorization"]
    API --> RUN["Run orchestration<br/>queue + workers"]
    RUN --> ENV["OpenEnv episode service<br/>reset / step / state"]
    ENV --> TOOL["Scientific tool gateway"]
    TOOL --> IMM["ImmunoGraph MCP"]
    TOOL --> SRC["Approved source adapters"]
    ENV --> VERIFY["Independent verifier<br/>and reward service"]
    RUN --> TRAIN["Policy trainer<br/>TRL + trainable model"]
    RUN --> EVAL["Held-out evaluation service"]
    RUN --> EVENTS["Append-only events"]
    EVENTS --> DB["PostgreSQL metadata"]
    EVENTS --> OBJ["Artifact object store"]
    TRAIN --> REG["Policy registry<br/>checkpoints + hashes"]
    EVAL --> REG
    API --> EXPORT["ZIP / manifest service"]
    EXPORT --> DB
    EXPORT --> OBJ
~~~

### 8.1 Proposed stack

- **Frontend:** React/TypeScript with a 2D game renderer such as PixiJS; accessible HTML panels for scientific detail; a real molecular-coordinate viewer such as Mol* for structure artifacts. Choose and pin versions during implementation.
- **Application API:** typed endpoints for rooms, sessions, episodes, runs, graph/timeline, policies, suites, comparison, and export. Authentication and authorization for private runs and admin case authoring.
- **Scientific runtime:** Python OpenEnv-compatible environment service with typed actions/observations, independent verifiers, and a tool gateway to ImmunoGraph MCP and source adapters.
- **Training/evaluation:** isolated GPU-capable trainer workers; separate evaluation workers with frozen policies and sealed held-out cases.
- **Persistence:** PostgreSQL for metadata and event indices; S3-compatible immutable artifact storage; a durable queue/worker mechanism for long-running jobs. Redis may support queue/cache but must not be the sole durable record.
- **Observability:** structured logs, traces, queue/job health, cost and tool latency, run IDs across services, and alerts for missing provenance or grader errors.

These choices need a small integration spike when implementation begins. A deployed architecture must document actual versions, interfaces, migrations, and rollback.

### 8.2 Reliability and isolation

Long-running tool calls and training jobs execute outside request handlers. Workers have bounded CPU/GPU, memory, disk, wall time, network allowlist, and per-service rate/cost quotas. Scientific tool calls use timeout, classified retry policy, circuit breaking, and idempotency where supported. Non-idempotent calls are not blindly retried. Store immutable attempt inputs before execution; checkpoint progress. A worker crash produces an explicit interrupted/failed state and can resume only from a defined safe boundary. Replayed attempts are read-only.

Secrets live in a secret manager or local development environment, not in case manifests, logs, graphs, screenshots, or ZIPs. Tool responses are untrusted; strip prompt-injection attempts from privileged instructions while retaining the original response as evidence. Validate file uploads and block arbitrary execution from agent-authored code except in a separately designed sandbox.

### 8.3 Data model, versioning and API boundaries

Minimum persisted entities: User, Project, RoomVersion, SessionVersion, EpisodeVersion, CaseVersion, SourceSnapshot, ToolInvocation, Attempt, Event, Artifact, VerifierFinding, RewardEvent, PolicyArtifact, TrainingJob, EvaluationSuiteVersion, EvaluationRun, Comparison, and ExportManifest. IDs are immutable; mutable display names do not affect history. Every result references exact case, suite, tool schema, scorer, source, environment image, policy and artifact hashes. Schema migrations preserve old runs and provide explicit compatibility adapters.

The UI reads projections of events; it does not calculate authoritative scientific scores. Only the verifier service can finalize grading. Training workers cannot write held-out result records; evaluation workers cannot modify policy artifacts. Source adapters cannot silently overwrite a captured snapshot.

## 9. Run ZIP and reproducibility

The ZIP is a first-class output, not a screenshot bundle. **Its minimum guarantee is complete offline inspection and deterministic replay of the recorded event stream and legally distributable captured artifacts.** “Replay” means reconstructing what happened from the saved log without re-calling a model or external source. A fresh computational rerun of the policy is a separate capability: it requires a retrievable, hash-pinned policy/base model, executable environment image, compatible tool versions, source snapshots or documented live dependencies, and available compute. The manifest explicitly marks the run as (a) offline replay verified and (b) computationally rerunnable, conditionally rerunnable, or not rerunnable, with reasons. Never call an audit ZIP a bit-for-bit reproducible policy rerun when these dependencies are absent.

The ZIP includes:

1. A manifest with schema version, file checksums, creation time, run/comparison IDs, license and redaction notes.
2. Episode/case/suite definitions that are permitted to export, seed, budgets, environment/container and code versions.
3. Policy metadata, tokenizer/chat-template and decoding settings, baseline/updated hashes, training recipe, and a loadable project-owned adapter/checkpoint when its license and size permit. If base or checkpoint weights cannot be bundled, include an immutable digest, retrieval instructions, access/retention policy, and explicit rerun limitation.
4. Ordered action/observation events, tool request/response envelopes, error/retry events, execution DAG, rewards, and verifier findings.
5. Inputs, intermediate outputs, scientific artifacts, source snapshots where redistribution is permitted, and citations/retrieval metadata otherwise.
6. Per-case and aggregate evaluation reports with denominators, metric definitions, uncertainty method and limitations.
7. A replay/readme file with commands or documented steps for offline event replay, snapshot-based computational rerun when supported, and a separately labeled optional fresh live rerun.

Export validation recomputes hashes, checks required files, detects orphan DAG nodes, tests offline event replay, and scans for credentials and prohibited private data. A missing required event or permitted captured artifact fails the export. A legally unredistributable source or unavailable external model is recorded as a computational-rerun limitation; the corresponding evidence reference and hash remain in the audit package. Live reruns may differ and must not overwrite the original record.

## 10. UX, accessibility and demo integrity

The selected visual reference is [docs/design/top-down-flow](design/top-down-flow/README.md). The game world is a navigation and replay metaphor; the scientific details use legible panels, tables, graphs, and provenance links. Animate the robot from event timestamps with pause, scrub, speed, step and direct-event jump. Preserve keyboard navigation, visible focus, reduced motion, readable text, and text/icon source-status labels. The moving-floor effect must not make the scientific record difficult to inspect.

Smallest.ai may narrate episode replay using an appropriately licensed/current TTS model. Voice is presentation only, does not change an episode score, and cannot claim an outcome that differs from the trace. Sarvam AI may power a baseline agent. Devin may assist implementation. Vendor availability, model names, pricing and API behavior require verification at build time.

Demo mode can use authored fixtures for a stable walkthrough, but must label them FIXTURE and retain a separate LIVE run when demonstrated. Generated visual art may depict a lab; sequences, structures, source labels, scoreboards and molecular surfaces presented as run evidence must resolve to real recorded artifacts.

## 11. Quality, security and scientific review

### 11.1 Verification pyramid

- **Contract checks:** schema compatibility for reset/step/state, action rejection, source envelopes, event ordering, manifest.
- **Episode checks:** expert-authored strong/weak trajectories, alternate valid paths, negative cases, fault recovery, reward-gaming attempts.
- **Integration checks:** pinned ImmunoGraph tool calls, configured live source calls, upstream failure and cache labeling.
- **Training checks:** nonzero optimizer steps, changed adapter/weights, checkpoint load, deterministic evaluation harness.
- **Evaluation checks:** split integrity, paired conditions, scorer independence, no held-out leakage, recomputable metrics.
- **UI checks:** event-to-robot/graph/timeline synchronization, door logic with failed attempts, keyboard/reduced motion, no fabricated score.
- **Export checks:** ZIP integrity, replay load, legal redistribution, secret scan.

Every scientific case and rubric needs at least one domain reviewer before being used for a performance claim. Document reviewer role and decision. Expert review is a quality gate for task validity, not a claim that the model's scientific findings are experimentally validated.

### 11.2 Threats and controls

Risks include prompt injection in upstream text, benchmark leakage, reward hacking, synthetic/live confusion, unsupported scientific claims, expensive tool loops, malicious uploads, credential exposure, and drift in external methods. Controls include sealed evaluation, hard budgets, provenance envelopes, independent grading, typed tool permissions, isolation, immutable snapshots, export scanning, and explicit limitations. Retain a red-team episode set for these failure modes.

## 12. Work breakdown and milestones

Work begins only after the owner explicitly authorizes implementation. Within that authorization, build in dependency order:

| Milestone | Deliverables | Exit condition |
| --- | --- | --- |
| M0 — Decisions and source audit | Model/compute choice; provider credentials and terms; case-review owner; schemas; reproducibility policy | Versioned architecture decision records and source licenses |
| M1 — Runtime spine | Database/artifact store; event log; run orchestration; OpenEnv reset/step/state; tool gateway; export skeleton | One replayable, independently graded episode with full trace |
| M2 — Three initial rooms | Reviewed sequence, structure and epitope cases; all applicable modes; fault injection; provenance | Each room passes case, rubric and integration gates |
| M3 — LLM post-training | Hosted Sarvam agent, trajectory corpus, curation, trainable Sarvam 30B compatibility pilot, warm-start if needed, GRPO/PPO update, checkpoints and registry | Proven **LLM** weight/adapter update and frozen held-out evaluation |
| M4 — TransferArena and Evaluation Hall | Sealed transfer suite, paired comparison, metrics, evidence wall, and live-failure-to-new-case protocol | No leakage; honest labeled result with denominator and source status; new case version linked to captured failure |
| M5 — Pixel frontend | Outer map, interiors, robot, synced timeline/graph, 3D viewer, accessibility, ZIP terminal | UI renders recorded events and export passes integrity checks |
| M6 — Release hardening | Security, failure recovery, observability, documentation, scientific review, end-to-end rehearsal | All completion gates in §2.1 evidenced |
| M7 — Expansion | Seven later scientific rooms, each with its own source/verifier/review plan | Every additional room clears the same gates as initial rooms |

Milestones can overlap where dependencies permit, but a polished UI cannot substitute for M1–M4 evidence. Maintain a task-to-requirement trace table during implementation; each PR should cite the relevant PRD section and its acceptance evidence.

## 13. Open decisions

These require concrete choices at implementation start, not assumptions disguised as facts:

1. Trainable open-weight model, license, GPU/VRAM budget, training time, and hosted baseline terms.
2. Initial scientific case corpus, biological-family splits, expert reviewer(s), and source redistribution rights.
3. Exact ImmunoGraph commit and tool set; configured local runtimes; live source quotas/credentials.
4. Hosting, worker isolation, artifact retention, privacy and authentication requirements.
5. Numeric launch thresholds for task success, integrity regressions, latency and cost after a pilot baseline establishes feasible ranges.
6. Which ResearchGym tasks are valid in each initial room; do not populate a mode with filler episodes just to fill a grid.

## 14. References and source-of-truth links

- [OpenEnv overview](https://huggingface.co/docs/openenv/index), [core environment API](https://huggingface.co/docs/openenv/main/reference/core), and [TRL integration for interactive RL](https://huggingface.co/docs/trl/openenv).
- [ImmunoGraph repository](https://github.com/Ajey95/immuno-graph), [tool catalog](https://github.com/Ajey95/immuno-graph/blob/main/docs/TOOL_CATALOG.md), [tool reality matrix](https://github.com/Ajey95/immuno-graph/blob/main/docs/TOOL_REALITY_MATRIX.md), and [live workflow guide](https://github.com/Ajey95/immuno-graph/blob/main/docs/LIVE_WORKFLOWS.md).
- [BioAgent Bench (ICML 2026)](https://proceedings.mlr.press/v306/fa26a.html), [BixBench](https://arxiv.org/abs/2503.00096), [BixBench3 preprint](https://arxiv.org/abs/2608.25286), and [BiomniBench preprint](https://www.biorxiv.org/content/10.64898/2026.05.12.724604v2) are related research, not BioGym Arena validation.
- [Sarvam API documentation](https://docs.sarvam.ai/api/token-factory/chat-completions) and [Smallest.ai speech documentation](https://docs.smallest.ai/api-reference/models/text-to-speech/synthesize-speech). Verify current model availability before implementation.
- [Sarvam open-weight announcement](https://www.sarvam.ai/blogs/sarvam-30b-105b) and [Sarvam 30B model card](https://huggingface.co/sarvamai/sarvam-30b). Open weights permit an independently trained adapter, subject to model compatibility and compute; a hosted API call does not update Sarvam's served model.

**Document maintenance:** Record changes as versioned PRD revisions with date, reason, affected requirements and migration impact. Keep the README's architecture/results placeholders empty until implementation and measured results exist. Do not turn an aspiration in this document into a claimed result.
