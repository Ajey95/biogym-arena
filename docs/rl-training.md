# BioGym RL training protocol

**Current suite:** `biogym-rl-v2`. The [November 20 pilot](../results/README.md)
used `biogym-rl-v1`; its 1/32 to 4/32 comparison is historical and cannot be
carried forward as a version 2 result.

This protocol is the first LLM post-training lane. It is separate from the
small neural action-selector smoke test. A valid result needs a changed LLM
adapter, optimizer steps, and a separate matched held-out comparison.

## Source and case inventory

`scripts/build_rl_snapshots.py` captures minimal UniProt and RCSB metadata into
`backend/biogym/data/rl_snapshots_v1.json`. Each record has its upstream URL,
capture time, raw response SHA-256, normalized fields, and `CACHED` status.
The repository does not bundle full UniProt/RCSB responses or present these
snapshots as newly collected experimental evidence. Rebuilding the snapshot
requires a new suite hash and fresh baseline/evaluation runs.

The current suite has 24 source records, assigned by biological entity or PDB
record to 16 train, 4 test, and 4 transfer sources. Each source yields sequence
identity/validity, peptide window occurrence, unsupported binding-claim audit,
or structure-method cases. Paired UniProt/RCSB records add independent
two-record audits that require both sequence and structure tools. This produces
**80 train, 20 test, and 20 transfer episodes**. Transfer has two joint cases.
The model sees the public task and tool results; expected claims, exact scope
labels and per-task answer options stay out of the observation. All attempts,
including no-action and failed attempts, belong in the evaluation denominator.
The source set is still small and authored; record-level splits do not prove
biological-family generalization or expert-reviewed case validity.

## Verifier and reward

The policy must call task tools, cite successful evidence IDs, report the
`CACHED` status returned by the external-record tool, and infer a scope label.
Sequence outcomes use the captured sequence window and FASTA validity. Peptide
outcomes use exact occurrence in that **32-residue window**; there is no authored
predictor score or claim that absence from the window proves absence from the
full protein. A separate claim audit requires abstention from an unsupported
experimental-binding assertion. Structure outcomes use the RCSB experimental
method. Joint audits require independent UniProt, FASTA and RCSB evidence;
they do not assert that the two records are biologically linked. A transient
failure requires a retry. Wrong identifiers, missing evidence, wrong source
status and unsupported claims fail verification. Repeated successful tool calls
do not earn more process reward. Step events form a hash chain and record
provenance. LabOps, ResearchSense and verified outcome remain separate. Scores
do not validate biological efficacy.

## Colab pilot

The currently accessible Colab account offers a T4 GPU. Sarvam 30B is outside
this pilot's verified memory budget. The pilot uses `Qwen/Qwen3-0.6B` to
validate the *LLM* training path. It is never labelled a Sarvam model.

The version 2 recipe first generates **train-only, verifier-checked tool-call
demonstrations** and uses TRL SFT on each agent decision, with prior tool
results in the prompt and loss on the next tool call. It then uses that adapter
as the before-RL policy and applies TRL GRPO through `environment_factory`.
This separates supervised tool-use
warm start from reward-driven RL updates. The SFT manifest records the exact
80 train case IDs, tool schema hash, demonstration hash, optimizer steps,
adapter hashes, and compute details. `rl_train` checks these before loading the
warm adapter and copies the SFT manifest and demonstrations into the run ZIP.
The demonstrations are generated from the current verifier and are **not**
expert-reviewed biological truth. They must not include held-out or transfer
answers.

From a GPU Linux runtime after cloning the repository:

```bash
pip install -e 'backend[train]'
python -m biogym.rl_sft --output /content/biogym-v2-sft --epochs 1
python -m biogym.rl_train --output /content/biogym-v2-run \
  --warm-adapter /content/biogym-v2-sft/adapter --max-steps 32
```

The GRPO runner saves initial and final adapters, their file digests, trainable
parameter count, optimizer-step count, GPU/library metadata, and training
traces. It does **not** evaluate held-out cases inside the training process.
The test and transfer splits are never used for SFT or GRPO gradient updates.
The version 2 notebook uses two generations per prompt and 32 GRPO steps as a
compute pilot;
further scale requires measured reward variance and compute cost. A run report
may say "after RL training" only when the adapter tensor hash changed,
optimizer steps are positive, the checkpoint reloads, and separate held-out
results are present. Repeated evaluations over one case do not increase the
count of independent cases.

For the final comparison, evaluate both frozen adapters in separate fresh GPU
processes. This removes optimizer state from the post-training memory budget
and checks the same 20 test cases with 40 attempts per adapter:

```bash
python -m biogym.rl_eval --run-dir /content/biogym-v2-run --adapter pre \
  --split test --output /content/biogym-v2-test-pre
python -m biogym.rl_eval --run-dir /content/biogym-v2-run --adapter post \
  --split test --output /content/biogym-v2-test-post
python -m biogym.rl_eval --run-dir /content/biogym-v2-run --adapter pre \
  --split transfer --output /content/biogym-v2-transfer-pre
python -m biogym.rl_eval --run-dir /content/biogym-v2-run --adapter post \
  --split transfer --output /content/biogym-v2-transfer-post
```

Do not reuse a run or evaluation output directory after a failed attempt;
partial traces would corrupt the denominator. Review GPU allocation warnings
and generation errors before using an evaluation as evidence.

After both evaluations finish, `python -m biogym.rl_compare` validates the
adapter lineage, matched case groups, the declared attempt budget, and clean
evaluation logs. Run it separately for test and transfer. It writes a comparison
JSON and bundles the checkpoints, training/evaluation traces, logs, source
snapshot, training code and notebook into one ZIP. The case-clustered bootstrap
interval is descriptive for this small authored suite. The packager checks each
reported attempt against its saved trace and verifies the event hash chain;
missing or divergent traces prevent a comparison artifact.

The 21 November 2026 v2 run completed this protocol. See the
[measured test and TransferArena comparisons](../results/README.md), including
the uncertainty intervals and downloadable full-run bundles. The result
establishes a working update and evaluation path, not a reliable scientific
agent.

## Current gates before a broad claim

- Expand independently reviewed source entities, especially epitope evidence
  and biologically meaningful transfer cases.
- Get domain review of the version 2 case cards and scope labels before making
  scientific-performance claims. Automated reachability tests are not expert review.
- Add adversarial source content, method alternatives, cross-source identity
  checks, and expert-reviewed case cards.
- Review qualitative traces and reward exploitation, then test policy behavior
  on fresh live ImmunoGraph tasks.
- For Sarvam 30B, obtain suitable GPU memory and run a model compatibility
  pilot before enabling its training recipe.
