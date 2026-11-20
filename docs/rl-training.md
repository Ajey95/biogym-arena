# BioGym RL training protocol

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
record to 16 train, 4 test, and 4 transfer sources. Each source yields multiple
task variants: sequence validation and identity, computational peptide scope,
structure-method provenance, and transient failure recovery. This produces 64
train, 16 test, and 16 transfer episodes. The training model sees the public
observation and tool results; expected claims stay in the verifier. All
attempts, including no-action and failed attempts, belong in the evaluation
denominator. The case count is still small and authored; record-level splits
do not prove broad biological generalization.

## Verifier and reward

The policy must call task tools, cite successful evidence IDs, state `CACHED`
for the external record, and use the required scope limitation. Sequence
outcomes are checked against the captured sequence window and FASTA validity;
peptide outcomes are bounded computational interpretations; structure outcomes
are checked against the RCSB experimental-method field. A transient failure
requires a retry. Wrong identifiers, missing evidence, wrong source status,
and unsupported claims fail verification. Step events form a hash chain and
record provenance. The scalar reward gives verified outcome most weight, then
process and evidence quality. Scores do not validate biological efficacy.

## Colab pilot

The currently accessible Colab account offers a T4 GPU. It does not offer an
enabled A100, H100, or L4. Sarvam 30B is therefore outside this pilot's
verified memory budget. The pilot uses `Qwen/Qwen3-0.6B` to validate the *LLM*
RL path, with a LoRA adapter and TRL GRPO `environment_factory`. It is never
labelled a Sarvam model. Sarvam-1 is a distinct, older base completion model
with a non-commercial license and requires a separate instruction warm-start;
it is not a drop-in substitute for Sarvam 30B.

From a GPU Linux runtime after cloning the repository:

```bash
pip install -e 'backend[train]'
python -m biogym.rl_train --output /content/biogym-run --max-steps 16
```

The runner saves initial and final adapters, their file digests, trainable
parameter count, optimizer-step count, GPU/library metadata, training traces,
and before/after held-out evaluation. Its test split is never used for gradient
updates. The pilot recipe uses two generations per prompt, short budgets, and
16 steps to establish the training path; further scale requires measured
reward variance and compute cost. A run report may say "after RL training"
only when the adapter tensor hash changed, optimizer steps are positive, the
checkpoint reloads, and held-out results are present. Repeated evaluations
over the same case do not increase the count of independent cases.

For the final comparison, evaluate both frozen adapters in separate fresh GPU
processes. This removes optimizer state from the post-training memory budget
and checks the exact same 16 cases and 32 attempts per adapter:

```bash
python -m biogym.rl_eval --run-dir /content/biogym-run --adapter pre \
  --output /content/biogym-pre-eval
python -m biogym.rl_eval --run-dir /content/biogym-run --adapter post \
  --output /content/biogym-post-eval
```

Do not reuse a run or evaluation output directory after a failed attempt;
partial traces would corrupt the denominator. Review GPU allocation warnings
and generation errors before using an evaluation as evidence.

## Current gates before a broad claim

- Expand independently reviewed source entities, especially epitope evidence
  and biologically meaningful transfer cases.
- Add adversarial source content, method alternatives, cross-source identity
  checks, and expert-reviewed case cards.
- Verify held-out checkpoint loading, qualitative traces, reward exploitation,
  and policy behavior on fresh live ImmunoGraph tasks.
- For Sarvam 30B, obtain suitable GPU memory and run a model compatibility
  pilot before enabling its training recipe.
