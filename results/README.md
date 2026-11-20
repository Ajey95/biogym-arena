# BioGym LLM RL pilots: measured results

## Version 2: supervised warm start followed by GRPO (21 November 2026)

This run used **Qwen/Qwen3-0.6B** on a Colab Tesla T4 with TRL 0.29.1. The
versioned suite contains 80 train, 20 test, and 20 TransferArena cases over 24
cached UniProt/RCSB source records. A train-only, verifier-checked supervised
warm start produced 248 tool decisions from 80 cases and completed 62 LoRA
optimizer steps. GRPO then completed 32 further optimizer steps from that
adapter, saved distinct pre/post adapter tensors, and recorded 64 training
rollouts, including 6 verified successes. The before policy below is the
**supervised warm-start adapter**, not the untouched base model.

| Split | Before RL | After RL | Observed change | Paired case bootstrap 95% interval |
| --- | ---: | ---: | ---: | ---: |
| Held-out test | 5/40 | 9/40 | +10 percentage points | −7.5 to +27.5 points |
| TransferArena | 3/40 | 6/40 | +7.5 percentage points | 0 to +15 points |

Each split has 20 cases and two attempts per policy. Test had five improved
cases and three regressed cases; TransferArena had three improved and zero
regressed cases. Each frozen adapter was evaluated in its own fresh GPU
process. The [test comparison](biogym-rl-v2-test.json) and [transfer comparison](biogym-rl-v2-transfer.json)
contain paired case outcomes and artifact hashes. The complete bundles are
`D:\c-downloads\biogym-v2-test-complete-7614218.zip` (77,202,877 bytes,
SHA-256 `34ec7f9b2ca0dafdda396dd1c4b10033cd98d5749244e3cb80d9c67b2d7b69af`)
and `D:\c-downloads\biogym-v2-transfer-complete-7614218.zip` (77,202,668
bytes, SHA-256 `75b0ad0629035dc386ed69b8f189f225f12cf7a9fbcab182d30851bcc2df396b`).
Both local hashes matched Colab, and all 234 ZIP entries in each bundle passed
an archive integrity check. The bundles include the source snapshot, tool
traces, evaluation logs, manifests, code, and model adapters. The training and
evaluation code commit is `76142182a179df3d3950b1b841af4ba31b3017a6`.

The failure audit still shows weak scientific decisions. Among post-RL attempts,
24/40 test and 27/40 transfer attempts submitted an incorrect claim; 16/40
test and 22/40 transfer attempts had incorrect provenance or scope labels.
These categories overlap. Tool execution alone therefore cannot be treated as
scientific task success.

This is evidence of real LLM adapter updates and a measured gain in this small
authored suite. Both intervals include zero, so it does not establish reliable
generalization or broad biological validity. The records are cached database
metadata, not experimental validation; the cases and rubrics have not received
independent scientist review. The adapter is not deployed as the app's serving
agent, and it is not a Sarvam model. The [training protocol](../docs/rl-training.md)
records the remaining scientific and deployment gates.

## Version 1: initial GRPO pilot (20 November 2026)

The 2026-11-20 Colab T4 pilot trained a **Qwen/Qwen3-0.6B** LoRA adapter with
TRL GRPO over BioGym's stateful tool episodes. This is a genuine LLM adapter
update, not Sarvam 30B training. The adapter is not yet deployed as the app's
serving agent.

| Evidence | Value |
| --- | --- |
| Training code commit | `ae8ae91a3f6039ad8448e9c6d68b5ad12dd60df9` |
| Isolated evaluation code commit | `fb9e0d838740cead80448cbb1e2b543bc7516c05` |
| Source snapshot | 24 cached UniProt/RCSB records; 64 train, 16 held-out test, 16 transfer episodes |
| Optimizer steps / trainable adapter parameters | 16 / 2,293,760 |
| Before adapter SHA-256 | `97fa97b3e85a9240204d350aa3b99f6d1a4361715e951e964078f78cd7d9e474` |
| After adapter SHA-256 | `2de3eb03a96175eae533305c95c31bc490665cf75e172902d820bb986f0342d2` |
| Frozen held-out comparison | 1/32 verified successes before; 4/32 after, on the same 16 cases with two attempts each |
| Observed difference | +9.375 percentage points; improved on two cases and regressed on zero |
| Paired case bootstrap 95% interval | 0 to +21.875 percentage points |

The [machine-readable comparison](biogym-rl-t4-pilot.json) records each case's
success, LabOps, ResearchSense and reward before and after. Both frozen
adapters were loaded and evaluated in separate GPU processes under the same
seed, suite, tools, and attempt budget. The comparison packager rejected
incomplete case sets, mismatched adapter hashes and memory or execution errors
in either evaluation log. The complete 70,932,010-byte ZIP had SHA-256
`23226989b035b04223d7113964dd9de6f2c87a3d09d2fc248bf201662e5f91d6`.
The ZIP contains checkpoints, train and evaluation traces, the source
snapshot, code, logs and manifests. It was downloaded to
`D:\c-downloads\biogym-rl-complete.zip` on this machine; its local SHA-256
matched the Colab digest, and all 231 ZIP entries passed an archive integrity
check. The [Colab training notebook](https://colab.research.google.com/drive/1P__T33x1GAxTuQd-l5vbH6PrU9jaYi9E)
also has the ZIP at `/content/biogym-rl-complete.zip` while its runtime is
alive. The repository contains the comparison JSON but does not commit the
70 MB binary archive.

This is a small authored metadata suite. UniProt/RCSB snapshots are not new
experimental observations, and the peptide scores are authored computational
fixtures. The interval includes zero, so the result does not establish broad
agent improvement or biological validity. The 16 transfer episodes were not
run in this pilot. Sarvam 30B needs a separately validated larger GPU and
model-specific training path. See [the protocol](../docs/rl-training.md) for
the remaining gates.
