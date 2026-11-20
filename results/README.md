# Qwen LLM RL pilot: measured result

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
