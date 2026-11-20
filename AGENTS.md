# BioGym Arena working instructions

- Read [docs/PRD.md](docs/PRD.md) before making product or architecture decisions. It is the build reference; read [docs/design/top-down-flow/README.md](docs/design/top-down-flow/README.md) for the selected frontend direction.
- The owner has explicitly said not to begin product implementation until they say to start. Documentation and design work do not waive this instruction.
- When implementation is authorized, make changes traceable to PRD requirements and update the PRD when a decision changes scope, scientific behavior, evaluation, or data provenance.
- Do not claim training from prompt or configuration changes. A training claim needs an actual changed weight or adapter artifact, recorded optimizer steps, and separate held-out evaluation.
- Preserve source status and scientific limits: LIVE, CACHED, FIXTURE, and SYNTHETIC are distinct. Computational predictions are not experimental or clinical validation.
- Keep run events and failures inspectable. UI animation and screenshots must follow recorded evidence and must not invent scores, biological data, or successful outcomes.
