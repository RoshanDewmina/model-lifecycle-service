# Interview guide

## Explain the system in two minutes

The service demonstrates the smallest credible ML lifecycle: deterministic data identity and splitting, a dummy baseline, a trained preprocessing-and-classification pipeline, an immutable artifact with hashes and metadata, a release gate, rollback, and a prediction API that validates the same features used at training time. The UI makes the active model and evidence inspectable.

## Decisions to defend

- **Why Wine?** It is public, non-sensitive, CPU-small, clearly licensed, and sufficient to demonstrate lifecycle integrity. It is deliberately not framed as a business-impact model.
- **Why balanced accuracy and per-class recall?** Accuracy alone can hide a weak class. The gate requires a 0.10 validation balanced-accuracy gain over a most-frequent dummy and at least 0.60 validation recall for every class. Held-out scores are reported only after the gate decision is fixed.
- **How is leakage limited?** Stable sample hashes define disjoint train, validation, and held-out identities. Preprocessing lives inside the pipeline and fits only on training rows. No hyperparameter search touches held-out data.
- **Why skops?** Loading pickle/joblib can execute arbitrary code. The loader checks the artifact and manifest hashes and refuses unknown skops types before loading an artifact created by this project.
- **Why CLI-only promotion?** Promotion and rollback are operational controls. Keeping them off anonymous HTTP removes an unnecessary administration attack surface.
- **What does rollback mean?** A target must have been promoted before. Its saved artifact, metadata, and manifest identity plus schema are verified before an atomic active-pointer change.

## Failure stories worth showing

1. Train a dummy candidate and attempt promotion. The command exits 2 and leaves the active version unchanged.
2. Alter one byte of a model artifact. Loading fails on the SHA-256 mismatch.
3. Omit a feature, send a numeric string, or exceed a physical bound. `/predict` returns 422.
4. Promote a second valid candidate, then roll back to the known first version and observe `/health` report it.

## Hands-on exercises

- Change the split seed and explain why results become a different experiment rather than a replacement for the recorded receipt.
- Add a fourth candidate algorithm without changing the serving contract.
- Replace the active-pointer JSON with a transactional registry and explain which concurrent-writer problem that solves.
- Add signed manifests or an external artifact store, then update the threat model.
- Design real drift collection with consent, retention, alert ownership, and delayed-label monitoring. Do not reuse the simulated diagnostic as real monitoring.

## Limits to say plainly

This is a single-host, local demonstration over 178 old rows. Mutations serialize through a fail-fast file lock and atomically replace the registry, but it has no distributed coordination, signature, remote storage, or staged deployment. Test scores and in-process latency do not predict production behavior. Personal mastery and contribution claims remain pending until the owner can reproduce and explain the system.
