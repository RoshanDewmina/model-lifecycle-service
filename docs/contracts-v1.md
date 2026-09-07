# Applicable portfolio contract

- `GET /health` returns service status and active model version; missing or invalid artifacts return 503.
- `POST /predict` accepts exactly 13 named, bounded numeric features and returns class, probabilities, and `model_version`.
- Training and serving use the same ordered feature contract.
- Release and rollback are CLI controls. There is no HTTP administration surface.
- Dataset and evaluation receipts record license, source, hashes, split identities, seed, revision, environment, results, and limitations.
- Held-out identities are excluded from fitting and tuning. The test partition is evaluated once in each immutable candidate run.
- Candidates must beat the dummy baseline on tuning validation by the configured balanced-accuracy margin and meet minimum validation per-class recall. Held-out results are informational and cannot change promotion eligibility.
- Registry mutations use a fail-fast exclusive process lock and durable atomic replacement. Approved versions bind the artifact, metadata, and dataset-manifest digests used by active serving and rollback.
- Local evidence is not production-performance evidence. Simulated drift is labeled as simulation.

The portfolio-wide authority is `../docs/contracts-v1.md` outside this independent repository.
