# Applicable portfolio contract

- `GET /health` returns service status and active model version; missing or invalid artifacts return 503.
- `POST /predict` accepts exactly 13 named, bounded numeric features and returns class, probabilities, and `model_version`.
- Training and serving use the same ordered feature contract.
- Release and rollback are CLI controls. There is no HTTP administration surface.
- Dataset and evaluation receipts record license, source, hashes, split identities, seed, revision, environment, results, and limitations.
- Held-out identities are excluded from fitting and tuning. The test partition is evaluated once in each immutable candidate run.
- Candidates must beat the dummy baseline by the configured balanced-accuracy margin and meet minimum per-class recall.
- Local evidence is not production-performance evidence. Simulated drift is labeled as simulation.

The portfolio-wide authority is `../docs/contracts-v1.md` outside this independent repository.

