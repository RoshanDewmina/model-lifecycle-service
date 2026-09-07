# Engineering sources checked

Current library guidance was checked through Context7 against the official scikit-learn and FastAPI documentation during implementation.

- scikit-learn model persistence: <https://scikit-learn.org/stable/model_persistence.html>
- scikit-learn common pitfalls and pipelines: <https://scikit-learn.org/stable/common_pitfalls.html>
- FastAPI response models: <https://fastapi.tiangolo.com/tutorial/response-model/>
- FastAPI static files: <https://fastapi.tiangolo.com/tutorial/static-files/>

The implementation follows the documented guidance to keep preprocessing in a pipeline, avoid fitting transformations on held-out data, use typed request/response models, and prefer `skops` with inspected trusted types over pickle-style arbitrary object loading.

