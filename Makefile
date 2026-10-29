.PHONY: regression-quick regression-quick-offline regression-full-offline

# Real medical fixtures and their curated selectors remain outside Git and CI.
regression-quick:
	RUN_APPROVED_DOCS=1 uv run --frozen pytest tests/regression/test_approved_docs.py -m approved_docs --golden-suite=quick -q

regression-quick-offline:
	RUN_LOCAL_GOLDENS=1 uv run --frozen pytest tests/regression/test_approved_docs.py -m golden_local --golden-suite=quick -q

regression-full-offline:
	RUN_LOCAL_GOLDENS=1 uv run --frozen pytest tests/regression/test_approved_docs.py -m golden_local -q

release-%:
	hatch version $*
	git add parselabs/__init__.py
	git commit -m "chore: release $$(hatch version)"
	git push
