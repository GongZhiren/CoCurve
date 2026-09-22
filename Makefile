.PHONY: install test verify verify-paper-masks privacy

install:
	pip install -e '.[all]'

verify:
	cocurve-artifacts verify

verify-paper-masks:
	python scripts/verify_all_masks.py

privacy:
	python scripts/check_self_contained.py --project-root .

test: verify privacy
	pytest -q
