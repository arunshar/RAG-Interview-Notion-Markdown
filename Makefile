.PHONY: check import

check:
	python3 scripts/validate_repository.py
	python3 scripts/build_import.py --check

import:
	python3 scripts/build_import.py
