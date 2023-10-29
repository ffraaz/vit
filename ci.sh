#!/usr/bin/env bash

set -eux

black --check .
pylint vit.py test_vit.py
mypy vit.py test_vit.py
pytest
