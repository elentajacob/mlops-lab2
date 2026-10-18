import os
import sys

import pytest

# Make the scripts in src/ importable the same way they import each other
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# Keep tests from writing their own tables into the real GitHub Actions run summary
@pytest.fixture(autouse=True)
def no_step_summary(monkeypatch):
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
