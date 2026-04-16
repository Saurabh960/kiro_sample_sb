"""Shared fixtures for PDF Duplicate Detector tests."""

import os
import tempfile

import pytest


@pytest.fixture
def tmp_dir():
    """Provide a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_pdf_path(tmp_dir):
    """Provide a path for a sample PDF in the temp directory."""
    return os.path.join(tmp_dir, "sample.pdf")
