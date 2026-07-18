

import pytest
from omegaconf import OmegaConf

@pytest.fixture
def sample_config():
    """Provide a sample configuration for testing."""
    return OmegaConf.create({
        "app": {
            "version": "0.1.0",
            "environment": "test"
        },
        "logging": {
            "level": "DEBUG",
            "format": "%(message)s"
        }
    })

@pytest.fixture
def temp_output_dir(tmp_path):
    """Provide a temporary output directory."""
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    return output_dir
