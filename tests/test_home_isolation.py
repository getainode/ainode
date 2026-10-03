"""The unit suite cannot read or write the operator's AINode home."""

import os
from pathlib import Path
import shutil

from ainode.auth import accounts, middleware
from ainode.core import config
from ainode.datasets import manager as datasets
from ainode.models import api_routes as models
from ainode.secrets import manager as secrets
from ainode.training import engine as training


def test_every_import_time_default_uses_the_session_home(isolate_ainode_home):
    home = Path(os.environ["AINODE_HOME"])

    assert home == isolate_ainode_home
    assert home != (Path.home() / ".ainode").resolve()
    assert config.AINODE_HOME == home
    assert config.CONFIG_FILE == home / "config.json"
    assert config.MODELS_DIR == home / "models"
    assert config.DATASETS_DIR == home / "datasets"
    assert config.TRAINING_DIR == home / "training"
    assert middleware.AUTH_FILE == home / "auth.json"
    assert accounts.USERS_FILE == home / "users.json"
    assert secrets.SECRETS_FILE == home / "secrets.json"
    assert datasets.DATASETS_DIR == home / "datasets"
    assert datasets.REGISTRY_FILE == home / "datasets" / "_registry.json"
    assert training.JOBS_DIR == home / "training" / "jobs"
    assert models._manifest_path() == home / "instances.json"

    defaults = config.NodeConfig()
    assert Path(defaults.models_dir) == home / "models"
    assert defaults.datasets_dir is None
    assert defaults.training_dir is None


def test_representative_default_writes_stay_in_the_session_home(isolate_ainode_home):
    config.NodeConfig(model="test/model").save()
    job = training.TrainingJob(
        training.TrainingConfig(base_model="test/model", dataset_path="user/data")
    )

    assert config.CONFIG_FILE.is_file()
    assert job._job_dir.is_relative_to(isolate_ainode_home)

    config.CONFIG_FILE.unlink()
    shutil.rmtree(isolate_ainode_home / "training")
