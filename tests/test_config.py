"""Configuration loading tests."""

import copy
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from ats_ceramic.config import (
    ENV_CONFIG_DIR,
    ConfigError,
    ScenarioId,
    SyntheticConfig,
    ToleranceConfig,
    config_fingerprint,
    default_config_dir,
    load_channel_config,
    load_synthetic_config,
    load_tolerance_config,
)
from ats_ceramic.schemas import DataOrigin


def _raw(name: str) -> dict:
    """Read a real config file as a plain dict (for building modified variants)."""
    return yaml.safe_load((default_config_dir() / name).read_text(encoding="utf-8"))


def _write(tmp_path: Path, name: str, data: dict) -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def _tolerance_data_with_override(sku_id: str = "SKU_X", threshold: float = 0.5) -> dict:
    """Tolerance config data (mapping form) with one SKU override."""
    data = _raw("tolerances.yaml")
    override = dict(
        data["acceptance"]["default"], delta_e00_threshold=threshold, source="test override"
    )
    data["acceptance"]["sku_overrides"] = {sku_id: override}
    return data


# ---- Shipped configs ---------------------------------------------------------


def test_shipped_configs_load():
    tolerances = load_tolerance_config()
    channels = load_channel_config()
    synthetic = load_synthetic_config()
    assert tolerances.schema_version == 1
    assert len(channels.channels) >= 1
    assert synthetic.data_origin is DataOrigin.SYNTHETIC


def test_shipped_tolerances_are_flagged_as_placeholders():
    tolerances = load_tolerance_config()
    assert tolerances.has_placeholders
    assert tolerances.acceptance.default.is_placeholder
    assert tolerances.measurement_quality.is_placeholder


def test_shipped_channels_are_synthetic_placeholders():
    channels = load_channel_config()
    assert channels.data_origin is DataOrigin.SYNTHETIC
    assert channels.is_placeholder


def test_all_intended_scenarios_configured():
    synthetic = load_synthetic_config()
    assert set(synthetic.scenario_ids) == set(ScenarioId)
    assert len(ScenarioId) == 7
    for scenario_id in ScenarioId:
        assert synthetic.scenario(scenario_id).n_batches > 0


def test_fingerprint_is_stable_and_content_sensitive(tmp_path):
    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    a.write_bytes(b"x: 1\ny: 2\n")
    b.write_bytes(b"x: 1\r\ny: 2\r\n")  # same content, CRLF line endings
    assert config_fingerprint(a) == config_fingerprint(b)
    b.write_bytes(b"x: 1\ny: 3\n")
    assert config_fingerprint(a) != config_fingerprint(b)


# ---- SKU overrides: lookup, representation, immutability ---------------------


def test_for_sku_falls_back_to_default_and_honours_override():
    config = ToleranceConfig.model_validate(_tolerance_data_with_override())
    default_threshold = config.acceptance.default.delta_e00_threshold
    assert config.for_sku("SKU_X").delta_e00_threshold == 0.5
    assert config.for_sku("SKU_OTHER").delta_e00_threshold == default_threshold


def test_override_list_form_equals_mapping_form():
    mapping_data = _tolerance_data_with_override()
    list_data = copy.deepcopy(mapping_data)
    list_data["acceptance"]["sku_overrides"] = [
        {"sku_id": "SKU_X", "tolerance": mapping_data["acceptance"]["sku_overrides"]["SKU_X"]}
    ]
    assert ToleranceConfig.model_validate(mapping_data) == ToleranceConfig.model_validate(
        list_data
    )


def test_duplicate_sku_overrides_rejected():
    data = _tolerance_data_with_override()
    spec = data["acceptance"]["sku_overrides"]["SKU_X"]
    data["acceptance"]["sku_overrides"] = [
        {"sku_id": "SKU_X", "tolerance": spec},
        {"sku_id": "SKU_X", "tolerance": spec},
    ]
    with pytest.raises(ValidationError, match="duplicate sku_id"):
        ToleranceConfig.model_validate(data)


def test_sku_overrides_are_an_immutable_tuple():
    config = ToleranceConfig.model_validate(_tolerance_data_with_override())
    overrides = config.acceptance.sku_overrides
    assert isinstance(overrides, tuple)
    assert not hasattr(overrides, "append")
    with pytest.raises(TypeError):
        overrides[0] = overrides[0]
    with pytest.raises(ValidationError):
        config.acceptance.sku_overrides = ()
    with pytest.raises(ValidationError):
        overrides[0].tolerance.delta_e00_threshold = 9.0


def test_config_does_not_alias_source_data():
    data = _tolerance_data_with_override()
    config = ToleranceConfig.model_validate(data)
    default_before = config.acceptance.default.delta_e00_threshold
    data["acceptance"]["sku_overrides"]["SKU_X"]["delta_e00_threshold"] = 99.0
    data["acceptance"]["default"]["delta_e00_threshold"] = 99.0
    assert config.for_sku("SKU_X").delta_e00_threshold == 0.5
    assert config.acceptance.default.delta_e00_threshold == default_before


def test_has_placeholders_considers_overrides():
    data = _tolerance_data_with_override()
    data["acceptance"]["default"]["is_placeholder"] = False
    data["measurement_quality"]["is_placeholder"] = False
    data["acceptance"]["sku_overrides"]["SKU_X"]["is_placeholder"] = True
    assert ToleranceConfig.model_validate(data).has_placeholders
    data["acceptance"]["sku_overrides"]["SKU_X"]["is_placeholder"] = False
    assert not ToleranceConfig.model_validate(data).has_placeholders


# ---- Scenarios: representation, immutability ---------------------------------


def test_scenarios_are_an_immutable_tuple():
    synthetic = load_synthetic_config()
    scenarios = synthetic.scenarios
    assert isinstance(scenarios, tuple)
    assert not hasattr(scenarios, "append")
    with pytest.raises(TypeError):
        scenarios[0] = scenarios[0]
    with pytest.raises(ValidationError):
        synthetic.scenarios = ()
    with pytest.raises(ValidationError):
        scenarios[0].spec.n_batches = 1


def test_duplicate_scenarios_rejected():
    data = _raw("synthetic_scenarios.yaml")
    spec = data["scenarios"]["near_boundary"]
    data["scenarios"] = [{"scenario_id": "near_boundary", "spec": spec}] * 2
    with pytest.raises(ValidationError, match="duplicate scenario_id"):
        SyntheticConfig.model_validate(data)


def test_unknown_scenario_name_rejected(tmp_path):
    data = _raw("synthetic_scenarios.yaml")
    data["scenarios"]["near_bondary"] = data["scenarios"].pop("near_boundary")  # typo
    with pytest.raises(ConfigError, match="near_bondary"):
        load_synthetic_config(_write(tmp_path, "synthetic_scenarios.yaml", data))


# ---- Failure modes -----------------------------------------------------------


def test_missing_file_raises_config_error(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_tolerance_config(tmp_path / "does_not_exist.yaml")


def test_duplicate_yaml_keys_rejected(tmp_path):
    path = tmp_path / "channels.yaml"
    path.write_text("schema_version: 1\nschema_version: 1\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="duplicate key"):
        load_channel_config(path)


def test_unknown_key_rejected(tmp_path):
    data = _raw("tolerances.yaml")
    data["acceptance"]["default"]["delta_e00_treshold"] = 2.0  # typo
    with pytest.raises(ConfigError):
        load_tolerance_config(_write(tmp_path, "tolerances.yaml", data))


def test_channel_range_must_contain_zero(tmp_path):
    data = _raw("channels.yaml")
    data["channels"][0]["min_delta"] = 0.05
    with pytest.raises(ConfigError, match="min_delta"):
        load_channel_config(_write(tmp_path, "channels.yaml", data))


def test_one_sided_channel_range_is_allowed(tmp_path):
    data = _raw("channels.yaml")
    data["channels"][0]["min_delta"] = 0.0  # channel that can only be increased
    config = load_channel_config(_write(tmp_path, "channels.yaml", data))
    assert config.channels[0].min_delta == 0.0


def test_degenerate_channel_range_rejected(tmp_path):
    data = _raw("channels.yaml")
    data["channels"][0]["min_delta"] = 0.0
    data["channels"][0]["max_delta"] = 0.0
    with pytest.raises(ConfigError, match="min_delta"):
        load_channel_config(_write(tmp_path, "channels.yaml", data))


def test_duplicate_channel_ids_rejected(tmp_path):
    data = _raw("channels.yaml")
    data["channels"][1]["channel_id"] = data["channels"][0]["channel_id"]
    with pytest.raises(ConfigError, match="duplicate channel_id"):
        load_channel_config(_write(tmp_path, "channels.yaml", data))


def test_replicate_warn_limit_cannot_exceed_fail_limit(tmp_path):
    data = _raw("tolerances.yaml")
    data["measurement_quality"]["replicate_std_warn"]["L"] = 5.0
    with pytest.raises(ConfigError, match="replicate_std_warn"):
        load_tolerance_config(_write(tmp_path, "tolerances.yaml", data))


def test_synthetic_config_must_be_labeled_synthetic(tmp_path):
    data = _raw("synthetic_scenarios.yaml")
    data["data_origin"] = "client"
    with pytest.raises(ConfigError, match="synthetic"):
        load_synthetic_config(_write(tmp_path, "synthetic_scenarios.yaml", data))


def test_missing_scenario_rejected(tmp_path):
    data = _raw("synthetic_scenarios.yaml")
    del data["scenarios"]["unseen_sku"]
    with pytest.raises(ConfigError, match="unseen_sku"):
        load_synthetic_config(_write(tmp_path, "synthetic_scenarios.yaml", data))


def test_scenario_range_must_be_ordered(tmp_path):
    data = _raw("synthetic_scenarios.yaml")
    data["scenarios"]["near_boundary"]["uncorrected_de00_range"] = [2.0, 1.0]
    with pytest.raises(ConfigError, match="range minimum"):
        load_synthetic_config(_write(tmp_path, "synthetic_scenarios.yaml", data))


def test_sparse_context_must_exist(tmp_path):
    data = _raw("synthetic_scenarios.yaml")
    data["generator"]["sparse_calibration_context_id"] = "ctx_nonexistent"
    with pytest.raises(ConfigError, match="sparse_calibration_context_id"):
        load_synthetic_config(_write(tmp_path, "synthetic_scenarios.yaml", data))


# ---- Config directory resolution --------------------------------------------


def test_default_config_dir_contains_shipped_files():
    directory = default_config_dir()
    for name in ("tolerances.yaml", "channels.yaml", "synthetic_scenarios.yaml"):
        assert (directory / name).is_file()


def test_env_var_overrides_config_dir(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV_CONFIG_DIR, str(tmp_path))
    assert default_config_dir() == tmp_path
    with pytest.raises(ConfigError, match="not found"):
        load_tolerance_config()  # nothing in tmp_path


def test_env_var_pointing_nowhere_raises(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV_CONFIG_DIR, str(tmp_path / "missing"))
    with pytest.raises(ConfigError, match="directory not found"):
        default_config_dir()