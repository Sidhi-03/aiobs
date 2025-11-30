"""Tests for session labeling functionality."""

import json
import os
import pytest
from unittest.mock import patch, MagicMock

from aiobs import observer
from aiobs.collector import (
    _validate_label_key,
    _validate_label_value,
    _validate_labels,
    _get_env_labels,
    _get_system_labels,
    LABEL_KEY_PATTERN,
    LABEL_VALUE_MAX_LENGTH,
    LABEL_MAX_COUNT,
    LABEL_RESERVED_PREFIX,
    LABEL_ENV_PREFIX,
)


@pytest.fixture(autouse=True)
def reset_observer():
    """Reset observer state before and after each test."""
    observer.reset()
    yield
    observer.reset()


@pytest.fixture
def mock_api_validation():
    """Mock API key validation to avoid network calls."""
    with patch.object(observer, "_validate_api_key"):
        yield


class TestLabelKeyValidation:
    """Tests for label key validation."""

    def test_valid_keys(self):
        """Valid keys should not raise."""
        valid_keys = [
            "environment",
            "team",
            "a",
            "a1",
            "my_label",
            "label_with_numbers_123",
            "a" * 63,  # Max length
        ]
        for key in valid_keys:
            _validate_label_key(key)  # Should not raise

    def test_invalid_key_uppercase(self):
        """Uppercase keys should be rejected."""
        with pytest.raises(ValueError, match="is invalid"):
            _validate_label_key("Environment")

    def test_invalid_key_starts_with_number(self):
        """Keys starting with numbers should be rejected."""
        with pytest.raises(ValueError, match="is invalid"):
            _validate_label_key("1environment")

    def test_invalid_key_hyphen(self):
        """Keys with hyphens should be rejected."""
        with pytest.raises(ValueError, match="is invalid"):
            _validate_label_key("my-label")

    def test_invalid_key_space(self):
        """Keys with spaces should be rejected."""
        with pytest.raises(ValueError, match="is invalid"):
            _validate_label_key("my label")

    def test_invalid_key_empty(self):
        """Empty keys should be rejected."""
        with pytest.raises(ValueError, match="is invalid"):
            _validate_label_key("")

    def test_invalid_key_too_long(self):
        """Keys exceeding max length should be rejected."""
        with pytest.raises(ValueError, match="is invalid"):
            _validate_label_key("a" * 64)

    def test_reserved_prefix_rejected(self):
        """Keys with reserved prefix should be rejected."""
        with pytest.raises(ValueError, match="reserved prefix"):
            _validate_label_key("aiobs_custom")

    def test_non_string_key(self):
        """Non-string keys should be rejected."""
        with pytest.raises(ValueError, match="must be a string"):
            _validate_label_key(123)


class TestLabelValueValidation:
    """Tests for label value validation."""

    def test_valid_values(self):
        """Valid values should not raise."""
        valid_values = [
            "production",
            "my-value",
            "Value With Spaces",
            "émoji 🚀",
            "",  # Empty is valid
            "x" * LABEL_VALUE_MAX_LENGTH,  # Max length
        ]
        for value in valid_values:
            _validate_label_value(value)  # Should not raise

    def test_value_too_long(self):
        """Values exceeding max length should be rejected."""
        with pytest.raises(ValueError, match="exceeds maximum length"):
            _validate_label_value("x" * (LABEL_VALUE_MAX_LENGTH + 1))

    def test_non_string_value(self):
        """Non-string values should be rejected."""
        with pytest.raises(ValueError, match="must be a string"):
            _validate_label_value(123, "mykey")


class TestLabelsValidation:
    """Tests for full labels dictionary validation."""

    def test_valid_labels(self):
        """Valid labels dictionary should not raise."""
        labels = {
            "environment": "production",
            "team": "ml-platform",
            "version": "v1.0.0",
        }
        _validate_labels(labels)  # Should not raise

    def test_empty_labels(self):
        """Empty labels dictionary should not raise."""
        _validate_labels({})  # Should not raise

    def test_too_many_labels(self):
        """Exceeding max label count should raise."""
        labels = {f"key{i}": f"value{i}" for i in range(LABEL_MAX_COUNT + 1)}
        with pytest.raises(ValueError, match="Too many labels"):
            _validate_labels(labels)

    def test_max_labels_allowed(self):
        """Exactly max labels should be allowed."""
        labels = {f"key{i}": f"value{i}" for i in range(LABEL_MAX_COUNT)}
        _validate_labels(labels)  # Should not raise

    def test_non_dict_labels(self):
        """Non-dict labels should be rejected."""
        with pytest.raises(ValueError, match="must be a dictionary"):
            _validate_labels(["key", "value"])


class TestEnvLabels:
    """Tests for environment variable label detection."""

    def test_get_env_labels(self):
        """Should extract labels from AIOBS_LABEL_* env vars."""
        with patch.dict(os.environ, {
            "AIOBS_LABEL_ENVIRONMENT": "production",
            "AIOBS_LABEL_TEAM": "ml-platform",
            "OTHER_VAR": "ignored",
        }, clear=False):
            labels = _get_env_labels()
            assert labels == {
                "environment": "production",
                "team": "ml-platform",
            }

    def test_env_labels_lowercase(self):
        """Env var names should be converted to lowercase."""
        with patch.dict(os.environ, {
            "AIOBS_LABEL_MY_LABEL": "value",
        }, clear=False):
            labels = _get_env_labels()
            assert "my_label" in labels

    def test_env_labels_truncated(self):
        """Long env var values should be truncated."""
        long_value = "x" * 500
        with patch.dict(os.environ, {
            "AIOBS_LABEL_LONG": long_value,
        }, clear=False):
            labels = _get_env_labels()
            assert len(labels["long"]) == LABEL_VALUE_MAX_LENGTH

    def test_empty_env_label_key_ignored(self):
        """Empty label key after prefix should be ignored."""
        with patch.dict(os.environ, {
            "AIOBS_LABEL_": "value",
        }, clear=False):
            labels = _get_env_labels()
            assert "" not in labels


class TestSystemLabels:
    """Tests for system-generated labels."""

    def test_system_labels_present(self):
        """System labels should include expected keys."""
        labels = _get_system_labels()
        assert "aiobs_sdk_version" in labels
        assert "aiobs_python_version" in labels
        assert "aiobs_hostname" in labels
        assert "aiobs_os" in labels

    def test_system_labels_have_values(self):
        """System labels should have non-empty values."""
        labels = _get_system_labels()
        for key, value in labels.items():
            assert value, f"System label {key} should have a value"


class TestObserveWithLabels:
    """Tests for observe() method with labels parameter."""

    def test_observe_with_labels(self, mock_api_validation):
        """Labels passed to observe() should be stored."""
        labels = {"environment": "test", "team": "testing"}
        observer.observe(api_key="test_key", labels=labels)
        
        stored_labels = observer.get_labels()
        assert stored_labels["environment"] == "test"
        assert stored_labels["team"] == "testing"

    def test_observe_without_labels(self, mock_api_validation):
        """observe() without labels should still have system labels."""
        observer.observe(api_key="test_key")
        
        stored_labels = observer.get_labels()
        assert "aiobs_sdk_version" in stored_labels

    def test_observe_labels_merged_with_system(self, mock_api_validation):
        """User labels should be merged with system labels."""
        observer.observe(api_key="test_key", labels={"custom": "value"})
        
        stored_labels = observer.get_labels()
        assert stored_labels["custom"] == "value"
        assert "aiobs_sdk_version" in stored_labels

    def test_observe_labels_override_env(self, mock_api_validation):
        """Explicit labels should override env var labels."""
        with patch.dict(os.environ, {"AIOBS_LABEL_ENVIRONMENT": "from_env"}, clear=False):
            observer.observe(api_key="test_key", labels={"environment": "explicit"})
            
            stored_labels = observer.get_labels()
            assert stored_labels["environment"] == "explicit"

    def test_observe_invalid_labels_rejected(self, mock_api_validation):
        """Invalid labels should raise ValueError."""
        with pytest.raises(ValueError, match="is invalid"):
            observer.observe(api_key="test_key", labels={"Invalid-Key": "value"})

    def test_observe_reserved_labels_rejected(self, mock_api_validation):
        """Reserved prefix labels should raise ValueError."""
        with pytest.raises(ValueError, match="reserved prefix"):
            observer.observe(api_key="test_key", labels={"aiobs_custom": "value"})


class TestSetLabels:
    """Tests for set_labels() method."""

    def test_set_labels_merge(self, mock_api_validation):
        """set_labels with merge=True should merge with existing."""
        observer.observe(api_key="test_key", labels={"key1": "value1"})
        observer.set_labels({"key2": "value2"})
        
        labels = observer.get_labels()
        assert labels["key1"] == "value1"
        assert labels["key2"] == "value2"

    def test_set_labels_replace(self, mock_api_validation):
        """set_labels with merge=False should replace user labels."""
        observer.observe(api_key="test_key", labels={"key1": "value1"})
        observer.set_labels({"key2": "value2"}, merge=False)
        
        labels = observer.get_labels()
        assert "key1" not in labels
        assert labels["key2"] == "value2"
        # System labels should be preserved
        assert "aiobs_sdk_version" in labels

    def test_set_labels_update_existing(self, mock_api_validation):
        """set_labels should update existing keys."""
        observer.observe(api_key="test_key", labels={"key1": "original"})
        observer.set_labels({"key1": "updated"})
        
        labels = observer.get_labels()
        assert labels["key1"] == "updated"

    def test_set_labels_no_session_raises(self):
        """set_labels without active session should raise."""
        with pytest.raises(RuntimeError, match="No active session"):
            observer.set_labels({"key": "value"})

    def test_set_labels_invalid_rejected(self, mock_api_validation):
        """Invalid labels in set_labels should raise ValueError."""
        observer.observe(api_key="test_key")
        with pytest.raises(ValueError, match="is invalid"):
            observer.set_labels({"Invalid": "value"})

    def test_set_labels_exceeds_limit(self, mock_api_validation):
        """set_labels exceeding limit after merge should raise."""
        # Start with near-max labels
        initial_labels = {f"key{i}": f"value{i}" for i in range(LABEL_MAX_COUNT - 5)}
        observer.observe(api_key="test_key", labels=initial_labels)
        
        # Try to add too many more
        new_labels = {f"new{i}": f"value{i}" for i in range(10)}
        with pytest.raises(ValueError, match="Too many labels"):
            observer.set_labels(new_labels)


class TestAddLabel:
    """Tests for add_label() method."""

    def test_add_label(self, mock_api_validation):
        """add_label should add a single label."""
        observer.observe(api_key="test_key")
        observer.add_label("custom", "value")
        
        labels = observer.get_labels()
        assert labels["custom"] == "value"

    def test_add_label_update_existing(self, mock_api_validation):
        """add_label should update existing key."""
        observer.observe(api_key="test_key", labels={"key": "original"})
        observer.add_label("key", "updated")
        
        labels = observer.get_labels()
        assert labels["key"] == "updated"

    def test_add_label_no_session_raises(self):
        """add_label without active session should raise."""
        with pytest.raises(RuntimeError, match="No active session"):
            observer.add_label("key", "value")

    def test_add_label_invalid_key_rejected(self, mock_api_validation):
        """Invalid key in add_label should raise ValueError."""
        observer.observe(api_key="test_key")
        with pytest.raises(ValueError, match="is invalid"):
            observer.add_label("Invalid-Key", "value")

    def test_add_label_invalid_value_rejected(self, mock_api_validation):
        """Invalid value in add_label should raise ValueError."""
        observer.observe(api_key="test_key")
        with pytest.raises(ValueError, match="exceeds maximum length"):
            observer.add_label("key", "x" * (LABEL_VALUE_MAX_LENGTH + 1))

    def test_add_label_exceeds_limit(self, mock_api_validation):
        """add_label exceeding limit should raise."""
        labels = {f"key{i}": f"value{i}" for i in range(LABEL_MAX_COUNT - 4)}  # Leave room for system labels
        observer.observe(api_key="test_key", labels=labels)
        
        # Fill up to max
        current_count = len(observer.get_labels())
        for i in range(LABEL_MAX_COUNT - current_count):
            observer.add_label(f"extra{i}", "value")
        
        # Next add should fail
        with pytest.raises(ValueError, match="Maximum"):
            observer.add_label("one_too_many", "value")


class TestRemoveLabel:
    """Tests for remove_label() method."""

    def test_remove_label(self, mock_api_validation):
        """remove_label should remove the label."""
        observer.observe(api_key="test_key", labels={"key": "value"})
        observer.remove_label("key")
        
        labels = observer.get_labels()
        assert "key" not in labels

    def test_remove_nonexistent_label(self, mock_api_validation):
        """remove_label on nonexistent key should not raise."""
        observer.observe(api_key="test_key")
        observer.remove_label("nonexistent")  # Should not raise

    def test_remove_system_label_rejected(self, mock_api_validation):
        """remove_label on system label should raise."""
        observer.observe(api_key="test_key")
        with pytest.raises(ValueError, match="Cannot remove system label"):
            observer.remove_label("aiobs_sdk_version")

    def test_remove_label_no_session_raises(self):
        """remove_label without active session should raise."""
        with pytest.raises(RuntimeError, match="No active session"):
            observer.remove_label("key")


class TestGetLabels:
    """Tests for get_labels() method."""

    def test_get_labels(self, mock_api_validation):
        """get_labels should return current labels."""
        observer.observe(api_key="test_key", labels={"key": "value"})
        
        labels = observer.get_labels()
        assert "key" in labels
        assert labels["key"] == "value"

    def test_get_labels_returns_copy(self, mock_api_validation):
        """get_labels should return a copy, not the original."""
        observer.observe(api_key="test_key", labels={"key": "value"})
        
        labels = observer.get_labels()
        labels["new_key"] = "new_value"
        
        # Original should be unchanged
        stored_labels = observer.get_labels()
        assert "new_key" not in stored_labels

    def test_get_labels_no_session_raises(self):
        """get_labels without active session should raise."""
        with pytest.raises(RuntimeError, match="No active session"):
            observer.get_labels()


class TestLabelsExport:
    """Tests for labels in exported data."""

    def test_labels_in_flush_output(self, mock_api_validation, tmp_path):
        """Labels should be included in flush output."""
        observer.observe(api_key="test_key", labels={"environment": "test"})
        
        output_file = tmp_path / "output.json"
        observer.flush(path=str(output_file))
        
        with open(output_file) as f:
            data = json.load(f)
        
        session = data["sessions"][0]
        assert "labels" in session
        assert session["labels"]["environment"] == "test"

    def test_system_labels_in_export(self, mock_api_validation, tmp_path):
        """System labels should be included in export."""
        observer.observe(api_key="test_key")
        
        output_file = tmp_path / "output.json"
        observer.flush(path=str(output_file))
        
        with open(output_file) as f:
            data = json.load(f)
        
        session = data["sessions"][0]
        assert "aiobs_sdk_version" in session["labels"]

    def test_dynamic_labels_in_export(self, mock_api_validation, tmp_path):
        """Dynamically added labels should be in export."""
        observer.observe(api_key="test_key")
        observer.add_label("added_later", "value")
        
        output_file = tmp_path / "output.json"
        observer.flush(path=str(output_file))
        
        with open(output_file) as f:
            data = json.load(f)
        
        session = data["sessions"][0]
        assert session["labels"]["added_later"] == "value"

