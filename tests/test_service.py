"""Tests for ainode.service.systemd — unit file generation and service management."""

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from ainode.service import systemd


class TestUnitFileGeneration:
    """Test systemd unit file content generation."""

    def test_generate_system_unit(self):
        """System-level unit file uses multi-user.target."""
        content = systemd.generate_unit_file(user_mode=False)
        assert "WantedBy=multi-user.target" in content
        assert "ainode start" in content
        assert "NVIDIA_VISIBLE_DEVICES=all" in content
        assert "After=network.target nvidia-persistenced.service" in content
        assert "Restart=on-failure" in content

    def test_generate_user_unit(self):
        """User-level unit file uses default.target."""
        content = systemd.generate_unit_file(user_mode=True)
        assert "WantedBy=default.target" in content
        assert "ainode start" in content

    def test_unit_has_gpu_env_vars(self):
        content = systemd.generate_unit_file()
        assert "NVIDIA_VISIBLE_DEVICES=all" in content
        assert "CUDA_DEVICE_ORDER=PCI_BUS_ID" in content

    def test_unit_has_hardening(self):
        content = systemd.generate_unit_file()
        assert "NoNewPrivileges=true" in content
        assert "ProtectSystem=strict" in content

    def test_unit_has_restart_policy(self):
        content = systemd.generate_unit_file()
        assert "Restart=on-failure" in content
        assert "RestartSec=10" in content

    def test_unit_has_description(self):
        content = systemd.generate_unit_file()
        assert "Description=AINode" in content
        assert "Documentation=https://ainode.dev" in content


class TestIsInstalled:
    """Test is_installed checks."""

    def test_not_installed(self, tmp_path):
        with patch.object(systemd, "SYSTEM_UNIT_DIR", tmp_path):
            assert systemd.is_installed(user_mode=False) is False

    def test_installed(self, tmp_path):
        unit_file = tmp_path / "ainode.service"
        unit_file.write_text("[Unit]\nDescription=test\n")
        with patch.object(systemd, "SYSTEM_UNIT_DIR", tmp_path):
            assert systemd.is_installed(user_mode=False) is True

    def test_user_mode_not_installed(self, tmp_path):
        with patch.object(systemd, "USER_UNIT_DIR", tmp_path):
            assert systemd.is_installed(user_mode=True) is False


class TestInstallService:
    """Test install_service writes unit file and reloads daemon."""

    def test_install_writes_unit_file(self, tmp_path):
        with (
            patch.object(systemd, "SYSTEM_UNIT_DIR", tmp_path),
            patch.object(systemd, "_systemctl") as mock_ctl,
        ):
            systemd.install_service(user_mode=False)

            unit_path = tmp_path / "ainode.service"
            assert unit_path.exists()
            content = unit_path.read_text()
            assert "ainode start" in content
            mock_ctl.assert_called_once_with(["daemon-reload"], user_mode=False)

    def test_install_user_mode(self, tmp_path):
        with (
            patch.object(systemd, "USER_UNIT_DIR", tmp_path),
            patch.object(systemd, "_systemctl") as mock_ctl,
        ):
            systemd.install_service(user_mode=True)

            unit_path = tmp_path / "ainode.service"
            assert unit_path.exists()
            content = unit_path.read_text()
            assert "WantedBy=default.target" in content
            mock_ctl.assert_called_once_with(["daemon-reload"], user_mode=True)


class TestUninstallService:
    """Test uninstall_service stops, disables, removes, and reloads."""

    def test_uninstall_removes_unit(self, tmp_path):
        unit_file = tmp_path / "ainode.service"
        unit_file.write_text("[Unit]\nDescription=test\n")

        with (
            patch.object(systemd, "SYSTEM_UNIT_DIR", tmp_path),
            patch.object(systemd, "_systemctl") as mock_ctl,
        ):
            systemd.uninstall_service(user_mode=False)

            assert not unit_file.exists()
            # Should have called stop, disable, daemon-reload
            calls = mock_ctl.call_args_list
            assert any("stop" in str(c) for c in calls)
            assert any("disable" in str(c) for c in calls)
            assert any("daemon-reload" in str(c) for c in calls)

    def test_uninstall_noop_if_not_installed(self, tmp_path):
        with (
            patch.object(systemd, "SYSTEM_UNIT_DIR", tmp_path),
            patch.object(systemd, "_systemctl") as mock_ctl,
        ):
            systemd.uninstall_service(user_mode=False)
            mock_ctl.assert_not_called()


class TestServiceActions:
    """Test enable, disable, start, stop, restart delegate to systemctl."""

    @patch.object(systemd, "_systemctl")
    def test_enable(self, mock_ctl):
        systemd.enable_service(user_mode=False)
        mock_ctl.assert_called_once_with(["enable", "ainode.service"], user_mode=False)

    @patch.object(systemd, "_systemctl")
    def test_disable(self, mock_ctl):
        systemd.disable_service(user_mode=True)
        mock_ctl.assert_called_once_with(["disable", "ainode.service"], user_mode=True)

    @patch.object(systemd, "_systemctl")
    def test_start(self, mock_ctl):
        systemd.start_service()
        mock_ctl.assert_called_once_with(["start", "ainode.service"], user_mode=False)

    @patch.object(systemd, "_systemctl")
    def test_stop(self, mock_ctl):
        systemd.stop_service()
        mock_ctl.assert_called_once_with(["stop", "ainode.service"], user_mode=False)

    @patch.object(systemd, "_systemctl")
    def test_restart(self, mock_ctl):
        systemd.restart_service()
        mock_ctl.assert_called_once_with(["restart", "ainode.service"], user_mode=False)


class TestStatusService:
    """Test status_service returns structured info."""

    @patch.object(systemd, "get_journal_lines", return_value=["line1", "line2"])
    @patch.object(systemd, "_systemctl")
    def test_status_active(self, mock_ctl, mock_journal):
        mock_ctl.side_effect = [
            MagicMock(stdout="active\n"),   # is-active
            MagicMock(stdout="enabled\n"),  # is-enabled
        ]
        info = systemd.status_service(user_mode=False)
        assert info["state"] == "active"
        assert info["enabled"] is True
        assert info["journal_lines"] == ["line1", "line2"]

    @patch.object(systemd, "get_journal_lines", return_value=[])
    @patch.object(systemd, "_systemctl")
    def test_status_inactive(self, mock_ctl, mock_journal):
        mock_ctl.side_effect = [
            MagicMock(stdout="inactive\n"),
            MagicMock(stdout="disabled\n"),
        ]
        info = systemd.status_service()
        assert info["state"] == "inactive"
        assert info["enabled"] is False


class TestGetJournalLines:
    """Test journal line fetching."""

    @patch("subprocess.run")
    def test_returns_lines(self, mock_run):
        mock_run.return_value = MagicMock(stdout="line1\nline2\nline3\n")
        lines = systemd.get_journal_lines(lines=3)
        assert lines == ["line1", "line2", "line3"]

    @patch("subprocess.run", side_effect=FileNotFoundError)
    def test_returns_empty_on_missing_journalctl(self, mock_run):
        lines = systemd.get_journal_lines()
        assert lines == []

    @patch("subprocess.run", side_effect=subprocess.TimeoutExpired("journalctl", 10))
    def test_returns_empty_on_timeout(self, mock_run):
        lines = systemd.get_journal_lines()
        assert lines == []


class TestSystemctlHelper:
    """Test _systemctl builds the correct command."""

    @patch("subprocess.run")
    def test_system_mode(self, mock_run):
        systemd._systemctl(["start", "ainode.service"], user_mode=False)
        mock_run.assert_called_once_with(
            ["systemctl", "start", "ainode.service"], check=True, timeout=30
        )

    @patch("subprocess.run")
    def test_user_mode(self, mock_run):
        systemd._systemctl(["start", "ainode.service"], user_mode=True)
        mock_run.assert_called_once_with(
            ["systemctl", "--user", "start", "ainode.service"], check=True, timeout=30
        )

    @patch("subprocess.run")
    def test_capture_mode(self, mock_run):
        mock_run.return_value = MagicMock(stdout="active\n")
        result = systemd._systemctl(["is-active", "ainode.service"], capture=True)
        mock_run.assert_called_once_with(
            ["systemctl", "is-active", "ainode.service"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.stdout == "active\n"
