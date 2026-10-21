"""Offline regressions for public installer credential and privilege boundaries."""
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import sys

import pytest

from ainode.core import atomic

ROOT = Path(__file__).resolve().parents[1]
INSTALLER = ROOT / 'scripts/install.sh'


@pytest.mark.parametrize('existing', [False, True])
def test_atomic_private_from_first_byte(tmp_path, monkeypatch, existing):
    target = tmp_path / 'config.json'
    if existing:
        target.write_text('old')
    victim = tmp_path / 'victim'
    victim.write_text('untouched')
    target.with_name('config.json.tmp').symlink_to(victim)
    original = os.fsync
    observed = []

    def fsync(fd):
        mode = os.fstat(fd).st_mode
        if stat.S_ISREG(mode):
            observed.append(stat.S_IMODE(mode))
        original(fd)

    monkeypatch.setattr(os, 'fsync', fsync)
    mask = os.umask(0)
    try:
        atomic.atomic_private_write(target, 'private credential')
    finally:
        os.umask(mask)
    assert observed == [0o600]
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert victim.read_text() == 'untouched'
    assert target.read_text() == 'private credential'


def test_failed_sync_preserves_config_and_cleans_temp(tmp_path, monkeypatch):
    target = tmp_path / 'config.json'
    target.write_text('previous')
    monkeypatch.setattr(os, 'fsync', lambda fd: (_ for _ in ()).throw(OSError('disk')))
    with pytest.raises(OSError):
        atomic.atomic_private_write(target, 'new secret')
    assert target.read_text() == 'previous'
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.parametrize('writer', ['node', 'join', 'tls', 'doctor'])
def test_all_writers_keep_mode_and_surgical_unknown_keys(tmp_path, monkeypatch, writer):
    from ainode.core import config
    from ainode.cluster.join import merge_config_keys
    from ainode.tls.config import TLSConfig, save_tls_config
    from ainode.cli.doctor import _write_config_key
    target = tmp_path / 'config.json'
    target.write_text(json.dumps({'cluster_secret': 'synthetic', 'future_key': {'keep': 1}}))
    mask = os.umask(0o022)
    try:
        if writer == 'node':
            monkeypatch.setattr(config, 'CONFIG_FILE', target)
            monkeypatch.setattr(config, 'AINODE_HOME', tmp_path)
            config.NodeConfig().save()
        elif writer == 'join':
            merge_config_keys({'cluster_id': 'synthetic'}, path=target)
        elif writer == 'tls':
            save_tls_config(TLSConfig(), config_path=target)
        else:
            _write_config_key(target, 'discovery_port', 54321)
    finally:
        os.umask(mask)
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    if writer != 'node':
        assert json.loads(target.read_text())['future_key'] == {'keep': 1}
        assert json.loads(target.read_text())['cluster_secret'] == 'synthetic'


def test_installer_embeds_canonical_writer():
    source = INSTALLER.read_text().split("<<'ATOMIC_PY'\n", 1)[1].split('ATOMIC_PY\n', 1)[0]
    assert source == (ROOT / 'ainode/core/atomic.py').read_text()
    text = INSTALLER.read_text()
    assert 'write_private_config "$AINODE_HOME/config.json" << CONFIG' in text
    assert 'write_private_config "$AUTH_JSON" << AUTHJSON' in text


@pytest.mark.parametrize('value', ['synthetic-token-only', ':synthetic-secret', 'synthetic-empty-host-token:'])
def test_installer_invalid_join_does_not_log_value(tmp_path, value):
    env = dict(os.environ, AINODE_HOME=str(tmp_path / 'home'),
               AINODE_IMAGE='synthetic:1', AINODE_VERSION='1', AINODE_NVIDIA_IMAGE='skip',
               AINODE_JOIN=value)
    result = subprocess.run(['bash', str(INSTALLER), '--dry-run'], env=env,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert value not in result.stdout + result.stderr
    assert stat.S_IMODE((tmp_path / 'home/config.json').stat().st_mode) == 0o600


@pytest.mark.parametrize("interrupted", [False, True])
def test_units_use_private_staging_and_root_owned_copy(tmp_path, interrupted):
    text = INSTALLER.read_text()
    block = text[text.index('STAGING_DIR=""'):text.index('# -- 5. Install host-side')]
    renewal = text[text.index('TLS_RENEW_STAGING='):text.index('# -- 6. Join an existing')]
    unit_dir = tmp_path / 'units'
    unit_dir.mkdir()
    # A hostile predictable file cannot be selected by the private-directory path.
    hostile = tmp_path / 'ainode.service'
    hostile.write_text('hostile')
    hostile.chmod(0o666)
    prelude = '''set -eu
umask 077
DRY_RUN=false
USER_MODE=false
AINODE_IMAGE=synthetic:1
AINODE_HOME=/synthetic
WANTED_BY=multi-user.target
WRAPPER_PATH=/synthetic/ainode
log() { :; }
systemctl() { :; }
sudo() { "$@"; }
'''
    # Executes real install only against synthetic paths; never systemd or Docker.
    if os.geteuid() != 0:
        # Simulate the privilege boundary without acquiring privileges. Verify
        # the exact requested ownership, then execute only the scratch copy.
        prelude += """sudo() {
    if [ "$1" = systemctl ]; then return 0; fi
    [ "$1 $2 $3 $4 $5 $6 $7 $8" = 'install -o root -g root -m 0644 --' ]
    [ -f "$9" ] && [ ! -L "$9" ] && [ -O "$9" ]
    case "$(stat -c %a "$(dirname "$9")")" in 700|2700) ;; *) return 1 ;; esac
    [ "$(stat -c %a "$9")" = 600 ]
    shift 8
    install -m 0644 -- "$@"
}
"""
    if interrupted:
        prelude += 'sudo() { return 9; }\n'
    result = subprocess.run(['bash', '-c', prelude + block + renewal],
                            env=dict(os.environ, TMPDIR=str(tmp_path), UNIT_DIR=str(unit_dir)),
                            capture_output=True, text=True)
    if interrupted:
        assert result.returncode == 9
        assert not list(tmp_path.glob('ainode-install.*'))
        assert not list(unit_dir.iterdir())
        return
    assert result.returncode == 0, result.stderr
    for name in ['ainode.service', 'ainode-tls-renew.service', 'ainode-tls-renew.timer']:
        info = (unit_dir / name).stat()
        assert info.st_uid == os.geteuid() and stat.S_IMODE(info.st_mode) == 0o644
    assert hostile.read_text() == 'hostile'
    assert not list(tmp_path.glob('ainode-install.*'))


def test_join_stdin_and_legacy_limits(monkeypatch, capsys):
    import ainode.cli.main as cli
    from types import SimpleNamespace
    token = 'synthetic-sensitive-join-token'
    args = SimpleNamespace(master='192.0.2.1:3000', token=None, token_stdin=True)
    monkeypatch.setattr(sys, 'stdin', io.StringIO(token + '\n'))
    seen = []
    monkeypatch.setattr(cli, '_http_post_json', lambda url, body: (seen.append(body) or (403, {'error': token})))
    with pytest.raises(SystemExit) as exc:
        cli.cmd_join(args)
    assert exc.value.code == 1
    assert seen == [{'token': token, 'node_name': ''}]
    assert token not in capsys.readouterr().out
    args.token_stdin = False
    args.token = token
    with pytest.raises(SystemExit) as exc:
        cli.cmd_join(args)
    assert exc.value.code == 2
    assert len(seen) == 1


def test_obsolete_site_sync_is_absent():
    assert not (ROOT / '.github/workflows/sync-site-installer.yml').exists()
    for path in (ROOT / '.github/workflows').glob('*.yml'):
        assert 'SITE_SYNC_TOKEN' not in path.read_text()
        assert 'x-access-token:' not in path.read_text()


def test_wrapper_forwards_join_token_only_on_stdin(tmp_path):
    home = tmp_path / 'home'
    env = dict(os.environ, AINODE_HOME=str(home), AINODE_IMAGE='synthetic:1',
               AINODE_VERSION='1', AINODE_NVIDIA_IMAGE='skip')
    rendered = subprocess.run(['bash', str(INSTALLER), '--dry-run'], env=env,
                              capture_output=True, text=True)
    assert rendered.returncode == 0, rendered.stderr
    wrapper = home / 'ainode-wrapper'
    assert wrapper.exists()
    bindir = tmp_path / 'bin'
    bindir.mkdir()
    docker = bindir / 'docker'
    docker.write_text('''#!/bin/bash
if [ "$*" = 'exec ainode true' ]; then exit 0; fi
printf '%s\\n' "$@" > "$TRACE_ARGS"
cat > "$TRACE_INPUT"
''')
    docker.chmod(0o700)
    token = 'synthetic-private-stdin-token'
    env.update(PATH=str(bindir) + ':' + env['PATH'], TRACE_ARGS=str(tmp_path / 'args'),
               TRACE_INPUT=str(tmp_path / 'input'))
    result = subprocess.run(['bash', str(wrapper), 'join', '192.0.2.1:3000', '--token-stdin'],
                            input=token + '\n', env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    argv = (tmp_path / 'args').read_text().splitlines()
    assert argv[:2] == ['exec', '-i']
    assert '--token-stdin' in argv
    assert token not in '\n'.join(argv) + result.stdout + result.stderr
    assert (tmp_path / 'input').read_text() == token + '\n'
    # The installer must select this tested transport, never a positional token.
    assert 'join "$JOIN_HOST" --token-stdin' in INSTALLER.read_text()
    assert 'join "$JOIN_HOST" "$JOIN_TOKEN"' not in INSTALLER.read_text()


def test_production_config_transport_uses_stdin_and_pinned_image(tmp_path):
    text = INSTALLER.read_text()
    block = text[text.index('ATOMIC_WRITER="'):text.index('# Dry-run has no image')]
    bindir = tmp_path / 'bin'
    bindir.mkdir()
    docker = bindir / 'docker'
    docker.write_text('''#!/bin/bash
set -eu
printf '%s\\n' "$@" > "$TRACE_ARGS"
[ "$1 $2 $3 $4 $5 $6 $7 $8 $9" = 'run --rm -i --user '"$(id -u):$(id -g)"' --entrypoint python3 -v '"$CONFIG_HOME:/config" ]
[ "${10}" = 'synthetic:old-pinned' ]
shift 10
[ "$1" = -c ]
code="$2"
exec python3 -c "$code" "$CONFIG_HOME/config.json"
''')
    docker.chmod(0o700)
    payload = '{"cluster_secret":"synthetic-transport-secret"}'
    script = 'set -eu\nDRY_RUN=false\nAINODE_IMAGE=synthetic:old-pinned\n' + block
    script += '\nwrite_private_config "$AINODE_HOME/config.json"\n'
    result = subprocess.run(['bash', '-c', script], input=payload, capture_output=True, text=True,
                            env=dict(os.environ, PATH=str(bindir) + ':' + os.environ['PATH'],
                                     CONFIG_HOME=str(tmp_path), AINODE_HOME=str(tmp_path),
                                     TRACE_ARGS=str(tmp_path / 'args')))
    assert result.returncode == 0, result.stderr
    assert (tmp_path / 'config.json').read_text() == payload
    assert stat.S_IMODE((tmp_path / 'config.json').stat().st_mode) == 0o600
    assert (tmp_path / 'config.json').stat().st_uid == os.geteuid()
    assert 'synthetic-transport-secret' not in (tmp_path / 'args').read_text()
