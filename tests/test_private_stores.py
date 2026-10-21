"""Inert filesystem controls for every credential-store writer, under umask 022."""
import errno
import json
import os
import stat
from pathlib import Path

import pytest

from ainode.core import atomic


@pytest.fixture(autouse=True)
def permissive_umask():
    previous = os.umask(0o022)
    try:
        yield
    finally:
        os.umask(previous)


@pytest.fixture(params=['auth', 'users', 'secrets', 'join'])
def store(request, tmp_path, monkeypatch):
    from ainode.auth import accounts, middleware
    from ainode.cluster import join
    from ainode.secrets import manager
    target = tmp_path / (request.param + '.json')
    if request.param == 'auth':
        monkeypatch.setattr(middleware, 'AINODE_HOME', tmp_path)
        monkeypatch.setattr(middleware, 'AUTH_FILE', target)
        obj = middleware.AuthConfig(enabled=True, api_keys=[{'key_hash': 'synthetic'}])
        module, save = middleware, obj.save
        expected = {'enabled': True, 'api_keys': [{'key_hash': 'synthetic'}]}
    elif request.param == 'users':
        obj = accounts.UsersStore(target)
        obj.users, obj.sessions = [{'name': 'synthetic'}], [{'token_hash': 'synthetic'}]
        module, save = accounts, obj.save
        expected = {'users': obj.users, 'sessions': obj.sessions}
    elif request.param == 'secrets':
        monkeypatch.setattr(manager, 'AINODE_HOME', tmp_path)
        # No host machine identifier or real profile is read by this fixture.
        monkeypatch.setattr(manager, '_machine_id', lambda: b'synthetic-test-only')
        obj = manager.SecretsManager(target)
        obj._values = {'openai_api_key': 'synthetic-value'}
        obj._custom = {'synthetic': 'synthetic-custom'}
        module, save = manager, obj._save
        expected = {'known': {'openai_api_key': manager._obfuscate('synthetic-value')},
                    'custom': {'synthetic': manager._obfuscate('synthetic-custom')}}
    else:
        obj = join.JoinTokenStore(target)
        rows = [{'token_hash': 'synthetic', 'used': False}]
        module, save = join, lambda: obj._save(rows)
        expected = {'tokens': rows}
    return module, save, target, expected, obj


def test_store_delegates_exact_format_and_propagates_failure(store, monkeypatch):
    module, save, target, expected, obj = store
    calls = []

    def refuse(path, text):
        calls.append((path, text))
        raise OSError('synthetic private-mode failure')

    monkeypatch.setattr(module, 'atomic_private_write', refuse)
    stamp = getattr(obj, '_stamp', None)
    with pytest.raises(OSError, match='private-mode failure'):
        save()
    assert calls == [(target, json.dumps(expected, indent=2))]
    assert not target.exists()
    assert getattr(obj, '_stamp', None) == stamp


def observe_writer(monkeypatch, target, expected):
    """Observe the actual inode before write, and publication/sync ordering."""
    events = []
    real_open, real_fdopen = os.open, os.fdopen
    real_sync, real_replace = os.fsync, os.replace

    def opened(path, flags, mode=0o777, *args, **kwargs):
        fd = real_open(path, flags, mode, *args, **kwargs)
        if flags & os.O_CREAT:
            assert flags & os.O_EXCL
            assert mode == 0o600
            assert stat.S_IMODE(os.fstat(fd).st_mode) == 0o600
            assert os.fstat(fd).st_size == 0
            events.append('exclusive-0600-create')
        return fd

    class Stream:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            self.stream.__enter__()
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def __getattr__(self, name):
            return getattr(self.stream, name)

        def write(self, text):
            assert events == ['exclusive-0600-create']
            assert stat.S_IMODE(os.fstat(self.fileno()).st_mode) == 0o600
            assert os.fstat(self.fileno()).st_size == 0
            assert json.loads(text) == expected
            events.append('private-before-content')
            return self.stream.write(text)

    def synced(fd):
        events.append('directory-sync' if stat.S_ISDIR(os.fstat(fd).st_mode) else 'file-sync')
        real_sync(fd)

    def replaced(source, destination):
        assert events == ['exclusive-0600-create', 'private-before-content', 'file-sync']
        assert Path(destination) == target
        assert stat.S_IMODE(Path(source).stat().st_mode) == 0o600
        real_replace(source, destination)
        events.append('atomic-replace')

    monkeypatch.setattr(os, 'open', opened)
    monkeypatch.setattr(os, 'fdopen', lambda *a, **kw: Stream(real_fdopen(*a, **kw)))
    monkeypatch.setattr(os, 'fsync', synced)
    monkeypatch.setattr(os, 'replace', replaced)
    return events


def test_store_private_from_creation_through_durable_replace(store, monkeypatch):
    _, save, target, expected, _ = store
    target.write_text('previous')
    events = observe_writer(monkeypatch, target, expected)
    save()
    assert events == ['exclusive-0600-create', 'private-before-content', 'file-sync',
                      'atomic-replace', 'directory-sync']
    assert json.loads(target.read_text()) == expected
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert list(target.parent.iterdir()) == [target]


@pytest.mark.parametrize('boundary', ['mode', 'file-sync', 'replace', 'published-mode', 'directory-sync'])
def test_store_failures_are_visible_and_clean_temporary_inode(store, monkeypatch, boundary):
    _, save, target, expected, obj = store
    target.write_text('previous')
    before_stamp = getattr(obj, '_stamp', None)

    def refuse(*args):
        raise OSError('synthetic failure')

    if boundary == 'mode':
        monkeypatch.setattr(os, 'fchmod', refuse)
    elif boundary == 'replace':
        monkeypatch.setattr(os, 'replace', refuse)
    elif boundary == 'published-mode':
        replace = os.replace

        def wrong_mode(source, destination):
            replace(source, destination)
            os.chmod(destination, 0o644)

        monkeypatch.setattr(os, 'replace', wrong_mode)
    else:
        sync = os.fsync

        def fail_sync(fd):
            directory = stat.S_ISDIR(os.fstat(fd).st_mode)
            if directory == (boundary == 'directory-sync'):
                refuse()
            sync(fd)

        monkeypatch.setattr(os, 'fsync', fail_sync)
    with pytest.raises(OSError):
        save()
    assert list(target.parent.iterdir()) == [target]
    if boundary in ['published-mode', 'directory-sync']:
        # A post-publication failure cannot roll back a completed replacement.
        assert json.loads(target.read_text()) == expected
    else:
        assert target.read_text() == 'previous'
    assert getattr(obj, '_stamp', None) == before_stamp


def test_exclusive_creation_refuses_existing_candidate_without_opening_it(tmp_path, monkeypatch):
    """Simulate EEXIST at the open boundary; never construct a live attack."""
    opened = os.open
    seen = []

    def collide(path, flags, mode=0o777, *args, **kwargs):
        if flags & os.O_CREAT:
            assert flags & os.O_EXCL
            assert mode == 0o600
            seen.append(Path(path))
            if len(seen) == 1:
                # O_EXCL refuses any occupied candidate, including a symlink.
                raise FileExistsError(errno.EEXIST, 'synthetic occupied symlink candidate')
        return opened(path, flags, mode, *args, **kwargs)

    monkeypatch.setattr(os, 'open', collide)
    target = tmp_path / 'store.json'
    atomic.atomic_private_write(target, 'synthetic')
    assert len(seen) == 2 and seen[0] != seen[1]
    assert target.read_text() == 'synthetic'
    assert list(tmp_path.iterdir()) == [target]
