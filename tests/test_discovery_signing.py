"""Signed discovery: what a node accepts off the wire, and what it refuses.

Issue #169. ``cluster_secret`` was declared in the config and scrubbed from
``/api/config`` as though it protected something, and no code path ever read it.
One UDP datagram was the whole join protocol: the cluster id is readable
unauthenticated over HTTP, the election prefers whoever announces
``role: "master"``, and an announcement also advertises the instances peers will
route ``/v1/chat/completions`` traffic to. So a host on the broadcast domain
could take the fleet over, and nothing anywhere would say so.

Issue #171 rides along in the same payload: the release is on the wire now, so a
cluster split across two of them stops looking exactly like one that is not.

The socket is faked rather than bound, so these run the REAL send and receive
loops (the only place the decision can be made) with no network, no port
conflict between parallel runs, and no sleeping.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import asdict
from typing import List, Optional, Tuple

import pytest

from ainode.discovery import broadcast as bc
from ainode.discovery.broadcast import (
    DEFAULT_DISCOVERY_PORT,
    MAX_ANNOUNCEMENT_BYTES,
    BroadcastListener,
    BroadcastSender,
    NodeAnnouncement,
)
from ainode.discovery.cluster import ClusterState
from ainode.discovery.signing import (
    ACCEPT,
    BAD_SIGNATURE,
    SIGNATURE_FIELD,
    UNSIGNED,
    ClusterSecret,
    canonical_bytes,
    rejection,
    seal,
    sign,
)

SECRET = "a-fleet-secret-nobody-has"


def _announcement(**overrides) -> NodeAnnouncement:
    defaults = dict(
        node_id="spark-2", node_name="Spark-2", gpu_name="NVIDIA GB10",
        gpu_memory_gb=128.0, unified_memory=True,
        model="ornith-ai/Ornith-1.5-35B-A3B-NVFP4", status="serving",
        api_port=8000, web_port=3000, timestamp=1700000000.0,
        cluster_id="titanium", ainode_version="0.5.27",
    )
    defaults.update(overrides)
    return NodeAnnouncement(**defaults)


def _signed(ann: NodeAnnouncement, secret: Optional[str] = SECRET) -> bytes:
    return json.dumps(seal(asdict(ann), secret)).encode()


def _forged(ann: NodeAnnouncement, secret: str = SECRET, **edits) -> bytes:
    """Sign an announcement, then edit it. What an attacker can actually do."""
    payload = json.loads(_signed(ann, secret).decode())
    payload.update(edits)
    return json.dumps(payload).encode()


class FakeSocket:
    """One datagram socket, both directions, no network.

    Reads drain a queue and then raise BlockingIOError, which is exactly what a
    non-blocking socket with nothing to read does, so the receive loop takes its
    real "nothing arrived" path.
    """

    def __init__(self, inbound: Optional[List[Tuple[bytes, str]]] = None):
        self.inbound = list(inbound or [])
        self.reads = 0
        self.sent: List[Tuple[bytes, tuple]] = []
        self.bound: Optional[tuple] = None
        self.options: List[tuple] = []
        self.closed = False

    # -- the parts the loops use -------------------------------------------
    def setsockopt(self, level, option, value):
        self.options.append((level, option, value))

    def setblocking(self, flag):
        self.blocking = flag

    def bind(self, address):
        self.bound = address

    def recvfrom(self, size):
        if not self.inbound:
            raise BlockingIOError("nothing to read")
        data, peer_ip = self.inbound.pop(0)
        self.reads += 1
        return data[:size], (peer_ip, DEFAULT_DISCOVERY_PORT)

    def sendto(self, data, address):
        self.sent.append((data, address))
        return len(data)

    def close(self):
        self.closed = True


@pytest.fixture
def fake_socket(monkeypatch):
    """Hand the next socket() call in discovery a fake, and keep hold of it."""
    created: List[FakeSocket] = []

    def factory(*args, **kwargs):
        sock = created[0] if created else FakeSocket()
        if not created:
            created.append(sock)
        return sock

    def install(sock: FakeSocket) -> FakeSocket:
        created.clear()
        created.append(sock)
        monkeypatch.setattr(bc.socket, "socket", factory)
        return sock

    return install


async def _run_listener(listener: BroadcastListener, sock: FakeSocket,
                        expected_reads: int, timeout: float = 2.0) -> None:
    """Start the real receive loop, let it consume the queue, stop it."""
    await listener.start()
    deadline = time.monotonic() + timeout
    while sock.reads < expected_reads and time.monotonic() < deadline:
        await asyncio.sleep(0.01)
    await asyncio.sleep(0.02)
    await listener.stop()


# --------------------------------------------------------------- the digest --


class TestSignature:
    def test_the_signature_covers_the_payload_and_not_itself(self):
        payload = asdict(_announcement())
        sealed = seal(payload, SECRET)
        assert sealed[SIGNATURE_FIELD] == sign(payload, SECRET)
        assert SIGNATURE_FIELD not in json.loads(canonical_bytes(sealed))

    def test_canonical_bytes_do_not_depend_on_key_order(self):
        payload = asdict(_announcement())
        shuffled = dict(reversed(list(payload.items())))
        assert canonical_bytes(payload) == canonical_bytes(shuffled)

    def test_a_round_trip_through_json_still_verifies(self):
        """The receiver verifies the mapping it parsed, not one it rebuilt: float
        and unicode spelling have to survive dumps/loads on both sides."""
        ann = _announcement(gpu_memory_gb=127.5, node_name="Spark-2 (läb)",
                            gpu_utilization=33.333333)
        wire = json.loads(_signed(ann).decode())
        assert rejection(wire, SECRET) == ACCEPT

    def test_a_different_secret_does_not_verify(self):
        wire = json.loads(_signed(_announcement()).decode())
        assert rejection(wire, "some-other-secret") == BAD_SIGNATURE

    def test_every_edited_field_invalidates_it(self):
        for edit in ({"role": "master"}, {"node_id": "aaa"}, {"api_port": 9999},
                     {"cluster_id": "other"}, {"is_master": True},
                     {"instances": [{"model": "evil/model", "api_port": 8000}]}):
            wire = json.loads(_forged(_announcement(), **edit).decode())
            assert rejection(wire, SECRET) == BAD_SIGNATURE, edit

    def test_a_node_with_no_secret_judges_nothing(self):
        wire = json.loads(_announcement().to_json())
        assert rejection(wire, None) == ACCEPT
        assert rejection(wire, "") == ACCEPT

    def test_an_unsigned_datagram_is_named_as_such(self):
        wire = json.loads(_announcement().to_json())
        assert rejection(wire, SECRET) == UNSIGNED

    def test_a_signed_announcement_still_parses_on_a_node_that_knows_nothing_of_it(self):
        """The migration guarantee: an older peer drops the unknown key and stays
        in the cluster view. A wrapped envelope would have vanished it."""
        restored = NodeAnnouncement.from_json(_signed(_announcement()).decode())
        assert restored.node_id == "spark-2"
        assert restored.model.endswith("NVFP4")
        assert not hasattr(restored, SIGNATURE_FIELD)


# ------------------------------------------------------------- the receiver --


class TestListenerAcceptsAndRefuses:
    @pytest.mark.asyncio
    async def test_a_signed_announcement_from_a_peer_lands(self, fake_socket):
        sock = fake_socket(FakeSocket([(_signed(_announcement()), "10.0.0.2")]))
        listener = BroadcastListener(local_node_id="spark-1",
                                     secret_provider=lambda: SECRET)
        await _run_listener(listener, sock, expected_reads=1)

        assert sock.bound == ("", DEFAULT_DISCOVERY_PORT)
        assert "spark-2" in listener.registry
        node = listener.registry["spark-2"]
        assert node.peer_ip == "10.0.0.2"
        assert node.announcement.ainode_version == "0.5.27"
        assert listener.dropped == {}

    @pytest.mark.asyncio
    async def test_an_unsigned_announcement_is_dropped_by_a_node_with_a_secret(
            self, fake_socket):
        sock = fake_socket(FakeSocket([
            (_announcement().to_json().encode(), "10.0.0.3")]))
        listener = BroadcastListener(local_node_id="spark-1",
                                     secret_provider=lambda: SECRET)
        await _run_listener(listener, sock, expected_reads=1)

        assert listener.registry == {}
        assert listener.dropped == {UNSIGNED: 1}

    @pytest.mark.asyncio
    async def test_a_forged_master_never_reaches_the_election(self, fake_socket):
        """The attack in #169, end to end: read the cluster id over HTTP, then
        announce a low node_id with role master and own the fleet."""
        hostile = _forged(_announcement(node_id="aaaa-attacker"),
                          role="master", is_master=True)
        sock = fake_socket(FakeSocket([(hostile, "10.0.0.66")]))
        local = _announcement(node_id="spark-1", node_name="Spark-1",
                              cluster_id="titanium")
        listener = BroadcastListener(local_node_id="spark-1",
                                     secret_provider=lambda: SECRET)
        await _run_listener(listener, sock, expected_reads=1)

        cluster = ClusterState(local_announcement=local)
        cluster.update_from_discovered(listener.registry)

        assert listener.dropped == {BAD_SIGNATURE: 1}
        assert cluster.get_master().node_id == "spark-1"
        assert [n.node_id for n in cluster.get_nodes()] == ["spark-1"]

    @pytest.mark.asyncio
    async def test_the_same_fleet_with_no_secret_behaves_exactly_as_before(
            self, fake_socket, caplog):
        """Nothing on the fleet sets a secret today. A release that made signing
        mandatory would partition every existing cluster on upgrade."""
        sock = fake_socket(FakeSocket([
            (_announcement().to_json().encode(), "10.0.0.2")]))
        listener = BroadcastListener(local_node_id="spark-1")
        with caplog.at_level(logging.WARNING, logger="ainode.discovery.broadcast"):
            await _run_listener(listener, sock, expected_reads=1)

        assert "spark-2" in listener.registry
        assert listener.dropped == {}
        assert sum("UNAUTHENTICATED" in r.message for r in caplog.records) == 1

    @pytest.mark.asyncio
    async def test_a_signed_peer_is_accepted_by_a_fleet_with_no_secret(self, fake_socket):
        """The roll order does not matter: signing early is harmless."""
        sock = fake_socket(FakeSocket([(_signed(_announcement()), "10.0.0.2")]))
        listener = BroadcastListener(local_node_id="spark-1")
        await _run_listener(listener, sock, expected_reads=1)
        assert "spark-2" in listener.registry

    def test_the_unauthenticated_warning_is_said_once_not_per_datagram(self, caplog):
        listener = BroadcastListener(local_node_id="spark-1")
        data = _announcement().to_json().encode()
        with caplog.at_level(logging.WARNING, logger="ainode.discovery.broadcast"):
            for _ in range(5):
                listener.handle_datagram(data, "10.0.0.2")
        assert sum("UNAUTHENTICATED" in r.message for r in caplog.records) == 1

    def test_a_flood_of_forgeries_is_reported_once_per_source(self, caplog):
        listener = BroadcastListener(local_node_id="spark-1",
                                     secret_provider=lambda: SECRET)
        hostile = _forged(_announcement(), role="master")
        with caplog.at_level(logging.WARNING, logger="ainode.discovery.broadcast"):
            for _ in range(50):
                listener.handle_datagram(hostile, "10.0.0.66")
            listener.handle_datagram(hostile, "10.0.0.67")
        named = [r for r in caplog.records if "dropped a discovery" in r.message]
        assert len(named) == 2
        assert listener.dropped == {BAD_SIGNATURE: 51}

    def test_the_source_report_is_capped_so_a_spoofer_cannot_fill_memory(self):
        listener = BroadcastListener(local_node_id="spark-1",
                                     secret_provider=lambda: SECRET)
        hostile = _forged(_announcement(), role="master")
        for i in range(BroadcastListener.MAX_REPORTED_SOURCES + 25):
            listener.handle_datagram(hostile, f"10.0.9.{i}")
        assert len(listener._reported) <= BroadcastListener.MAX_REPORTED_SOURCES

    def test_rubbish_on_the_port_is_dropped_without_raising(self):
        listener = BroadcastListener(local_node_id="spark-1",
                                     secret_provider=lambda: SECRET)
        assert listener.handle_datagram(b"not json at all") == "unparseable"
        assert listener.handle_datagram(b'"a string"') == "unparseable"
        assert listener.handle_datagram(b'{"node_id": "x"}') == UNSIGNED
        assert listener.registry == {}

    def test_a_secret_provider_that_raises_does_not_take_the_cluster_down(self):
        def broken():
            raise RuntimeError("config.json is on fire")

        listener = BroadcastListener(local_node_id="spark-1", secret_provider=broken)
        assert listener.handle_datagram(
            _announcement().to_json().encode(), "10.0.0.2") == ACCEPT
        assert "spark-2" in listener.registry


# --------------------------------------------------------------- the sender --


class TestSenderSigns:
    def test_the_datagram_is_signed_with_the_current_secret(self):
        sender = BroadcastSender(_announcement(node_id="spark-1"),
                                 secret_provider=lambda: SECRET)
        payload = json.loads(sender.datagram().decode())
        assert rejection(payload, SECRET) == ACCEPT

    def test_with_no_secret_it_sends_what_it_always_sent(self):
        ann = _announcement(node_id="spark-1")
        sender = BroadcastSender(ann)
        assert json.loads(sender.datagram().decode()) == json.loads(ann.to_json())

    def test_a_broken_provider_sends_unsigned_and_says_so_once(self, caplog):
        def broken():
            raise RuntimeError("no key here")

        sender = BroadcastSender(_announcement(), secret_provider=broken)
        with caplog.at_level(logging.WARNING, logger="ainode.discovery.broadcast"):
            for _ in range(3):
                sender.datagram()
        assert sum("UNSIGNED" in r.message for r in caplog.records) == 1

    @pytest.mark.asyncio
    async def test_rotating_the_secret_needs_no_restart(self, fake_socket, tmp_path):
        """The operational rule. A fleet-wide restart to change a key is a key
        nobody will ever change, so the secret is read per send: write the file,
        and the next tick is signed with the new value."""
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({"cluster_secret": "first-key"}))
        secret = ClusterSecret(path=config_file)

        sock = fake_socket(FakeSocket())
        sender = BroadcastSender(_announcement(node_id="spark-1"),
                                 broadcast_interval=0.01, secret_provider=secret)
        await sender.start()
        while not sock.sent:
            await asyncio.sleep(0.005)
        first = json.loads(sock.sent[-1][0].decode())

        config_file.write_text(json.dumps({"cluster_secret": "second-key"}))
        sent_before = len(sock.sent)
        while len(sock.sent) <= sent_before + 1:
            await asyncio.sleep(0.005)
        await sender.stop()
        second = json.loads(sock.sent[-1][0].decode())

        assert rejection(first, "first-key") == ACCEPT
        assert rejection(second, "second-key") == ACCEPT
        assert rejection(second, "first-key") == BAD_SIGNATURE
        # Broadcast, on the port everything else agrees on.
        assert sock.sent[-1][1] == ("<broadcast>", DEFAULT_DISCOVERY_PORT)

    @pytest.mark.asyncio
    async def test_the_sender_keeps_stamping_live_telemetry_while_signing(
            self, fake_socket):
        """The signature is an envelope: everything the announcement already
        carried still gets refreshed per tick."""
        sock = fake_socket(FakeSocket())
        sender = BroadcastSender(
            _announcement(node_id="spark-1"), broadcast_interval=0.01,
            metrics_provider=lambda: {"memory_used_mb": 41000, "memory_total_mb": 131072,
                                      "utilization_percent": 77, "temperature_c": 61},
            secret_provider=lambda: SECRET)
        await sender.start()
        while not sock.sent:
            await asyncio.sleep(0.005)
        await sender.stop()

        payload = json.loads(sock.sent[-1][0].decode())
        assert payload["gpu_memory_used_mb"] == 41000
        assert payload["gpu_utilization"] == 77
        assert payload["timestamp"] > 1700000000.0
        assert rejection(payload, SECRET) == ACCEPT


# ----------------------------------------------------------- the secret file --


class TestClusterSecret:
    def test_it_reads_the_config_file(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text(json.dumps({"cluster_secret": "from-disk"}))
        assert ClusterSecret(path=path)() == "from-disk"

    def test_a_change_is_picked_up_and_a_removal_too(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text(json.dumps({"cluster_secret": "one"}))
        secret = ClusterSecret(path=path)
        assert secret() == "one"
        path.write_text(json.dumps({"cluster_secret": "two"}))
        assert secret() == "two"
        path.write_text(json.dumps({"cluster_enabled": True}))
        assert secret() is None

    def test_the_in_memory_config_answers_when_there_is_no_file(self, tmp_path):
        class Config:
            cluster_secret = "in-memory"

        secret = ClusterSecret(Config(), path=tmp_path / "missing.json")
        assert secret() == "in-memory"

    def test_a_half_written_config_keeps_the_last_good_value(self, tmp_path):
        """A broken save must not take a fleet's authentication down with it."""
        path = tmp_path / "config.json"
        path.write_text(json.dumps({"cluster_secret": "good"}))
        secret = ClusterSecret(path=path)
        assert secret() == "good"
        path.write_text('{"cluster_secret": "hal')
        assert secret() == "good"

    def test_no_file_and_no_config_is_simply_no_secret(self, tmp_path):
        assert ClusterSecret(path=tmp_path / "nope.json")() is None


# ------------------------------------------------------------- the size cap --


def test_a_fully_populated_signed_announcement_still_fits_the_listener():
    """MAX_ANNOUNCEMENT_BYTES is a hard ceiling: the listener reads ONE datagram
    of that size, so a byte over it is truncated on arrival, fails to parse, and
    the node disappears from every peer's cluster view with nothing logged. The
    signature and the version field are the two newest tenants."""
    ann = NodeAnnouncement(
        node_id="spark1-head", node_name="Spark-1-DGX", gpu_name="NVIDIA GB10",
        gpu_memory_gb=128.0, unified_memory=True,
        model="unsloth/Qwen3.8-27B-NVFP4", status="starting",
        api_port=8000, web_port=3000, fabric_ip="10.100.0.1",
        distributed_mode="head", distributed_instance_id="spark1-head:big-moe",
        distributed_peers=["10.100.0.2", "10.100.0.3", "10.100.0.4"],
        instances=[{"instance_id": f"spark1-head:model-{i}",
                    "model": "ornith-ai/Ornith-1.5-35B-A3B-NVFP4",
                    "head_node_id": "spark1-head", "peer_ips": ["10.100.0.2"],
                    "api_port": 8000 + i, "tensor_parallel_size": 2,
                    "status": "serving", "distributed_executor": "mp"}
                   for i in range(4)],
        load_phase="loading_weights", load_started_at=time.time(),
        load_elapsed_seconds=195.3, expected_ready_minutes=12.0,
        ainode_version="0.5.27",
    )
    sender = BroadcastSender(ann, secret_provider=lambda: SECRET)
    signed = sender.datagram()
    unsigned = ann.to_json().encode()

    assert len(signed) < MAX_ANNOUNCEMENT_BYTES
    # What the envelope costs: 64 hex characters plus the key and its quotes.
    assert len(signed) - len(unsigned) < 100
    # And the headroom that is left for the next field somebody adds.
    assert MAX_ANNOUNCEMENT_BYTES - len(signed) > 2000
