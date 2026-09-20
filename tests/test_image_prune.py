"""What `ainode update` may delete, decided without a docker daemon in sight.

Issue #184: the update pulled a release every time and removed nothing, so a node
carried every release it had ever run, under each of the three names the image is
mirrored as. Spark-1: 180 images, 226 GB reclaimable, root filesystem at 82
percent, on a box whose whole job is holding 35 to 400 GB of model weights.

The decision is a pure function of a `docker images` listing, which is what makes
these tests possible and also what makes the dry run honest: the same function
can be pointed at a listing copied off another node.

The fake docker below is the whole daemon as far as this module is concerned: it
answers `docker images` from a canned listing, records every `docker rmi`, and
can refuse one the way a real daemon refuses an image a container is using.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import pytest

from ainode.core.image_prune import (
    AINODE_IMAGE_REPOS,
    DOCKER_IMAGES_FORMAT,
    ImageRow,
    format_plan,
    human_size,
    parse_images,
    parse_size,
    plan_prune,
    prune_images,
    release_key,
    running_image,
)

GHCR = AINODE_IMAGE_REPOS[0]


class FakeDocker:
    """A docker that lists what it is told and remembers what it was asked to do."""

    def __init__(self, listing: str = "", refuse: Sequence[str] = ()):
        self.listing = listing
        self.refuse = set(refuse)
        self.calls: List[List[str]] = []

    def __call__(self, argv: Sequence[str]) -> Tuple[int, str, str]:
        argv = list(argv)
        self.calls.append(argv)
        if argv[:2] == ["docker", "images"]:
            return 0, self.listing, ""
        if argv[:2] == ["docker", "rmi"]:
            ref = argv[2]
            if ref in self.refuse:
                return 1, "", (f"Error response from daemon: conflict: unable to "
                               f"remove repository reference {ref} (must force)")
            return 0, f"Untagged: {ref}\n", ""
        return 0, "", ""

    @property
    def removed(self) -> List[str]:
        return [c[2] for c in self.calls if c[:2] == ["docker", "rmi"]]


def _listing(*rows: Tuple[str, str, str, str]) -> str:
    return "\n".join("\t".join(row) for row in rows) + "\n"


# A node mid-history: six releases of the app image, the newest two mirrored
# under all three names, plus things that are not ours at all.
FLEET_LISTING = _listing(
    (GHCR, "0.5.26", "aaa111", "1.52GB"),
    (GHCR, "0.5.25", "bbb222", "1.51GB"),
    (GHCR, "0.5.24", "ccc333", "1.5GB"),
    (GHCR, "0.5.23", "ddd444", "1.49GB"),
    (GHCR, "0.4.9", "eee555", "1.2GB"),
    (GHCR, "latest", "aaa111", "1.52GB"),
    ("argentos/ainode", "0.5.26", "aaa111", "1.52GB"),
    ("argentos/ainode", "0.5.24", "ccc333", "1.5GB"),
    ("ainode", "dev", "fff666", "1.6GB"),
    ("ainode-base", "cu13", "999aaa", "24.1GB"),
    ("vllm/vllm-openai", "v0.27.1", "888bbb", "18.4GB"),
    ("<none>", "<none>", "777ccc", "2.1GB"),
)


class TestParsing:
    def test_the_format_is_the_one_the_parser_reads(self):
        """A change to either without the other is a silent no-op prune."""
        assert DOCKER_IMAGES_FORMAT.count("\t") == 3
        assert DOCKER_IMAGES_FORMAT.startswith("{{.Repository}}")

    def test_rows_header_and_untagged(self):
        rows = parse_images(
            "REPOSITORY\tTAG\tIMAGE ID\tSIZE\n"
            f"{GHCR}\t0.5.26\taaa111\t1.52GB\n"
            "<none>\t<none>\tbbb222\t2.1GB\n"
            "\n"
        )
        assert [r.ref for r in rows] == [f"{GHCR}:0.5.26"]
        assert rows[0].image_id == "aaa111"
        assert rows[0].size_bytes == 1_520_000_000

    @pytest.mark.parametrize("text,expected", [
        ("0B", 0),
        ("392MB", 392_000_000),
        ("1.52GB", 1_520_000_000),
        ("24.1GB", 24_100_000_000),
        ("1.5 kB", 1_500),
        ("2GiB", 2 * 1024 ** 3),
        ("", 0),
        ("who knows", 0),
    ])
    def test_sizes(self, text, expected):
        assert parse_size(text) == expected

    @pytest.mark.parametrize("tag,expected", [
        ("0.5.26", (0, 5, 26)),
        ("1.0.0", (1, 0, 0)),
        ("latest", None),
        ("0.5", None),
        ("0.5.26-rc1", None),
        ("sha-abc123", None),
    ])
    def test_release_tags(self, tag, expected):
        assert release_key(tag) == expected

    def test_human_size_reads_like_dockers_own(self):
        assert human_size(1_520_000_000) == "1.5 GB"
        assert human_size(0) == "0 B"


class TestPlan:
    def test_one_rollback_generation_by_default(self):
        plan = plan_prune(parse_images(FLEET_LISTING), f"{GHCR}:0.5.26", keep=1)

        assert not plan.refused
        assert sorted(d.ref for d in plan.removals) == [
            "argentos/ainode:0.5.24",
            f"{GHCR}:0.4.9",
            f"{GHCR}:0.5.23",
            f"{GHCR}:0.5.24",
        ]
        kept = {d.ref: d.reason for d in plan.kept}
        assert kept[f"{GHCR}:0.5.26"] == "the running release"
        assert kept[f"{GHCR}:0.5.25"] == "rollback generation 1 of 1"
        assert kept[f"{GHCR}:latest"] == "not a release tag"
        assert kept["ainode:dev"] == "not a release tag"

    def test_keeping_zero_takes_the_rollback_generation_too(self):
        plan = plan_prune(parse_images(FLEET_LISTING), f"{GHCR}:0.5.26", keep=0)
        assert f"{GHCR}:0.5.25" in [d.ref for d in plan.removals]
        assert f"{GHCR}:0.5.26" in [d.ref for d in plan.kept]

    def test_keeping_two_keeps_two(self):
        plan = plan_prune(parse_images(FLEET_LISTING), f"{GHCR}:0.5.26", keep=2)
        kept = {d.ref: d.reason for d in plan.kept}
        assert kept[f"{GHCR}:0.5.25"] == "rollback generation 1 of 2"
        assert kept[f"{GHCR}:0.5.24"] == "rollback generation 2 of 2"
        # A kept generation keeps every name it has: the mirrored tag holds the
        # same image, so dropping it would free nothing and lose the rollback.
        assert kept["argentos/ainode:0.5.24"] == "rollback generation 2 of 2"

    def test_only_ainodes_own_repositories_are_ever_considered(self):
        """A node's engine images are what it serves models with, and ainode-base
        is 24 GB of somebody else's build. Neither is an update's business."""
        plan = plan_prune(parse_images(FLEET_LISTING), f"{GHCR}:0.5.26")
        touched = {d.ref for d in plan.decisions}
        assert "ainode-base:cu13" not in touched
        assert "vllm/vllm-openai:v0.27.1" not in touched

    def test_the_running_image_is_kept_under_every_name_it_has(self):
        """One image, three tags, one id. `docker rmi <id>` would take all three,
        which is why removals are always by reference."""
        plan = plan_prune(parse_images(FLEET_LISTING), f"{GHCR}:0.5.26")
        kept = {d.ref: d.reason for d in plan.kept}
        assert kept["argentos/ainode:0.5.26"] == (
            "the running release, under another name")
        assert all(d.image_id != "aaa111" for d in plan.removals)

    def test_a_newer_release_somebody_pre_pulled_is_left_alone(self):
        listing = _listing(
            (GHCR, "0.6.0", "new111", "1.6GB"),
            (GHCR, "0.5.26", "aaa111", "1.52GB"),
            (GHCR, "0.5.20", "old222", "1.4GB"),
        )
        plan = plan_prune(parse_images(listing), f"{GHCR}:0.5.26")
        kept = {d.ref: d.reason for d in plan.kept}
        assert kept[f"{GHCR}:0.6.0"] == "newer than the running release"
        assert [d.ref for d in plan.removals] == []  # 0.5.20 is the one rollback

    def test_nothing_is_decided_when_the_new_image_is_not_on_the_host(self):
        """The baseline has to be real: pruning against a version that never
        landed is how a node loses the image it is running."""
        plan = plan_prune(parse_images(FLEET_LISTING), f"{GHCR}:9.9.9")
        assert plan.removals == []
        assert "not in the image listing" in plan.refused
        assert "Not pruning" in format_plan(plan)

    def test_nothing_is_decided_for_a_floating_tag(self):
        plan = plan_prune(parse_images(FLEET_LISTING), f"{GHCR}:latest")
        assert plan.removals == []
        assert "does not name a release tag" in plan.refused

    def test_the_freed_figure_is_reported_as_an_upper_bound(self):
        plan = plan_prune(parse_images(FLEET_LISTING), f"{GHCR}:0.5.26")
        # Four tags over three distinct images: the mirrored 0.5.24 is one image.
        assert len(plan.removals) == 4
        assert plan.distinct_images_removed == 3
        text = format_plan(plan)
        assert "4 tag(s), 3 distinct image(s)" in text
        assert "upper bound" in text

    def test_the_plan_explains_every_kept_image_when_asked(self):
        plan = plan_prune(parse_images(FLEET_LISTING), f"{GHCR}:0.5.26")
        text = format_plan(plan, verbose=True)
        for decision in plan.kept:
            assert decision.ref in text


class TestExecution:
    def test_removals_go_by_tag_one_docker_rmi_each(self):
        docker = FakeDocker(FLEET_LISTING)
        plan, log = prune_images(f"{GHCR}:0.5.26", keep=1, runner=docker)

        assert docker.calls[0] == ["docker", "images", "--format", DOCKER_IMAGES_FORMAT]
        assert sorted(docker.removed) == [
            "argentos/ainode:0.5.24",
            f"{GHCR}:0.4.9",
            f"{GHCR}:0.5.23",
            f"{GHCR}:0.5.24",
        ]
        assert len(log) == 4
        assert all(line.startswith("removed ") for line in log)
        assert plan.freed_bytes_upper_bound > 0

    def test_a_dry_run_asks_docker_for_nothing_but_the_listing(self):
        docker = FakeDocker(FLEET_LISTING)
        plan, log = prune_images(f"{GHCR}:0.5.26", keep=1, dry_run=True, runner=docker)

        assert docker.removed == []
        assert log == []
        assert len(plan.removals) == 4, "the plan is still the plan"

    def test_an_image_docker_refuses_is_reported_and_not_fatal(self):
        """An image a container still uses. The rest of the plan still applies and
        the next update tries again."""
        docker = FakeDocker(FLEET_LISTING, refuse=[f"{GHCR}:0.5.24"])
        plan, log = prune_images(f"{GHCR}:0.5.26", keep=1, runner=docker)

        assert len(docker.removed) == 4, "it tried all four"
        assert len(plan.removals) == 3, "and counts three"
        kept = {d.ref: d.reason for d in plan.kept}
        assert "docker refused" in kept[f"{GHCR}:0.5.24"]
        assert any("kept" in line for line in log)

    def test_a_listing_can_be_replayed_from_anywhere(self):
        """The read-only dry run: a listing copied off another node decides
        nothing on this one, and needs no docker at all."""
        rows = parse_images(FLEET_LISTING)
        docker = FakeDocker("")
        plan, log = prune_images(f"{GHCR}:0.5.26", keep=1, dry_run=True,
                                 rows=rows, runner=docker)
        assert docker.calls == []
        assert len(plan.removals) == 4

    def test_a_docker_that_cannot_list_is_an_error_not_an_empty_plan(self):
        def broken(argv):
            return 1, "", "Cannot connect to the Docker daemon"

        with pytest.raises(RuntimeError, match="docker images failed"):
            prune_images(f"{GHCR}:0.5.26", runner=broken)


class TestRunningImage:
    def test_image_env_wins_because_it_is_what_the_unit_reads(self, tmp_path, monkeypatch):
        (tmp_path / "image.env").write_text(f"AINODE_IMAGE={GHCR}:0.5.26\n")
        monkeypatch.setenv("AINODE_HOME", str(tmp_path))
        monkeypatch.setenv("AINODE_IMAGE", f"{GHCR}:0.0.1")
        assert running_image("0.5.27") == f"{GHCR}:0.5.26"

    def test_then_the_env_then_the_running_version(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AINODE_HOME", str(tmp_path))
        monkeypatch.setenv("AINODE_IMAGE", f"{GHCR}:0.0.1")
        assert running_image("0.5.27") == f"{GHCR}:0.0.1"

        monkeypatch.delenv("AINODE_IMAGE")
        assert running_image("0.5.27") == f"{GHCR}:0.5.27"
        assert running_image("") == ""


def test_an_imagerow_knows_its_own_reference_and_size():
    row = ImageRow(repository=GHCR, tag="0.5.26", image_id="aaa111", size="1.52GB")
    assert row.ref == f"{GHCR}:0.5.26"
    assert row.size_bytes == 1_520_000_000
