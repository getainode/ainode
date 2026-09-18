"""The labeled item set: 110 typed decisions with one checkable answer each.

An item is a state, a typed question about it, and the label a person would give.
Five sets ask five different shapes of the same thing:

    ``route``     which engine class should serve a request (choice, 4 options)
    ``triage``    which support team owns a ticket (choice, 4 options)
    ``urgency``   does this ticket have to be handled within the hour (yes/no)
    ``pr_safe``   is this diff safe to merge without review (yes/no)
    ``fact``      is this statement true (yes/no)

The set lives in ``bench/decide/items.json`` as repo data rather than package data,
the same way the harness bench's tasks do: it is versioned next to the results it
produces, so a record's numbers can be read against the exact items that produced
them.

Nothing here talks to a model or a node. Loading is strict on purpose: a bench that
ran 104 of 110 items because six of them were malformed would report an accuracy
against an item count nobody chose, so a bad item is a load error and not a skip.
"""
from __future__ import annotations

import json
import os
import pathlib
from dataclasses import dataclass

#: Item kinds. ``choice`` picks one named option; ``noul`` is TypeSafe's name for a
#: yes/no question answered with the probability of yes, and the local backends map
#: it onto a boolean question.
CHOICE = "choice"
NOUL = "noul"
KINDS = (CHOICE, NOUL)

#: The two option keys a ``noul`` item is scored over, so a yes/no answer has a
#: distribution with names rather than a bare float.
TRUE = "true"
FALSE = "false"

ENV_ITEMS = "AINODE_DECIDE_ITEMS"
_REPO_ITEMS = (pathlib.Path(__file__).resolve().parents[3] / "bench" / "decide"
               / "items.json")
_HOME_ITEMS = pathlib.Path.home() / ".ainode" / "bench" / "decide" / "items.json"

REQUIRED_KEYS = ("id", "set", "kind", "state", "question", "label")


class ItemError(RuntimeError):
    """An item file that cannot be loaded, or a ``--sets`` selection that is empty."""


def items_file_label(path) -> str:
    """How a record names the item file: repo-relative, or just its basename.

    A record is committed and read by other people, so it says
    ``bench/decide/items.json`` rather than whichever absolute path the run happened
    to be started from.
    """
    path = pathlib.Path(path)
    try:
        return str(path.resolve().relative_to(_REPO_ITEMS.parents[2]))
    except ValueError:
        return path.name


def default_items_path() -> pathlib.Path:
    """``$AINODE_DECIDE_ITEMS``, else the repo's set, else the installed copy."""
    override = os.environ.get(ENV_ITEMS)
    if override:
        return pathlib.Path(override).expanduser()
    if _REPO_ITEMS.is_file():
        return _REPO_ITEMS
    return _HOME_ITEMS


@dataclass(frozen=True)
class Item:
    """One labeled decision, exactly as the file states it."""

    id: str
    set: str
    kind: str
    state: str
    question: str
    label: object
    criteria: dict | None = None

    @property
    def options(self) -> list:
        """The option names this item is scored over, in file order.

        A ``noul`` item's options are ``true``/``false`` so that every item has a
        distribution over named options and one metric fits both kinds.
        """
        if self.kind == CHOICE:
            return list((self.criteria or {}).keys())
        return [TRUE, FALSE]

    @property
    def label_option(self) -> str:
        """The distribution key the label sits under: the option, or true/false."""
        return self.option_for(self.label)

    def option_for(self, answer) -> str:
        """The distribution key one answer sits under, in this item's answer space."""
        if self.kind == CHOICE:
            return str(answer)
        return TRUE if answer else FALSE

    def description(self, option: str) -> str | None:
        return (self.criteria or {}).get(option)


@dataclass(frozen=True)
class ItemSet:
    """The file's own header plus the items that survived validation."""

    id: str
    version: int
    description: str
    sets: dict
    items: tuple
    path: pathlib.Path

    @property
    def set_names(self) -> list:
        """Set names in the order the file declares them, not the items' order."""
        return [name for name in self.sets if any(i.set == name for i in self.items)]

    def as_json(self) -> dict:
        """The ``item_set`` sub-block of a record: what ran, not the items again."""
        return {"id": self.id, "version": self.version, "file": self.path.name,
                "count": len(self.items),
                "sets": {name: sum(1 for i in self.items if i.set == name)
                         for name in self.set_names}}


def _require(condition, message: str) -> None:
    if not condition:
        raise ItemError(message)


def validate_document(doc, path=None) -> list:
    """Every item in ``doc``, or ``ItemError`` naming the first thing that is wrong.

    The rules are the ones a reader of a result has to be able to assume:

    * every item carries ``id``, ``set``, ``kind``, ``state``, ``question`` and a
      ``label``, and the ids are unique;
    * a ``choice`` item carries ``criteria`` with at least two options and its label
      is one of them, so the label is always inside the answer space;
    * a ``noul`` item's label is a real boolean, never the string "true";
    * every item of one set shares that set's kind, question and criteria, because a
      set is one measurement and a set whose items ask slightly different questions
      is five measurements with one accuracy printed over them;
    * a set the header declares with a ``count`` has exactly that many items.
    """
    where = f" in {path}" if path else ""
    _require(isinstance(doc, dict), f"the item file{where} is not a JSON object")
    declared = doc.get("sets")
    _require(isinstance(declared, dict) and declared,
             f"the item file{where} declares no `sets` header")
    raw_items = doc.get("items")
    _require(isinstance(raw_items, list) and raw_items,
             f"the item file{where} holds no `items`")

    items = []
    seen = set()
    for index, raw in enumerate(raw_items):
        _require(isinstance(raw, dict), f"item {index} is not an object")
        missing = [k for k in REQUIRED_KEYS if raw.get(k) in (None, "")]
        # `label` false is a real label, so only a missing key counts as missing.
        missing = [k for k in missing if k != "label" or "label" not in raw]
        _require(not missing,
                 f"item {raw.get('id', index)} is missing {', '.join(missing)}")
        item_id = str(raw["id"])
        _require(item_id not in seen, f"item id {item_id} appears twice")
        seen.add(item_id)
        name = raw["set"]
        _require(name in declared,
                 f"item {item_id} is in set {name!r}, which the header does not "
                 f"declare (declared: {', '.join(declared)})")
        kind = raw["kind"]
        _require(kind in KINDS,
                 f"item {item_id} has kind {kind!r}; pick from {', '.join(KINDS)}")
        criteria = raw.get("criteria")
        if kind == CHOICE:
            _require(isinstance(criteria, dict) and len(criteria) >= 2,
                     f"choice item {item_id} needs `criteria` with at least two "
                     "options")
            for option, text in criteria.items():
                _require(text is None or isinstance(text, str),
                         f"item {item_id} option {option!r} has a description that "
                         "is neither a string nor null")
            _require(raw["label"] in criteria,
                     f"item {item_id} is labeled {raw['label']!r}, which is not one "
                     "of its options")
        else:
            _require(isinstance(raw["label"], bool),
                     f"noul item {item_id} is labeled {raw['label']!r}; a yes/no "
                     "item's label is true or false")
            if criteria is not None:
                _require(isinstance(criteria, dict)
                         and set(criteria) == {TRUE, FALSE},
                         f"noul item {item_id} carries `criteria` with keys other "
                         f"than {TRUE}/{FALSE}")
        items.append(Item(id=item_id, set=name, kind=kind, state=raw["state"],
                          question=raw["question"], label=raw["label"],
                          criteria=criteria))

    for name, header in declared.items():
        mine = [i for i in items if i.set == name]
        if not mine:
            continue
        first = mine[0]
        if isinstance(header, dict) and header.get("kind"):
            _require(header["kind"] == first.kind,
                     f"set {name} declares kind {header['kind']!r} but its items "
                     f"are {first.kind!r}")
        if isinstance(header, dict) and header.get("count") is not None:
            _require(header["count"] == len(mine),
                     f"set {name} declares {header['count']} items and holds "
                     f"{len(mine)}")
        for item in mine[1:]:
            _require(item.kind == first.kind,
                     f"set {name} mixes kinds: {first.id} is {first.kind} and "
                     f"{item.id} is {item.kind}")
            _require(item.question == first.question,
                     f"set {name} asks two different questions: {first.id} and "
                     f"{item.id}")
            _require(item.criteria == first.criteria,
                     f"set {name} gives {item.id} different options from "
                     f"{first.id}")
    return items


def load_items(path=None, sets=None) -> ItemSet:
    """Load and validate the item file, optionally narrowed to ``sets``.

    ``sets`` is a list of set names; an unknown name is an error rather than an
    empty selection, because a typo that silently measured nothing would publish an
    accuracy over the wrong items.
    """
    path = pathlib.Path(path) if path else default_items_path()
    if not path.is_file():
        raise ItemError(f"no item file at {path}; pass --items or set ${ENV_ITEMS}")
    try:
        doc = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ItemError(f"{path} is not valid JSON: {exc}") from exc
    items = validate_document(doc, path)
    declared = doc["sets"]
    if sets:
        unknown = [s for s in sets if s not in declared]
        _require(not unknown,
                 f"unknown set(s) {', '.join(unknown)}; pick from "
                 f"{', '.join(declared)}")
        items = [i for i in items if i.set in sets]
        _require(items, "that --sets selection holds no items")
    return ItemSet(id=str(doc.get("id") or path.stem),
                   version=int(doc.get("version") or 1),
                   description=str(doc.get("description") or ""),
                   sets={name: declared[name] for name in declared},
                   items=tuple(items), path=path)


__all__ = ["CHOICE", "NOUL", "KINDS", "TRUE", "FALSE", "ENV_ITEMS", "Item",
           "ItemError", "ItemSet", "default_items_path", "items_file_label",
           "load_items", "validate_document"]
