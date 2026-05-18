"""JSON-backed v2 Memory Atom store."""

from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

from src.memory.schema import (
    DEFAULT_MEMORY,
    DEFAULT_USER_ID,
    DEFAULT_VALUE_MEMORY,
    MAX_SHORT_TERM_ITEMS,
    MEMORY_STORE_VERSION,
    MemoryAtom,
    atom_from_legacy_item,
    build_memory_atom,
)
from src.memory.utils import as_list, deep_merge, now_iso, unique


PROFILE_KEYS = {
    "stable_profile",
    "companion_profile",
    "preference_profile",
    "history_feedback",
    "derived_defaults",
    "value_profile",
}

LEGACY_TOP_LEVEL_KEYS = {
    "user_id",
    "version",
    "profile",
    "atoms",
    "graph",
    "episodic_memory",
    "short_term_items",
    *PROFILE_KEYS,
}


def memory_store_path() -> Path | None:
    raw_path = os.environ.get("WF_MEMORY_STORE_PATH", "").strip()
    return Path(raw_path).expanduser() if raw_path else None


def normalize_memory(memory: dict[str, Any] | None, user_id: str) -> dict[str, Any]:
    source = deepcopy(memory or {})
    if isinstance(source.get("profile"), dict):
        source = deep_merge(source["profile"], source)
    normalized = deep_merge(DEFAULT_MEMORY, source)
    normalized["user_id"] = user_id
    normalized.setdefault("stable_profile", {})
    normalized.setdefault("companion_profile", {})
    normalized.setdefault("preference_profile", {})
    normalized.setdefault("history_feedback", [])
    normalized.setdefault("derived_defaults", {})
    normalized.setdefault("episodic_memory", [])
    normalized.setdefault("short_term_items", [])
    normalized["value_profile"] = unique(
        normalized.get("value_profile") or deepcopy(DEFAULT_VALUE_MEMORY)
    )
    return normalized


def _profile_from_memory(memory: dict[str, Any]) -> dict[str, Any]:
    return {
        "stable_profile": deepcopy(memory.get("stable_profile", {})),
        "companion_profile": deepcopy(memory.get("companion_profile", {})),
        "preference_profile": deepcopy(memory.get("preference_profile", {})),
        "history_feedback": deepcopy(memory.get("history_feedback", [])),
        "derived_defaults": deepcopy(memory.get("derived_defaults", {})),
        "value_profile": deepcopy(memory.get("value_profile") or DEFAULT_VALUE_MEMORY),
    }


def _legacy_extra_fields(memory: dict[str, Any]) -> dict[str, Any]:
    return {
        key: deepcopy(value)
        for key, value in memory.items()
        if key not in LEGACY_TOP_LEVEL_KEYS and not key.startswith("_")
    }


def _profile_atoms(memory: dict[str, Any], user_id: str) -> list[MemoryAtom]:
    atoms: list[MemoryAtom] = []
    stable_profile = memory.get("stable_profile", {}) or {}
    preference_profile = memory.get("preference_profile", {}) or {}

    atoms.append(
        build_memory_atom(
            "stable_profile",
            "stable_profile",
            "long_term",
            "用户长期稳定画像",
            [
                stable_profile.get("consumption_level"),
                stable_profile.get("default_transport"),
            ],
            stable_profile,
            user_id=user_id,
            source="stable_profile",
            confidence=0.82,
            entities=["user"],
        )
    )

    atoms.append(
        build_memory_atom(
            "preference_profile",
            "preference",
            "long_term",
            "用户偏好和规避项",
            [
                *as_list(preference_profile.get("food")),
                *as_list(preference_profile.get("activity")),
                *as_list(preference_profile.get("avoid")),
            ],
            preference_profile,
            user_id=user_id,
            source="preference_profile",
            confidence=0.78,
            entities=["user"],
        )
    )

    companion_profile = memory.get("companion_profile", {}) or {}
    child_profile = companion_profile.get("child")
    if isinstance(child_profile, dict):
        atoms.append(
            build_memory_atom(
                "companion_child",
                "companion",
                child_profile.get("ttl", "long_term"),
                "常见同行孩子画像",
                ["child", *as_list(child_profile.get("needs"))],
                child_profile,
                user_id=user_id,
                source=child_profile.get("source", "companion_profile"),
                confidence=child_profile.get("confidence", 0.75),
                entities=["child"],
                relations=[
                    {
                        "source": "user",
                        "predicate": "travels_with",
                        "target": "child",
                        "confidence": child_profile.get("confidence", 0.75),
                    }
                ],
            )
        )

    spouse_profile = companion_profile.get("wife")
    if isinstance(spouse_profile, dict) and spouse_profile.get("state"):
        atoms.append(
            build_memory_atom(
                "companion_wife",
                "companion",
                spouse_profile.get("ttl", "short_term"),
                "伴侣近期饮食状态",
                [
                    "wife",
                    spouse_profile.get("state"),
                    *as_list(spouse_profile.get("needs")),
                ],
                spouse_profile,
                user_id=user_id,
                source=spouse_profile.get("source", "companion_profile"),
                confidence=spouse_profile.get("confidence", 0.7),
                ttl_turns=spouse_profile.get("ttl_turns"),
                entities=["wife"],
                relations=[
                    {
                        "source": "wife",
                        "predicate": "prefers",
                        "target": "low_calorie",
                        "confidence": spouse_profile.get("confidence", 0.7),
                    }
                ],
            )
        )

    for item in memory.get("value_profile", []) or []:
        if not isinstance(item, dict):
            continue
        atoms.append(
            build_memory_atom(
                f"value_{item.get('value_id')}",
                "value",
                item.get("ttl", "long_term"),
                item.get("label", item.get("value_id", "value")),
                [item.get("value_id"), *as_list(item.get("evidence"))],
                dict(item),
                user_id=user_id,
                source=item.get("source", "value_profile"),
                confidence=item.get("confidence", 0.7),
                entities=["user"],
            )
        )

    for idx, feedback in enumerate(memory.get("history_feedback", []) or []):
        if not isinstance(feedback, dict):
            continue
        atoms.append(
            build_memory_atom(
                f"feedback_{feedback.get('plan_id', idx)}",
                "feedback",
                "long_term",
                "历史方案反馈",
                [
                    *as_list(feedback.get("positive")),
                    *as_list(feedback.get("negative")),
                ],
                feedback,
                user_id=user_id,
                source=feedback.get("source", "history_feedback"),
                confidence=feedback.get("confidence", 0.65),
                entities=["user"],
            )
        )

    return atoms


def _normalize_atom_list(values: Any, user_id: str) -> list[MemoryAtom]:
    atoms = []
    for item in as_list(values):
        if isinstance(item, dict):
            atoms.append(atom_from_legacy_item(item, user_id=user_id))
    return atoms


def _merge_atoms(*groups: list[MemoryAtom]) -> list[MemoryAtom]:
    merged: dict[str, MemoryAtom] = {}
    for group in groups:
        for atom in group:
            memory_id = atom.get("memory_id")
            if not memory_id:
                continue
            merged[str(memory_id)] = atom
    return list(merged.values())


def build_memory_graph(atoms: list[MemoryAtom]) -> dict[str, Any]:
    entities: dict[str, dict[str, Any]] = {}
    relations: list[dict[str, Any]] = []

    for atom in atoms:
        memory_id = atom.get("memory_id")
        for entity in atom.get("entities", []) or []:
            entity_id = str(entity)
            record = entities.setdefault(
                entity_id,
                {"entity_id": entity_id, "memory_ids": []},
            )
            if memory_id and memory_id not in record["memory_ids"]:
                record["memory_ids"].append(memory_id)

        for relation in atom.get("relations", []) or []:
            relation_record = dict(relation)
            relation_record.setdefault("memory_id", memory_id)
            relations.append(relation_record)

    return {
        "entities": entities,
        "relations": relations,
    }


def memory_to_v2_user(memory: dict[str, Any]) -> dict[str, Any]:
    user_id = str(memory.get("user_id") or DEFAULT_USER_ID)
    normalized = normalize_memory(memory, user_id)
    existing_atoms = _normalize_atom_list(normalized.get("atoms", []), user_id)
    episodic_atoms = _normalize_atom_list(
        normalized.get("episodic_memory", []), user_id
    )
    short_term_atoms = _normalize_atom_list(
        normalized.get("short_term_items", []), user_id
    )
    atoms = _merge_atoms(
        _profile_atoms(normalized, user_id),
        existing_atoms,
        episodic_atoms,
        short_term_atoms,
    )

    return {
        "user_id": user_id,
        "profile": _profile_from_memory(normalized),
        "atoms": atoms,
        "graph": build_memory_graph(atoms),
        "legacy": _legacy_extra_fields(normalized),
        "updated_at": now_iso(),
    }


def v2_user_to_memory(user_record: dict[str, Any], user_id: str) -> dict[str, Any]:
    profile = user_record.get("profile", {}) if isinstance(user_record, dict) else {}
    legacy = user_record.get("legacy", {}) if isinstance(user_record, dict) else {}
    memory = normalize_memory({**legacy, **profile}, user_id)
    atoms = _normalize_atom_list(user_record.get("atoms", []), user_id)
    graph = user_record.get("graph") or build_memory_graph(atoms)

    memory["profile"] = deepcopy(profile)
    memory["atoms"] = atoms
    memory["graph"] = graph
    memory["episodic_memory"] = [
        atom for atom in atoms if atom.get("kind") == "episode"
    ][-20:]
    memory["short_term_items"] = [
        atom
        for atom in atoms
        if atom.get("scope") == "short_term"
        and atom.get("kind") in {"episode", "companion"}
    ][-MAX_SHORT_TERM_ITEMS:]
    memory["_store_version"] = MEMORY_STORE_VERSION
    return memory


def migrate_store_payload(payload: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    if not isinstance(payload, dict):
        return {"version": MEMORY_STORE_VERSION, "users": {}}, True

    if payload.get("version") == MEMORY_STORE_VERSION and isinstance(
        payload.get("users"), dict
    ):
        users = {}
        changed = False
        for user_id, record in payload["users"].items():
            if (
                isinstance(record, dict)
                and isinstance(record.get("profile"), dict)
                and isinstance(record.get("atoms"), list)
                and isinstance(record.get("graph"), dict)
            ):
                users[str(user_id)] = record
            else:
                changed = True
                users[str(user_id)] = memory_to_v2_user(
                    normalize_memory(
                        record if isinstance(record, dict) else {}, str(user_id)
                    )
                )
        return {"version": MEMORY_STORE_VERSION, "users": users}, changed

    if isinstance(payload.get("users"), dict):
        legacy_users = payload["users"]
    elif payload.get("user_id"):
        legacy_users = {str(payload["user_id"]): payload}
    else:
        legacy_users = {}

    users = {
        str(user_id): memory_to_v2_user(
            normalize_memory(record if isinstance(record, dict) else {}, str(user_id))
        )
        for user_id, record in legacy_users.items()
    }
    return {"version": MEMORY_STORE_VERSION, "users": users}, True


def _empty_store() -> dict[str, Any]:
    return {"version": MEMORY_STORE_VERSION, "users": {}}


class JSONMemoryStore:
    """Small JSON store with automatic v1 to v2 migration."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return _empty_store()

        try:
            with self.path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return _empty_store()

        store, migrated = migrate_store_payload(payload)
        if migrated:
            self._write(store)
        return store

    def _write(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        temp_path.replace(self.path)

    def load(
        self, user_id: str, seed_memory: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        base = normalize_memory(seed_memory or {}, user_id)
        store = self._read()
        persisted = store.get("users", {}).get(user_id)
        if isinstance(persisted, dict):
            stored_memory = v2_user_to_memory(persisted, user_id)
            base = normalize_memory(deep_merge(base, stored_memory), user_id)
            base["profile"] = stored_memory.get("profile", {})
            base["atoms"] = stored_memory.get("atoms", [])
            base["graph"] = stored_memory.get("graph", {})
            base["_store_version"] = MEMORY_STORE_VERSION
        return base

    def save(self, memory: dict[str, Any]) -> None:
        user_id = str(memory.get("user_id") or DEFAULT_USER_ID)
        store = self._read()
        store.setdefault("users", {})[user_id] = memory_to_v2_user(memory)
        store["version"] = MEMORY_STORE_VERSION
        self._write(store)


def _decay_item_ttl(item: dict[str, Any]) -> dict[str, Any] | None:
    ttl_turns = item.get("ttl_turns")
    if ttl_turns is None:
        return dict(item)
    try:
        ttl_turns = int(ttl_turns)
    except (TypeError, ValueError):
        return dict(item)
    if ttl_turns <= 1:
        return None
    next_item = dict(item)
    next_item["ttl_turns"] = ttl_turns - 1
    return next_item


def _decay_items(items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], set[str]]:
    retained = []
    expired_ids: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        decayed = _decay_item_ttl(item)
        if decayed is None:
            memory_id = item.get("memory_id")
            if memory_id:
                expired_ids.add(str(memory_id))
            continue
        retained.append(decayed)
    return retained, expired_ids


def _decay_short_term_profile(memory: dict[str, Any]) -> None:
    companion_profile = memory.get("companion_profile", {}) or {}
    wife_profile = companion_profile.get("wife")
    if not isinstance(wife_profile, dict) or wife_profile.get("ttl_turns") is None:
        return
    decayed_wife = _decay_item_ttl(wife_profile)
    if decayed_wife is None:
        companion_profile["wife"] = {
            "state": None,
            "needs": [],
            "ttl": "expired",
            "ttl_turns": 0,
            "source": wife_profile.get("source", "ttl_decay"),
        }
    else:
        companion_profile["wife"] = decayed_wife


def decay_short_term_items(memory: dict[str, Any]) -> dict[str, Any]:
    atoms = [atom for atom in memory.get("atoms", []) or [] if isinstance(atom, dict)]
    if atoms:
        retained_atoms, expired_ids = _decay_items(
            [
                atom
                for atom in atoms
                if atom.get("scope") == "short_term"
                and atom.get("ttl_turns") is not None
            ]
        )
        retained_by_id = {
            str(atom.get("memory_id")): atom
            for atom in retained_atoms
            if atom.get("memory_id")
        }
        next_atoms = []
        for atom in atoms:
            memory_id = str(atom.get("memory_id") or "")
            if memory_id in expired_ids:
                continue
            next_atoms.append(retained_by_id.get(memory_id, atom))
        memory["atoms"] = next_atoms
        memory["graph"] = build_memory_graph(next_atoms)
        memory["short_term_items"] = [
            atom
            for atom in next_atoms
            if atom.get("scope") == "short_term"
            and atom.get("kind") in {"episode", "companion"}
        ][-MAX_SHORT_TERM_ITEMS:]
    else:
        retained, _expired_ids = _decay_items(memory.get("short_term_items", []) or [])
        memory["short_term_items"] = retained[-MAX_SHORT_TERM_ITEMS:]

    _decay_short_term_profile(memory)
    return memory


def load_memory(
    user_id: str | None,
    seed_memory: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load user memory from the optional v2 store and deterministic seed data."""

    resolved_user_id = str(
        user_id or (seed_memory or {}).get("user_id") or DEFAULT_USER_ID
    )
    store_path = memory_store_path()
    if store_path:
        return JSONMemoryStore(store_path).load(resolved_user_id, seed_memory)
    return normalize_memory(seed_memory or {}, resolved_user_id)


def save_memory(memory: dict[str, Any]) -> None:
    """Persist memory when `WF_MEMORY_STORE_PATH` is configured."""

    store_path = memory_store_path()
    if not store_path:
        return
    JSONMemoryStore(store_path).save(memory)
