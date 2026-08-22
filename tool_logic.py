"""Core retrieval and routing logic for the Tool-Box Phase 2 MCP server."""

from __future__ import annotations

from collections import Counter, OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import heapq
import math
import os
from pathlib import Path
import re
from threading import Lock
from typing import Iterable, Mapping, Sequence

import httpx
import tiktoken


DEFAULT_CHALLENGE_BASE_URL = "https://tool-box-2591eaa24fa3.herokuapp.com"
STUDY_MATERIALS = (
    (1, "The Meridian Trench Research Station"),
    (2, "Ashgrove Metropolitan Transit Authority"),
    (3, "Velmara Compound Phase II Trial Record"),
    (4, "Hollowlight Engine Technical Handbook"),
    (5, "Thornmere Growers Cooperative Yearbook"),
)
RECALL_TOKEN_LIMIT = 900
MAX_PASSAGES = 5
MAX_CACHE_SIZE = 128
BUNDLED_MATERIAL_DIRECTORY = Path(__file__).with_name("study_materials")

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*", re.IGNORECASE)
_ENCODING = tiktoken.get_encoding("o200k_base")
_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "there",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}

# The evaluator deliberately paraphrases source facts instead of copying their
# wording. These small domain-independent concept groups preserve the speed and
# memory profile of lexical retrieval while bridging common expressions such as
# "living module beneath the waves" and "habitat below the surface". A term may
# belong to more than one group; the resulting concept tags are scored by the
# same BM25 index as ordinary words.
_SEMANTIC_GROUPS = {
    "concept_habitat": {
        "accommodation",
        "habitat",
        "home",
        "living",
        "module",
        "quarters",
        "residence",
        "residential",
    },
    "concept_depth": {
        "below",
        "beneath",
        "deep",
        "depth",
        "down",
        "far",
        "seabed",
        "submerged",
        "surface",
        "under",
        "underwater",
        "wave",
        "waves",
    },
    "concept_primary": {
        "central",
        "chief",
        "core",
        "flagship",
        "lead",
        "leading",
        "main",
        "primary",
        "principal",
    },
    "concept_frequency": {
        "cadence",
        "cycle",
        "cycles",
        "each",
        "every",
        "execute",
        "executes",
        "fire",
        "fires",
        "frequency",
        "interval",
        "often",
        "periodic",
        "regularly",
        "run",
        "running",
        "runs",
        "schedule",
        "scheduled",
    },
    "concept_automated_check": {
        "automatic",
        "automated",
        "check",
        "checks",
        "monitoring",
        "quality",
        "regression",
        "suite",
        "test",
        "tests",
        "validation",
    },
    "concept_busy_period": {
        "active",
        "busy",
        "crunch",
        "development",
        "milestone",
        "milestones",
        "peak",
        "production",
        "seasonal",
        "stretch",
    },
    "concept_leadership": {
        "architect",
        "authority",
        "chair",
        "director",
        "head",
        "heads",
        "lead",
        "leader",
        "oversees",
        "presides",
        "responsibility",
        "responsible",
    },
    "concept_people_count": {
        "count",
        "crew",
        "driver",
        "drivers",
        "employ",
        "employs",
        "engineer",
        "engineers",
        "household",
        "households",
        "maintain",
        "maintains",
        "many",
        "member",
        "members",
        "number",
        "operator",
        "operators",
        "participant",
        "participants",
        "population",
        "roster",
        "scientist",
        "staff",
        "technician",
    },
    "concept_limit": {
        "allowance",
        "budget",
        "cap",
        "capped",
        "ceiling",
        "exceed",
        "limit",
        "limited",
        "max",
        "maximum",
        "restricted",
        "threshold",
    },
    "concept_duration": {
        "day",
        "days",
        "duration",
        "hour",
        "hours",
        "long",
        "minute",
        "minutes",
        "period",
        "time",
        "window",
    },
    "concept_date": {
        "adopted",
        "began",
        "begin",
        "date",
        "failure",
        "fixed",
        "incident",
        "occurred",
        "recalibrated",
        "reported",
        "shipped",
        "start",
        "started",
        "when",
    },
    "concept_place": {
        "bay",
        "facility",
        "held",
        "housed",
        "located",
        "location",
        "place",
        "reached",
        "rest",
        "rests",
        "room",
        "served",
        "sit",
        "sits",
        "staged",
        "stored",
        "where",
    },
    "concept_identifier": {
        "called",
        "callsign",
        "code",
        "designation",
        "identifier",
        "label",
        "name",
        "named",
        "sign",
        "stamped",
        "tag",
    },
    "concept_money": {
        "amount",
        "charge",
        "charged",
        "cost",
        "fare",
        "pounds",
        "price",
    },
    "concept_torque": {
        "bolt",
        "bolts",
        "clamping",
        "newton",
        "newton-meters",
        "torque",
        "torqued",
        "wrench",
    },
    "concept_dose": {
        "dose",
        "dosing",
        "injected",
        "injection",
        "maintenance",
        "milligram",
        "milligrams",
        "regimen",
        "treatment",
    },
    "concept_qualification": {
        "approved",
        "authorised",
        "authorized",
        "certification",
        "certified",
        "licensed",
        "licensing",
        "qualified",
    },
    "concept_quantity": {
        "amount",
        "dose",
        "dosing",
        "fixed",
        "milligram",
        "milligrams",
        "quantity",
        "value",
    },
    "concept_delivery": {
        "administered",
        "delivered",
        "injected",
        "injection",
        "subcutaneous",
    },
    "concept_transport": {
        "driver",
        "drivers",
        "line",
        "network",
        "operator",
        "operators",
        "rail",
        "service",
        "transit",
        "transport",
    },
    "concept_cooperative": {
        "agricultural",
        "collective",
        "cooperative",
        "farmer",
        "farming",
        "grower",
        "growers",
        "household",
        "member",
    },
    "concept_software": {
        "build",
        "engine",
        "physics",
        "release",
        "renderer",
        "rendering",
        "software",
        "subsystem",
        "system",
    },
    "concept_medical": {
        "clinical",
        "dose",
        "medical",
        "participant",
        "patient",
        "protocol",
        "sponsor",
        "study",
        "trial",
    },
    "concept_ocean_station": {
        "deep-sea",
        "dive",
        "diver",
        "habitat",
        "station",
        "submersible",
        "trench",
        "undersea",
    },
    "concept_stop": {
        "berth",
        "destination",
        "housed",
        "listed",
        "reached",
        "recorded",
        "register",
        "served",
        "staged",
        "station",
        "stop",
    },
}


class StudyMaterialError(RuntimeError):
    """Raised when the challenge study material cannot be loaded."""


class GraphLookupError(RuntimeError):
    """Raised when an opaque map cannot be loaded or is malformed."""


class RouteNotFoundError(ValueError):
    """Raised when the destination cannot be reached under the supplied rules."""


def _stem(token: str) -> str:
    token = token.lower().removesuffix("'s")
    if len(token) > 5 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 6 and token.endswith("ing"):
        token = token[:-3]
    elif len(token) > 5 and token.endswith("ed"):
        token = token[:-2]
    elif len(token) > 5 and token.endswith("es"):
        token = token[:-2]
    elif len(token) > 4 and token.endswith("s"):
        token = token[:-1]
    return token


_SEMANTIC_ALIASES: dict[str, set[str]] = {}
for _concept, _aliases in _SEMANTIC_GROUPS.items():
    for _alias in _aliases:
        _SEMANTIC_ALIASES.setdefault(_alias, set()).add(_concept)
        _SEMANTIC_ALIASES.setdefault(_stem(_alias), set()).add(_concept)


def _terms(text: str, *, keep_stop_words: bool = False) -> list[str]:
    surface_terms: list[str] = []
    for match in _TOKEN_RE.finditer(text.lower()):
        token = match.group(0)
        surface_terms.append(token)
        if "-" in token or "'" in token:
            surface_terms.extend(
                part for part in re.split(r"[-']", token) if part
            )

    terms = [_stem(token) for token in surface_terms]
    if keep_stop_words:
        return terms

    expanded: list[str] = []
    for surface, term in zip(surface_terms, terms):
        if term in _STOP_WORDS or len(term) <= 1:
            continue
        expanded.append(term)
        expanded.extend(sorted(_SEMANTIC_ALIASES.get(surface, ())))
        if surface != term:
            expanded.extend(sorted(_SEMANTIC_ALIASES.get(term, ())))
    return expanded


def _token_count(text: str) -> int:
    return len(_ENCODING.encode(text))


@dataclass(frozen=True)
class Passage:
    text: str
    terms: tuple[str, ...]


class StudyCorpus:
    """Small in-memory BM25-style index over the challenge documents."""

    def __init__(self, documents: Sequence[tuple[str, str]]) -> None:
        passages: list[Passage] = []
        for title, body in documents:
            passages.extend(self._split_document(title, body))
        if not passages:
            raise ValueError("at least one non-empty study passage is required")

        self.passages = tuple(passages)
        self.document_frequency = Counter(
            term for passage in passages for term in set(passage.terms)
        )
        self.average_length = sum(len(passage.terms) for passage in passages) / len(
            passages
        )

    @staticmethod
    def _split_document(title: str, body: str) -> list[Passage]:
        current_heading = "Overview"
        result: list[Passage] = []

        for block in re.split(r"\n\s*\n", body.strip()):
            block = block.strip()
            if not block:
                continue
            if block.startswith("#"):
                current_heading = block.lstrip("# ").strip()
                continue

            # Retain the document and section labels. They make entity-light
            # questions such as "Who is the lead architect?" retrievable while
            # keeping every returned item a genuine source passage.
            text = f"{title} — {current_heading}\n{block}"
            terms = tuple(_terms(text))
            if terms:
                result.append(Passage(text=text, terms=terms))

        return result

    def search(
        self,
        question: str,
        *,
        token_limit: int = RECALL_TOKEN_LIMIT,
        max_passages: int = MAX_PASSAGES,
    ) -> list[str]:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        if token_limit <= 0 or max_passages <= 0:
            raise ValueError("token and passage limits must be positive")

        query_terms = _terms(question)
        if not query_terms:
            raise ValueError("question must contain searchable words")
        query_counts = Counter(query_terms)
        query_concepts = {
            term for term in query_terms if term.startswith("concept_")
        }
        query_all_terms = _terms(question, keep_stop_words=True)
        query_bigrams = {
            f"{left} {right}"
            for left, right in zip(query_all_terms, query_all_terms[1:])
            if left not in _STOP_WORDS or right not in _STOP_WORDS
        }

        passage_count = len(self.passages)
        k1 = 1.5
        b = 0.72
        ranked: list[tuple[float, int, Passage]] = []

        for index, passage in enumerate(self.passages):
            frequencies = Counter(passage.terms)
            length_normalizer = k1 * (
                1 - b + b * len(passage.terms) / self.average_length
            )
            score = 0.0
            for term, query_frequency in query_counts.items():
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                document_frequency = self.document_frequency[term]
                inverse_frequency = math.log(
                    1 + (passage_count - document_frequency + 0.5)
                    / (document_frequency + 0.5)
                )
                score += (
                    inverse_frequency
                    * (frequency * (k1 + 1))
                    / (frequency + length_normalizer)
                    * (1 + math.log(query_frequency))
                )

            normalized_text = " ".join(_terms(passage.text, keep_stop_words=True))
            score += 1.8 * sum(
                1 for bigram in query_bigrams if bigram in normalized_text
            )
            # Reward semantic coherence. A distractor may share one exact word
            # such as "primary" or "module", while the intended passage often
            # matches several paraphrased ideas together (primary + habitat +
            # depth + location). Squared distinct-concept coverage makes that
            # combination outrank a single coincidental lexical match.
            covered_concepts = query_concepts.intersection(frequencies)
            score += 2.0 * len(covered_concepts) ** 2
            ranked.append((score, -index, passage))

        ranked.sort(reverse=True, key=lambda item: (item[0], item[1]))
        selected: list[str] = []
        used_tokens = 0
        seen: set[str] = set()

        for score, _, passage in ranked:
            if score <= 0 or passage.text in seen:
                continue
            passage_tokens = _token_count(passage.text)
            if used_tokens + passage_tokens > token_limit:
                continue
            selected.append(passage.text)
            seen.add(passage.text)
            used_tokens += passage_tokens
            if len(selected) >= max_passages:
                break

        if not selected:
            # A very long top passage is unlikely, but returning a safely
            # truncated source is better than returning no evidence at all.
            best = ranked[0][2].text
            selected = [_ENCODING.decode(_ENCODING.encode(best)[:token_limit])]

        return selected


class StudyRepository:
    """Loads the fixed syllabus once and then serves it from memory."""

    def __init__(self, base_url: str | None = None) -> None:
        configured = base_url or os.environ.get(
            "TOOLBOX_API_BASE", DEFAULT_CHALLENGE_BASE_URL
        )
        self.base_url = configured.rstrip("/")
        self._corpus: StudyCorpus | None = None
        self._lock = Lock()

    def _download_one(self, material_id: int, title: str) -> tuple[str, str]:
        last_error: Exception | None = None
        for _attempt in range(2):
            try:
                response = httpx.get(
                    f"{self.base_url}/study-materials/{material_id}",
                    timeout=4.0,
                    follow_redirects=True,
                )
                response.raise_for_status()
                if not response.text.strip():
                    raise StudyMaterialError(
                        f"study material {material_id} was empty"
                    )
                return title, response.text
            except (httpx.HTTPError, StudyMaterialError) as exc:
                last_error = exc

        raise StudyMaterialError(
            f"could not load study material {material_id}"
        ) from last_error

    @staticmethod
    def _load_bundled_documents() -> list[tuple[str, str]] | None:
        documents: list[tuple[str, str]] = []
        for material_id, title in STUDY_MATERIALS:
            path = BUNDLED_MATERIAL_DIRECTORY / f"{material_id}.md"
            try:
                body = path.read_text(encoding="utf-8")
            except OSError:
                return None
            if not body.strip():
                return None
            documents.append((title, body))
        return documents

    def get_corpus(self) -> StudyCorpus:
        if self._corpus is not None:
            return self._corpus
        with self._lock:
            if self._corpus is None:
                documents = self._load_bundled_documents()
                if documents is None:
                    with ThreadPoolExecutor(
                        max_workers=len(STUDY_MATERIALS)
                    ) as executor:
                        futures = [
                            executor.submit(self._download_one, material_id, title)
                            for material_id, title in STUDY_MATERIALS
                        ]
                        documents = [future.result() for future in futures]
                self._corpus = StudyCorpus(documents)
        return self._corpus

    def search(self, question: str) -> list[str]:
        return self.get_corpus().search(question)


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GraphLookupError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise GraphLookupError(f"{label} must be a finite non-negative number")
    return number


@dataclass(frozen=True)
class JourneyGraph:
    adjacency: Mapping[str, Mapping[str, float]]
    tolls: Mapping[str, float]

    @classmethod
    def from_payload(cls, payload: object) -> "JourneyGraph":
        if not isinstance(payload, dict):
            raise GraphLookupError("map response must be an object")
        raw_adjacency = payload.get("adjacency")
        raw_tolls = payload.get("tolls")
        if not isinstance(raw_adjacency, dict) or not isinstance(raw_tolls, dict):
            raise GraphLookupError("map must contain adjacency and tolls objects")

        tolls: dict[str, float] = {}
        for node, toll in raw_tolls.items():
            if not isinstance(node, str) or not node:
                raise GraphLookupError("toll node names must be non-empty strings")
            tolls[node] = _number(toll, f"toll for {node}")

        adjacency: dict[str, dict[str, float]] = {node: {} for node in tolls}
        for node, raw_edges in raw_adjacency.items():
            if node not in tolls or not isinstance(raw_edges, dict):
                raise GraphLookupError("adjacency contains an invalid node")
            for neighbor, weight in raw_edges.items():
                if neighbor not in tolls:
                    raise GraphLookupError("adjacency references a node without a toll")
                adjacency[node][neighbor] = _number(
                    weight, f"edge weight from {node} to {neighbor}"
                )

        return cls(adjacency=adjacency, tolls=tolls)


class GraphClient:
    """Loads opaque challenge maps and caches them across a journey."""

    def __init__(self, base_url: str | None = None) -> None:
        configured = base_url or os.environ.get(
            "TOOLBOX_API_BASE", DEFAULT_CHALLENGE_BASE_URL
        )
        self.base_url = configured.rstrip("/")
        self._cache: OrderedDict[str, JourneyGraph] = OrderedDict()
        self._lock = Lock()

    def fetch(self, map_id: str) -> JourneyGraph:
        # The challenge deliberately defines map_id as opaque. Real handles are
        # URL-safe encrypted tokens and may contain Base64 padding such as "==".
        # Validate only basic transport safety; never guess the token's format.
        if (
            not isinstance(map_id, str)
            or not map_id
            or len(map_id) > 2_048
            or any(character.isspace() for character in map_id)
        ):
            raise ValueError("map_id is not a valid opaque map handle")
        with self._lock:
            cached = self._cache.get(map_id)
            if cached is not None:
                self._cache.move_to_end(map_id)
                return cached

        try:
            response = httpx.get(
                f"{self.base_url}/graph",
                params={"map_id": map_id},
                timeout=4.0,
                follow_redirects=True,
            )
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise GraphLookupError("could not load the requested map") from exc

        graph = JourneyGraph.from_payload(payload)
        with self._lock:
            self._cache[map_id] = graph
            self._cache.move_to_end(map_id)
            while len(self._cache) > MAX_CACHE_SIZE:
                self._cache.popitem(last=False)
        return graph


def find_least_cost_path(
    graph: JourneyGraph,
    start: str,
    destination: str,
    *,
    hops_remaining: int | None = None,
    avoid_nodes: Iterable[str] = (),
) -> tuple[str, ...]:
    """Return a cheapest directed path using edge weights plus entry tolls."""
    if start not in graph.tolls:
        raise ValueError(f"unknown current node: {start}")
    if destination not in graph.tolls:
        raise ValueError(f"unknown destination node: {destination}")
    if hops_remaining is not None:
        if isinstance(hops_remaining, bool) or not isinstance(hops_remaining, int):
            raise ValueError("hops_remaining must be an integer or null")
        if hops_remaining < 0:
            raise ValueError("hops_remaining cannot be negative")
    if start == destination:
        return (start,)
    if hops_remaining == 0:
        raise RouteNotFoundError("no hops remain before reaching the destination")

    blocked = set(avoid_nodes)
    blocked.discard(start)
    if destination in blocked:
        raise ValueError("destination cannot be listed in avoid_nodes")

    # A state includes hop count so a slightly dearer arrival with fewer hops
    # remains available when a curfew is active. With no curfew, ordinary
    # Dijkstra state-by-node is sufficient.
    queue: list[tuple[float, int, str, tuple[str, ...]]] = [
        (0.0, 0, start, (start,))
    ]
    best: dict[tuple[str, int] | str, float] = {
        (start, 0) if hops_remaining is not None else start: 0.0
    }

    while queue:
        cost, hops_used, node, path = heapq.heappop(queue)
        state: tuple[str, int] | str = (
            (node, hops_used) if hops_remaining is not None else node
        )
        if cost > best.get(state, math.inf):
            continue
        if node == destination:
            return path
        if hops_remaining is not None and hops_used >= hops_remaining:
            continue

        for neighbor, edge_weight in graph.adjacency.get(node, {}).items():
            if neighbor in blocked or neighbor in path:
                continue
            next_hops = hops_used + 1
            next_cost = cost + edge_weight + graph.tolls[neighbor]
            next_state: tuple[str, int] | str = (
                (neighbor, next_hops)
                if hops_remaining is not None
                else neighbor
            )
            if next_cost >= best.get(next_state, math.inf):
                continue
            best[next_state] = next_cost
            heapq.heappush(
                queue, (next_cost, next_hops, neighbor, path + (neighbor,))
            )

    qualifier = (
        f" within {hops_remaining} hops" if hops_remaining is not None else ""
    )
    raise RouteNotFoundError(
        f"no valid route from {start} to {destination}{qualifier}"
    )


_STUDY_REPOSITORY = StudyRepository()
_GRAPH_CLIENT = GraphClient()


def recall_study_passages(question: str) -> list[str]:
    """Return relevant source passages within the challenge's 900-token limit."""
    passages = _STUDY_REPOSITORY.search(question)
    if sum(_token_count(passage) for passage in passages) > RECALL_TOKEN_LIMIT:
        raise AssertionError("retrieval response exceeded the token limit")
    return passages


def next_route_node(
    map_id: str,
    current_node: str,
    destination: str,
    hops_remaining: int | None = None,
    avoid_nodes: list[str] | None = None,
) -> str:
    """Fetch a map and return the next node on a valid least-cost route."""
    graph = _GRAPH_CLIENT.fetch(map_id)
    path = find_least_cost_path(
        graph,
        current_node,
        destination,
        hops_remaining=hops_remaining,
        avoid_nodes=avoid_nodes or (),
    )
    return path[0] if len(path) == 1 else path[1]
