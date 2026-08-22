import unittest
from unittest.mock import Mock, patch

import tiktoken

from tool_logic import (
    JourneyGraph,
    GraphClient,
    RECALL_TOKEN_LIMIT,
    RouteNotFoundError,
    StudyCorpus,
    find_least_cost_path,
)


class StudyCorpusTests(unittest.TestCase):
    def setUp(self) -> None:
        self.corpus = StudyCorpus(
            [
                (
                    "Research Station",
                    """## Calibration

The Kesterline array was recalibrated on 14 March, restoring the outer ring.

The Halberd sub-array received routine maintenance on 12 March.

## Locations

The Pellucid Shelf observation post is reached from STOP_04.
""",
                ),
                (
                    "Transit Authority",
                    """## Timetable

The Verity Observatory is served by STOP_05 on the Russet Line.

## Fares

The daily fare cap is four pounds ninety.
""",
                ),
            ]
        )

    def test_retrieval_finds_the_required_fact(self) -> None:
        passages = self.corpus.search(
            "When was the Kesterline array brought back into alignment?"
        )

        self.assertTrue(passages)
        self.assertIn("14 March", passages[0])
        self.assertNotIn("12 March", passages[0])

    def test_retrieval_finds_school_trip_stop(self) -> None:
        passages = self.corpus.search("Which stop serves the Verity Observatory?")

        self.assertIn("STOP_05", "\n".join(passages))

    def test_retrieval_never_exceeds_900_content_tokens(self) -> None:
        passages = self.corpus.search("Tell me about the station", token_limit=25)
        encoding = tiktoken.get_encoding("o200k_base")

        self.assertLessEqual(
            sum(len(encoding.encode(passage)) for passage in passages), 25
        )
        self.assertLessEqual(
            sum(len(encoding.encode(passage)) for passage in passages),
            RECALL_TOKEN_LIMIT,
        )


class RoutingTests(unittest.TestCase):
    def test_padded_opaque_map_id_is_forwarded_unchanged(self) -> None:
        map_id = (
            "gAAAAABqiRM5wJJTn2ginyItG0HtpqFjuUAipnW-s_67o084XnL6EhUtSCcg"
            "NpXl4wPHSWJdjV2Ih8DKXNI9PCQhg8lbldOz_w=="
        )
        response = Mock()
        response.json.return_value = {
            "adjacency": {"A": {"B": 1}},
            "tolls": {"A": 0, "B": 0},
        }

        with patch("tool_logic.httpx.get", return_value=response) as get:
            graph = GraphClient("https://example.test").fetch(map_id)

        self.assertEqual(graph.adjacency["A"]["B"], 1)
        get.assert_called_once_with(
            "https://example.test/graph",
            params={"map_id": map_id},
            timeout=4.0,
            follow_redirects=True,
        )

    def test_map_id_rejects_only_invalid_transport_values(self) -> None:
        client = GraphClient("https://example.test")

        for map_id in ("", "contains whitespace", "\n"):
            with self.subTest(map_id=map_id), self.assertRaises(ValueError):
                client.fetch(map_id)

    def test_entry_tolls_change_the_cheapest_route(self) -> None:
        graph = JourneyGraph.from_payload(
            {
                "adjacency": {
                    "A": {"B": 1, "C": 3},
                    "B": {"D": 1},
                    "C": {"D": 3},
                },
                "tolls": {"A": 0, "B": 10, "C": 0, "D": 0},
            }
        )

        self.assertEqual(find_least_cost_path(graph, "A", "D"), ("A", "C", "D"))

    def test_hop_limit_is_inclusive_of_the_requested_move(self) -> None:
        graph = JourneyGraph.from_payload(
            {
                "adjacency": {
                    "A": {"B": 1, "D": 10},
                    "B": {"C": 1},
                    "C": {"D": 1},
                },
                "tolls": {"A": 0, "B": 0, "C": 0, "D": 0},
            }
        )

        self.assertEqual(
            find_least_cost_path(graph, "A", "D", hops_remaining=3),
            ("A", "B", "C", "D"),
        )
        self.assertEqual(
            find_least_cost_path(graph, "A", "D", hops_remaining=1),
            ("A", "D"),
        )

    def test_graph_is_directed_and_avoids_visited_nodes(self) -> None:
        graph = JourneyGraph.from_payload(
            {
                "adjacency": {
                    "A": {"B": 1, "C": 3},
                    "B": {"A": 1, "D": 1},
                    "C": {"D": 2},
                },
                "tolls": {"A": 0, "B": 0, "C": 0, "D": 0},
            }
        )

        self.assertEqual(
            find_least_cost_path(graph, "A", "D", avoid_nodes=["B"]),
            ("A", "C", "D"),
        )
        with self.assertRaises(RouteNotFoundError):
            find_least_cost_path(graph, "D", "A")

    def test_unreachable_within_hop_limit_is_rejected(self) -> None:
        graph = JourneyGraph.from_payload(
            {
                "adjacency": {"A": {"B": 1}, "B": {"D": 1}},
                "tolls": {"A": 0, "B": 0, "D": 0},
            }
        )

        with self.assertRaises(RouteNotFoundError):
            find_least_cost_path(graph, "A", "D", hops_remaining=1)


if __name__ == "__main__":
    unittest.main()
