import unittest

import tiktoken

from tool_logic import (
    JourneyGraph,
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
