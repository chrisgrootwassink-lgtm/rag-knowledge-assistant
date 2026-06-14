"""Unit tests for core utilities — no API keys or external services required.

Run with:
    pytest tests/
"""

import re

from rag_assistant.graph.builder import (
    Checkpoint,
    _chunk_id,
    _node_to_dict,
    _props_to_dict,
    _sanitize_label,
)
from rag_assistant.graph.models import Node, Property


class TestChunkId:
    """Tests for the _chunk_id hashing utility."""

    def test_same_text_produces_same_hash(self) -> None:
        assert _chunk_id("hello world") == _chunk_id("hello world")

    def test_different_texts_produce_different_hashes(self) -> None:
        assert _chunk_id("hello") != _chunk_id("world")

    def test_returns_64_char_hex_string(self) -> None:
        result = _chunk_id("test input")
        assert len(result) == 64
        assert all(char in "0123456789abcdef" for char in result)


class TestSanitizeLabel:
    """Tests for the Neo4j label sanitisation utility."""

    def test_spaces_replaced_with_underscores(self) -> None:
        assert _sanitize_label("Cloud Infrastructure") == "`Cloud_Infrastructure`"

    def test_commas_replaced_with_underscores(self) -> None:
        assert _sanitize_label("a,b") == "`a_b`"

    def test_consecutive_underscores_collapsed(self) -> None:
        assert _sanitize_label("a  b") == "`a_b`"

    def test_result_is_backtick_wrapped(self) -> None:
        result = _sanitize_label("Person")
        assert result.startswith("`") and result.endswith("`")

    def test_no_leading_or_trailing_underscores_inside_backticks(self) -> None:
        result = _sanitize_label(" Person ")
        assert not result.startswith("`_")
        assert not result.endswith("_`")


class TestPropsToDict:
    """Tests for Property list → dict conversion."""

    def test_converts_property_list_to_dict(self) -> None:
        properties = [
            Property(key="name", value="Alice"),
            Property(key="role", value="Engineer"),
        ]
        assert _props_to_dict(properties) == {
            "name": "Alice",
            "role": "Engineer",
        }

    def test_empty_list_returns_empty_dict(self) -> None:
        assert _props_to_dict([]) == {}


class TestNodeToDict:
    """Tests for Node model serialisation."""

    def test_node_label_is_preserved(self) -> None:
        node = Node(
            node_label="Person",
            node_properties=[Property(key="name", value="Alice")],
        )
        assert _node_to_dict(node)["node_label"] == "Person"

    def test_node_properties_are_flattened(self) -> None:
        node = Node(
            node_label="Person",
            node_properties=[Property(key="name", value="Alice")],
        )
        assert _node_to_dict(node)["node_properties"] == {"name": "Alice"}


class TestCheckpoint:
    """Tests for the Checkpoint persistence class."""

    def test_new_checkpoint_is_empty(self, tmp_path) -> None:
        checkpoint = Checkpoint(path=tmp_path / "cp.json")
        assert checkpoint.count == 0

    def test_mark_done_adds_to_processed(self, tmp_path) -> None:
        checkpoint = Checkpoint(path=tmp_path / "cp.json")
        checkpoint.mark_done("abc123")
        assert checkpoint.contains("abc123")

    def test_checkpoint_persists_across_instances(self, tmp_path) -> None:
        path = tmp_path / "cp.json"
        first = Checkpoint(path=path)
        first.mark_done("abc123")

        second = Checkpoint(path=path)
        assert second.contains("abc123")

    def test_unknown_id_returns_false(self, tmp_path) -> None:
        checkpoint = Checkpoint(path=tmp_path / "cp.json")
        assert not checkpoint.contains("not_here")


class TestSpeakerSeparatorPatterns:
    """Tests that separator regex patterns match expected transcript formats."""

    def test_respondent_label(self) -> None:
        assert re.search(r"\nRESPONDENT:\t", "text\nRESPONDENT:\tanswer")

    def test_interviewer_initial(self) -> None:
        assert re.search(r"\nI:\t", "text\nI:\tquestion")

    def test_respondent_initial(self) -> None:
        assert re.search(r"\nR:\t", "text\nR:\tresponse")
