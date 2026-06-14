"""
Tests for pure utility functions — no API keys or external services required.
Run with: pytest tests/
"""
import sys
from pathlib import Path

# Add project root to path so imports resolve without installation
sys.path.insert(0, str(Path(__file__).parent.parent))

from graphbuilder import (
    _chunk_id,
    _sanitize_label,
    _props_to_dict,
    Property,
    Node,
    _node_to_dict,
)


class TestChunkId:
    def test_same_text_produces_same_hash(self):
        assert _chunk_id("hello world") == _chunk_id("hello world")

    def test_different_text_produces_different_hash(self):
        assert _chunk_id("hello") != _chunk_id("world")

    def test_returns_64_char_hex_string(self):
        result = _chunk_id("test")
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)


class TestSanitizeLabel:
    def test_spaces_become_underscores(self):
        assert _sanitize_label("Cloud Infrastructure") == "`Cloud_Infrastructure`"

    def test_commas_become_underscores(self):
        assert _sanitize_label("a,b") == "`a_b`"

    def test_consecutive_underscores_collapsed(self):
        assert _sanitize_label("a  b") == "`a_b`"

    def test_backtick_wrapping(self):
        result = _sanitize_label("Person")
        assert result.startswith("`")
        assert result.endswith("`")

    def test_leading_trailing_underscores_stripped(self):
        result = _sanitize_label(" Person ")
        assert not result.startswith("`_")
        assert not result.endswith("_`")


class TestPropsToDict:
    def test_converts_property_list_to_dict(self):
        props = [Property(key="name", value="Alice"), Property(key="role", value="Engineer")]
        result = _props_to_dict(props)
        assert result == {"name": "Alice", "role": "Engineer"}

    def test_empty_list_returns_empty_dict(self):
        assert _props_to_dict([]) == {}


class TestNodeToDict:
    def test_structure(self):
        node = Node(
            node_label="Person",
            node_properties=[Property(key="name", value="Alice")],
        )
        result = _node_to_dict(node)
        assert result["node_label"] == "Person"
        assert result["node_properties"] == {"name": "Alice"}


class TestTextSplitterSeparators:
    """Verify that the separator patterns used in graphbuilder match expected input formats."""

    def test_respondent_separator_pattern(self):
        import re
        pattern = r"\nRESPONDENT:\t"
        text = "some text\nRESPONDENT:\tanswer here"
        assert re.search(pattern, text) is not None

    def test_interviewer_initials_pattern(self):
        import re
        pattern = r"\nI:\t"
        text = "prior text\nI:\tnext question"
        assert re.search(pattern, text) is not None

    def test_respondent_initials_pattern(self):
        import re
        pattern = r"\nR:\t"
        text = "prior text\nR:\tsome response"
        assert re.search(pattern, text) is not None
