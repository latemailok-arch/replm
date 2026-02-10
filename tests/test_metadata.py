"""Tests for rlm.metadata."""

from replm.metadata import (
    context_chunk_lengths,
    context_total_length,
    context_type_label,
    make_metadata,
)


class TestMakeMetadata:
    def test_short_content_returned_in_full(self):
        content = "Hello, world!"
        assert make_metadata(content, prefix_chars=1000) == content

    def test_exact_length_returned_in_full(self):
        content = "x" * 1000
        assert make_metadata(content, prefix_chars=1000) == content

    def test_truncation(self):
        content = "a" * 5000
        result = make_metadata(content, prefix_chars=100)
        assert "[Output: 5,000 chars total]" in result
        assert "First 100 chars:" in result
        assert "[truncated]" in result
        assert "a" * 100 in result

    def test_empty_content(self):
        assert make_metadata("", prefix_chars=100) == ""

    def test_custom_prefix_size(self):
        content = "Hello " * 100
        result = make_metadata(content, prefix_chars=10)
        assert "[truncated]" in result


class TestHelpers:
    def test_context_type_label_string(self):
        assert context_type_label("hello") == "string"

    def test_context_type_label_list(self):
        assert context_type_label(["a", "b"]) == "list of 2 strings"

    def test_context_total_length_string(self):
        assert context_total_length("hello") == 5

    def test_context_total_length_list(self):
        assert context_total_length(["abc", "de"]) == 5

    def test_context_chunk_lengths_string(self):
        assert context_chunk_lengths("hello") == [5]

    def test_context_chunk_lengths_list(self):
        assert context_chunk_lengths(["abc", "de"]) == [3, 2]
