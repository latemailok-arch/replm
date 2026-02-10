"""Tests for the sub-call result cache."""

from __future__ import annotations

from typing import Any

from rlm.cache import CacheStats, SubCallCache
from rlm.client import CompletionResult
from rlm.config import RLMConfig
from rlm.orchestrator import Orchestrator
from rlm.types import RLMEvent

# ---------------------------------------------------------------------------
# Unit tests for SubCallCache
# ---------------------------------------------------------------------------


class TestSubCallCacheBasic:
    def test_put_and_get(self):
        cache = SubCallCache()
        cache.put("k1", "value1")
        assert cache.get("k1") == "value1"

    def test_get_miss_returns_none(self):
        cache = SubCallCache()
        assert cache.get("nonexistent") is None

    def test_key_is_deterministic(self):
        k1 = SubCallCache.key("model", "prompt", 0.5)
        k2 = SubCallCache.key("model", "prompt", 0.5)
        assert k1 == k2

    def test_different_models_different_keys(self):
        k1 = SubCallCache.key("model-a", "prompt", 0.5)
        k2 = SubCallCache.key("model-b", "prompt", 0.5)
        assert k1 != k2

    def test_different_temperatures_different_keys(self):
        k1 = SubCallCache.key("model", "prompt", 0.3)
        k2 = SubCallCache.key("model", "prompt", 0.7)
        assert k1 != k2

    def test_different_prompts_different_keys(self):
        k1 = SubCallCache.key("model", "prompt A", 0.5)
        k2 = SubCallCache.key("model", "prompt B", 0.5)
        assert k1 != k2

    def test_overwrite_existing_key(self):
        cache = SubCallCache()
        cache.put("k", "v1")
        cache.put("k", "v2")
        assert cache.get("k") == "v2"


class TestSubCallCacheStats:
    def test_initial_stats(self):
        cache = SubCallCache()
        s = cache.stats
        assert s.hits == 0
        assert s.misses == 0
        assert s.size == 0

    def test_hit_tracking(self):
        cache = SubCallCache()
        cache.put("k", "v")
        cache.get("k")
        cache.get("k")
        s = cache.stats
        assert s.hits == 2
        assert s.misses == 0
        assert s.size == 1

    def test_miss_tracking(self):
        cache = SubCallCache()
        cache.get("missing1")
        cache.get("missing2")
        s = cache.stats
        assert s.hits == 0
        assert s.misses == 2

    def test_size_tracking(self):
        cache = SubCallCache()
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        assert cache.stats.size == 3


class TestSubCallCacheLRU:
    def test_eviction_at_max_size(self):
        cache = SubCallCache(max_size=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.put("k4", "v4")  # should evict k1
        assert cache.get("k1") is None
        assert cache.get("k4") == "v4"
        assert cache.stats.size == 3

    def test_lru_access_prevents_eviction(self):
        cache = SubCallCache(max_size=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.get("k1")  # touch k1 — now k2 is LRU
        cache.put("k4", "v4")  # should evict k2
        assert cache.get("k1") == "v1"
        assert cache.get("k2") is None
        assert cache.get("k4") == "v4"

    def test_max_size_one(self):
        cache = SubCallCache(max_size=1)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        assert cache.get("k1") is None
        assert cache.get("k2") == "v2"
        assert cache.stats.size == 1


class TestCacheStatsDataclass:
    def test_fields(self):
        s = CacheStats(hits=10, misses=5, size=3)
        assert s.hits == 10
        assert s.misses == 5
        assert s.size == 3


# ---------------------------------------------------------------------------
# Integration tests: caching with the orchestrator
# ---------------------------------------------------------------------------


class _MockClient:
    """Mock LLM client that tracks call count."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        **kwargs: Any,
    ) -> CompletionResult:
        self.calls.append(
            {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        text = self._responses.pop(0) if self._responses else "FINAL(fallback)"
        return CompletionResult(content=text, input_tokens=100, output_tokens=50)


class TestCacheIntegration:
    def test_same_prompt_twice_one_api_call(self):
        """Two identical llm_query calls should produce only one sub-call API hit."""
        client = _MockClient(
            [
                # Root iteration 1: code calling llm_query twice with same prompt
                (
                    "```repl\n"
                    'r1 = llm_query("what is 2+2")\n'
                    'r2 = llm_query("what is 2+2")\n'
                    "print(r1, r2)\n"
                    "```"
                ),
                # Sub-call response (only one needed with cache)
                "four",
                # Root iteration 2: FINAL
                "FINAL(done)",
            ]
        )
        config = RLMConfig(
            max_iterations=5,
            max_recursion_depth=0,
            cache_sub_calls=True,
        )
        orch = Orchestrator(client, config, "m", "m")
        resp = orch.run("q", "ctx")
        assert resp.answer == "done"
        assert resp.sub_calls == 1  # only 1 budget call
        assert resp.cache_hits == 1  # second call was cached

    def test_cache_disabled_by_default(self):
        """Without cache_sub_calls=True, no caching occurs."""
        client = _MockClient(
            [
                (
                    "```repl\n"
                    'r1 = llm_query("what is 2+2")\n'
                    'r2 = llm_query("what is 2+2")\n'
                    "print(r1, r2)\n"
                    "```"
                ),
                "four",  # sub-call 1
                "four",  # sub-call 2 (no cache)
                "FINAL(done)",
            ]
        )
        config = RLMConfig(
            max_iterations=5,
            max_recursion_depth=0,
            cache_sub_calls=False,
        )
        orch = Orchestrator(client, config, "m", "m")
        resp = orch.run("q", "ctx")
        assert resp.answer == "done"
        assert resp.sub_calls == 2
        assert resp.cache_hits == 0

    def test_cache_hits_in_response(self):
        """RLMResponse.cache_hits reflects the actual hit count."""
        client = _MockClient(
            [
                (
                    "```repl\n"
                    'r1 = llm_query("a")\n'
                    'r2 = llm_query("b")\n'
                    'r3 = llm_query("a")\n'
                    'r4 = llm_query("b")\n'
                    'r5 = llm_query("a")\n'
                    "```"
                ),
                "resp_a",  # sub-call for "a"
                "resp_b",  # sub-call for "b"
                # "a" again -> cache hit, "b" again -> cache hit, "a" again -> cache hit
                "FINAL(done)",
            ]
        )
        config = RLMConfig(
            max_iterations=5,
            max_recursion_depth=0,
            cache_sub_calls=True,
        )
        orch = Orchestrator(client, config, "m", "m")
        resp = orch.run("q", "ctx")
        assert resp.sub_calls == 2  # only 2 actual API calls
        assert resp.cache_hits == 3  # 3 cache hits

    def test_different_prompts_not_cached(self):
        """Different prompts produce separate API calls."""
        client = _MockClient(
            [
                ('```repl\nr1 = llm_query("alpha")\nr2 = llm_query("beta")\n```'),
                "resp_alpha",
                "resp_beta",
                "FINAL(done)",
            ]
        )
        config = RLMConfig(
            max_iterations=5,
            max_recursion_depth=0,
            cache_sub_calls=True,
        )
        orch = Orchestrator(client, config, "m", "m")
        resp = orch.run("q", "ctx")
        assert resp.sub_calls == 2
        assert resp.cache_hits == 0


class TestCacheEvent:
    def test_cache_hit_event_fired(self):
        """A sub_call_cache_hit event fires on cache hits."""
        events: list[RLMEvent] = []
        client = _MockClient(
            [
                ('```repl\nr1 = llm_query("same")\nr2 = llm_query("same")\n```'),
                "response",
                "FINAL(done)",
            ]
        )
        config = RLMConfig(
            max_iterations=5,
            max_recursion_depth=0,
            cache_sub_calls=True,
        )
        orch = Orchestrator(client, config, "m", "m")
        orch.run("q", "ctx", on_event=events.append)
        cache_events = [e for e in events if e.type == "sub_call_cache_hit"]
        assert len(cache_events) == 1

    def test_no_cache_hit_event_when_disabled(self):
        """No cache hit events when caching is disabled."""
        events: list[RLMEvent] = []
        client = _MockClient(
            [
                ('```repl\nr1 = llm_query("same")\nr2 = llm_query("same")\n```'),
                "response1",
                "response2",
                "FINAL(done)",
            ]
        )
        config = RLMConfig(
            max_iterations=5,
            max_recursion_depth=0,
            cache_sub_calls=False,
        )
        orch = Orchestrator(client, config, "m", "m")
        orch.run("q", "ctx", on_event=events.append)
        cache_events = [e for e in events if e.type == "sub_call_cache_hit"]
        assert len(cache_events) == 0
