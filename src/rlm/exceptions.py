"""Custom exceptions for the RLM library."""


class RLMError(Exception):
    """Base exception for all RLM errors."""


class MaxIterationsExceeded(RLMError):
    """Raised when the root REPL loop exceeds the maximum number of iterations."""

    def __init__(self, iterations: int, max_iterations: int) -> None:
        self.iterations = iterations
        self.max_iterations = max_iterations
        super().__init__(
            f"Reached {iterations}/{max_iterations} iterations without a final answer"
        )


class MaxSubCallsExceeded(RLMError):
    """Raised when the total number of sub-LLM calls exceeds the configured limit."""

    def __init__(self, call_count: int, max_sub_calls: int) -> None:
        self.call_count = call_count
        self.max_sub_calls = max_sub_calls
        super().__init__(
            f"Sub-call limit reached: {call_count}/{max_sub_calls}"
        )


class REPLExecutionError(RLMError):
    """Raised for unrecoverable errors during REPL code execution."""


class SandboxTimeout(RLMError):
    """Raised when REPL code execution exceeds the configured timeout."""

    def __init__(self, timeout: int) -> None:
        self.timeout = timeout
        super().__init__(f"REPL execution timed out after {timeout}s")
