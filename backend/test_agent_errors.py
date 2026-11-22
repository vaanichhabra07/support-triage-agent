
"""
Test Gemini API failure handling without making real API calls.
"""

import os
from unittest.mock import MagicMock, patch

# Set a dummy key before importing the agent.
os.environ["GEMINI_API_KEY"] = "test-key"

from app import agent
from app.agent import AgentServiceError


def run_test(name, fake_error, expected_status):
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = fake_error

    try:
        with patch.object(agent, "_get_client", return_value=mock_client):
            agent.process_ticket(
                issue="Cannot log in",
                subject="Login problem",
                company="HackerRank",
                retrieved_docs="Password reset documentation"
            )

        print(f"FAIL: {name} - No exception raised")

    except AgentServiceError as exc:
        if exc.status_code == expected_status:
            print(f"PASS: {name} -> HTTP {exc.status_code}")
        else:
            print(
                f"FAIL: {name} -> Expected {expected_status}, "
                f"got {exc.status_code}"
            )


class FakeGeminiError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


print("\nTesting Gemini API error handling...\n")

run_test(
    "Gemini quota exceeded",
    FakeGeminiError(429, "Resource exhausted"),
    429
)

run_test(
    "Gemini service unavailable",
    FakeGeminiError(503, "Service unavailable"),
    503
)

run_test(
    "Gemini model unavailable",
    FakeGeminiError(404, "Model not found"),
    503
)

run_test(
    "Unexpected Gemini failure",
    FakeGeminiError(500, "Internal error"),
    503
)

print("\nTesting completed.")
