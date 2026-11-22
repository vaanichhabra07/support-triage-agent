
"""
Test FastAPI Gemini error handling without calling Gemini.
"""

from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app
from app.agent import AgentServiceError


client = TestClient(app, raise_server_exceptions=False)


def run_test(name, error_code, expected_status):
    fake_error = AgentServiceError(
        message="Simulated Gemini failure",
        status_code=error_code
    )

    with patch(
        "app.main.process_ticket",
        side_effect=fake_error
    ):
        response = client.post(
            "/api/triage",
            json={
                "issue": "Unable to log in",
                "subject": "Login problem",
                "company": "HackerRank"
            }
        )

    if response.status_code != expected_status:
        print(
            f"FAIL: {name} - Expected HTTP {expected_status}, "
            f"got {response.status_code}"
        )
        print(response.text)
        return

    data = response.json()

    if data.get("error_type") != "ai_service_error":
        print(
            f"FAIL: {name} - Incorrect error response: {data}"
        )
        return

    print(f"PASS: {name} -> HTTP {response.status_code}")


print("\nTesting FastAPI error handling...\n")

run_test("Gemini quota exceeded", 429, 429)
run_test("Gemini unavailable", 503, 503)
run_test("Gemini invalid response", 502, 502)

print("\nTesting completed.")
