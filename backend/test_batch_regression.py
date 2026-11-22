
"""
Step 15: Batch CSV regression tests.

Tests:
1. Valid CSV processes all tickets.
2. Output CSV contains expected columns.
3. Gemini failure produces an HTTP error, not a fake result CSV.

No real Gemini API calls.
"""

import io
import csv
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app
from app.agent import AgentServiceError


client = TestClient(app, raise_server_exceptions=False)


def make_csv():
    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow(["issue", "subject", "company"])
    writer.writerow(["Cannot log in", "Login issue", "HackerRank"])
    writer.writerow(["Payment failed", "Billing issue", "Visa"])
    writer.writerow(["Reset password", "Account issue", "Claude"])

    return output.getvalue().encode("utf-8")


def fake_process_ticket(issue, subject, company):
    return {
        "issue": issue,
        "subject": subject,
        "company": company,
        "response": "Follow the documented support instructions.",
        "product_area": "general_support",
        "status": "replied",
        "request_type": "product_issue",
        "justification": "Answer found in knowledge base.",
        "confidence": 0.95,
        "escalation_reason": None,
        "retrieved_sources": [],
    }


def test_successful_batch():
    with patch("app.main.process_ticket", side_effect=fake_process_ticket) as mock:
        response = client.post(
            "/api/triage/batch",
            files={
                "file": (
                    "valid_tickets.csv",
                    make_csv(),
                    "text/csv"
                )
            }
        )

    assert response.status_code == 200, response.text
    assert "text/csv" in response.headers["content-type"]
    assert mock.call_count == 3

    reader = csv.DictReader(io.StringIO(response.text))
    rows = list(reader)

    assert len(rows) == 3, f"Expected 3 rows, got {len(rows)}"

    required_columns = {
        "issue", "subject", "company", "response",
        "product_area", "status", "request_type",
        "justification", "confidence", "escalation_reason"
    }

    assert required_columns.issubset(set(reader.fieldnames or []))

    assert rows[0]["issue"] == "Cannot log in"
    assert rows[1]["company"] == "Visa"
    assert rows[2]["company"] == "Claude"

    print("PASS: Valid CSV returns HTTP 200")
    print("PASS: All 3 tickets processed")
    print("PASS: Output CSV contains required columns")
    print("PASS: Ticket data preserved")


def test_gemini_failure():
    fake_error = AgentServiceError(
        "Gemini service temporarily unavailable.",
        status_code=503
    )

    with patch("app.main.process_ticket", side_effect=fake_error):
        response = client.post(
            "/api/triage/batch",
            files={
                "file": (
                    "valid_tickets.csv",
                    make_csv(),
                    "text/csv"
                )
            }
        )

    assert response.status_code == 503, response.text

    data = response.json()
    assert data["error_type"] == "ai_service_error"
    assert "text/csv" not in response.headers["content-type"]

    print("PASS: Gemini failure returns HTTP 503")
    print("PASS: No misleading results CSV generated")


if __name__ == "__main__":
    print("\nStep 15: Batch Regression Testing\n")

    test_successful_batch()
    test_gemini_failure()

    print("\nALL BATCH REGRESSION TESTS PASSED")
