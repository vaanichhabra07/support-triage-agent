
"""
Step 14D: Pipeline regression and escalation reason tests.
No Gemini API calls required.
"""

from unittest.mock import patch
from app import pipeline


def fake_retrieval(confidence):
    return {
        "context": "Password reset instructions are available.",
        "confidence": confidence,
        "matches": [],
    }


def fake_agent_response():
    return {
        "status": "replied",
        "product_area": "account_access",
        "response": "Follow the password reset instructions.",
        "justification": "Answer found in documentation.",
        "request_type": "product_issue",
    }


def run_test(name, issue, confidence, expected_status, reason_type):
    with (
        patch.object(
            pipeline.retriever,
            "list_companies",
            return_value=[{"name": "hackerrank"}]
        ),
        patch.object(
            pipeline.retriever,
            "query_with_scores",
            return_value=fake_retrieval(confidence)
        ),
        patch.object(
            pipeline.agent,
            "process_ticket",
            return_value=fake_agent_response()
        ),
    ):
        result = pipeline.process_ticket(
            issue=issue,
            subject="Support request",
            company="HackerRank"
        )

    assert result["status"] == expected_status, (
        f"{name}: expected {expected_status}, got {result['status']}"
    )

    reason = result.get("escalation_reason")

    if reason_type == "none":
        assert reason is None, f"{name}: unexpected reason {reason}"

    elif reason_type == "risk":
        assert reason is not None
        assert "High-risk" in reason, f"{name}: {reason}"
        assert result["response"].strip() == "Escalate to a human"

    elif reason_type == "confidence":
        assert reason is not None
        assert "Low retrieval confidence" in reason, f"{name}: {reason}"
        assert result["response"].strip() == "Escalate to a human"

    print(f"PASS: {name}")
    print(f"  Status: {result['status']}")
    print(f"  Escalation reason: {reason}")


print("\nStep 14D: Escalation Reason Testing\n")

run_test(
    "Normal support issue",
    "How do I reset my password?",
    1.0,
    "replied",
    "none"
)

run_test(
    "Fraud escalation",
    "I suspect fraud on my account.",
    1.0,
    "escalated",
    "risk"
)

run_test(
    "Low-confidence escalation",
    "How do I reset my password?",
    0.0,
    "escalated",
    "confidence"
)

print("\nALL ESCALATION REASON TESTS PASSED")
