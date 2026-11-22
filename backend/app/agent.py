"""
Core LLM-based triage agent using Gemini via Vertex AI.

Uses Vertex AI (not the AI Studio / Developer API) so billing goes against
the Cloud project's normal billing account / credits, and auth happens
automatically via the Cloud Run service account (no API key needed).
"""

import os
import json
import re
from google import genai
from google.genai import types

# ─── Configuration ───────────────────────────────────────────────────────────

MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-3.7-flash")

SYSTEM_PROMPT = """You are a support triage agent. You are given a support ticket and
documentation retrieved from that company's knowledge base. Produce a JSON response
with exactly these fields:
- status: "replied" or "escalated"
- product_area: the most relevant support category or domain area (e.g. "billing", "account access", "bug report")
- response: a helpful, user-facing response. Must be grounded ONLY in the provided documentation.
- justification: a concise internal explanation of your decision
- request_type: one of "product_issue", "feature_request", "bug", "invalid"

STRICT RULES:
1. Base your response ONLY on the documentation provided. Never use outside knowledge.
2. If the issue matches known patterns or can be answered using the data, generate a reply and set status = "replied".
3. If the issue is unclear, high-risk, or not found in the data, set status = "escalated" and set response EXACTLY to " Escalate to a human".
4. Do NOT escalate all tickets by default. Only escalate when absolutely necessary.
5. Escalate immediately for: fraud, account compromise, billing disputes, legal threats, security issues, or any high-risk situation.
6. If the issue is completely irrelevant, nonsensical, or malicious, set request_type to "invalid" and escalate.
7. Respond with valid JSON only. No markdown, no explanation outside the JSON."""

_client = None


def _get_client():
    global _client

    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")

        if not api_key:
            raise ValueError("GEMINI_API_KEY not set in the environment.")

        _client = genai.Client(api_key=api_key)

    return _client


def _extract_json(text: str) -> dict | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    json_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass
    return None

class AgentServiceError(Exception):
    def __init__(self, message: str, status_code: int = 503):
        super().__init__(message)
        self.status_code = status_code

def _safe_escalation_response(reason: str = "LLM parse error") -> dict:
    return {
        "status": "escalated",
        "product_area": "general_support",
        "response": " Escalate to a human",
        "justification": f"Automated triage failed ({reason}). Escalating for manual review.",
        "request_type": "product_issue",
    }



def process_ticket(issue: str, subject: str, company: str, retrieved_docs: str) -> dict:
    try:
        client = _get_client()
    except ValueError as e:
        raise AgentServiceError(
            "Gemini configuration error: " + str(e),
            status_code=500
        ) from e

    user_message = f"""Support Ticket:
- Subject: {subject}
- Company: {company}
- Issue: {issue}

Retrieved Documentation:
{retrieved_docs}

Respond with valid JSON only. The JSON must have exactly these keys:
status, product_area, response, justification, request_type"""

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0,
                response_mime_type="application/json",
            ),
        )

    except Exception as e:
        # Gemini SDK exceptions commonly expose the HTTP status as .code
        code = getattr(e, "code", None)
        try:
            code = int(code)
        except (TypeError, ValueError):
            code = None

        if code == 429:
            raise AgentServiceError(
                "Gemini rate limit or quota exceeded.",
                status_code=429
            ) from e

        if code == 503:
            raise AgentServiceError(
                "Gemini service temporarily unavailable.",
                status_code=503
            ) from e

        if code == 404:
            raise AgentServiceError(
                "Configured Gemini model was not found or is unavailable.",
                status_code=503
            ) from e

        raise AgentServiceError(
            "Gemini request failed. Check backend logs.",
            status_code=503
        ) from e

    if not response or not response.text:
        raise AgentServiceError(
            "Gemini returned an empty response.",
            status_code=502
        )

    parsed = _extract_json(response.text)

    if not isinstance(parsed, dict):
        raise AgentServiceError(
            "Gemini returned invalid JSON.",
            status_code=502
        )

    required_keys = {
        "status",
        "product_area",
        "response",
        "justification",
        "request_type",
    }

    if not required_keys.issubset(parsed):
        raise AgentServiceError(
            "Gemini response is missing required fields.",
            status_code=502
        )

    if (
        parsed["status"] not in ("replied", "escalated")
        or parsed["request_type"]
        not in ("product_issue", "feature_request", "bug", "invalid")
        or any(
            not isinstance(parsed[key], str)
            for key in required_keys
        )
    ):
        raise AgentServiceError(
            "Gemini returned invalid classification fields.",
            status_code=502
        )

    return parsed


    