
"""
Step 13: RAG regression tests.

Tests retrieval, company isolation, and confidence
without making any Gemini API calls.
"""

import uuid

from app import retriever


def check(condition, message):
    if not condition:
        raise AssertionError(message)
    print(f"PASS: {message}")


def main():
    # Unique names prevent interference with existing companies.
    suffix = uuid.uuid4().hex[:8]
    company_a = f"RegressionAlpha{suffix}"
    company_b = f"RegressionBeta{suffix}"

    # Unique facts make cross-company leakage easier to detect.
    fact_a = f"alpha-{suffix}-4821"
    fact_b = f"beta-{suffix}-7396"

    doc_a = (
        "Account recovery policy. "
        "Customers of this company must provide the "
        f"verification phrase {fact_a} to recover an account. "
        "This phrase is required for account recovery."
    )

    doc_b = (
        "Account recovery policy. "
        "Customers of this company must provide the "
        f"verification phrase {fact_b} to recover an account. "
        "This phrase is required for account recovery."
    )

    try:
        print("\nStep 13: RAG Regression Testing\n")

        # Test 1: Index two separate knowledge bases.
        retriever.add_documents(
            company_a,
            [("account_policy.md", doc_a)]
        )

        retriever.add_documents(
            company_b,
            [("account_policy.md", doc_b)]
        )

        companies = [
            c["name"] for c in retriever.list_companies()
        ]

        check(
            retriever.slugify(company_a) in companies,
            "Company A indexed successfully"
        )

        check(
            retriever.slugify(company_b) in companies,
            "Company B indexed successfully"
        )

        # Test 2: Retrieve Company A documentation.
        result_a = retriever.query_with_scores(
            "What verification phrase is required for account recovery?",
            company_a
        )

        context_a = result_a["context"]

        check(
            fact_a in context_a,
            "Company A retrieves its own knowledge"
        )

        check(
            fact_b not in context_a,
            "Company A does not retrieve Company B knowledge"
        )

        # Test 3: Retrieve Company B documentation.
        result_b = retriever.query_with_scores(
            "What verification phrase is required for account recovery?",
            company_b
        )

        context_b = result_b["context"]

        check(
            fact_b in context_b,
            "Company B retrieves its own knowledge"
        )

        check(
            fact_a not in context_b,
            "Company B does not retrieve Company A knowledge"
        )

        # Test 4: Validate retrieval confidence.
        for label, result in [
            ("Company A", result_a),
            ("Company B", result_b),
        ]:
            confidence = result["confidence"]

            check(
                isinstance(confidence, (int, float))
                and 0 <= confidence <= 1,
                f"{label} confidence is between 0 and 1"
            )

            print(
                f"INFO: {label} retrieval confidence = "
                f"{confidence:.4f}"
            )

        print("\nALL RAG REGRESSION TESTS PASSED")

    finally:
        # Remove only the temporary test companies.
        retriever.delete_company(company_a)
        retriever.delete_company(company_b)
        print("\nTemporary test companies cleaned up.")


if __name__ == "__main__":
    main()
