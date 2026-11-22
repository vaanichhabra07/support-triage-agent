
"""
FastAPI backend for the Multi-Domain Support Triage Agent.

Endpoints:
  GET    /api/health
  GET    /api/companies
  POST   /api/companies/upload
  DELETE /api/companies/{name}
  POST   /api/triage
  POST   /api/triage/batch
"""

import io
import csv
import os

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from .agent import AgentServiceError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from . import retriever
from .pipeline import process_ticket


# --------------------------------------------------
# FastAPI Application
# --------------------------------------------------

app = FastAPI(
    title="Support Triage Agent API",
    version="1.0.0"
)
from fastapi.responses import JSONResponse


@app.exception_handler(AgentServiceError)
async def agent_service_error_handler(request, exc: AgentServiceError):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": str(exc),
            "error_type": "ai_service_error"
        }
    )


# --------------------------------------------------
# CORS Configuration
# --------------------------------------------------

# CORS Configuration

allowed_origins = os.environ.get(
    "ALLOWED_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173"
)

origins = [
    origin.strip()
    for origin in allowed_origins.split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# Startup - Seed Knowledge Base
# --------------------------------------------------

@app.on_event("startup")
def startup():
    retriever.seed_from_disk()


# --------------------------------------------------
# Request Model
# --------------------------------------------------

class TriageRequest(BaseModel):
    issue: str
    subject: str = ""
    company: str = ""


# --------------------------------------------------
# Health Check
# --------------------------------------------------

@app.get("/api/health")
def health():
    return {"status": "ok"}


# --------------------------------------------------
# List Companies
# --------------------------------------------------

@app.get("/api/companies")
def get_companies():
    return {
        "companies": retriever.list_companies()
    }


# --------------------------------------------------
# Upload Company Documents
# --------------------------------------------------

@app.post("/api/companies/upload")
async def upload_company(
    company: str = Form(...),
    files: list[UploadFile] = File(...)
):
    if not company.strip():
        raise HTTPException(
            status_code=400,
            detail="Company name is required."
        )

    docs = []

    for f in files:
        if not f.filename or not f.filename.lower().endswith((".md", ".txt")):
            continue

        content = (await f.read()).decode(
            "utf-8",
            errors="ignore"
        )

        docs.append((f.filename, content))

    if not docs:
        raise HTTPException(
            status_code=400,
            detail="No valid .md/.txt files were uploaded."
        )

    chunk_count = retriever.add_documents(company, docs)

    return {
        "company": retriever.slugify(company),
        "files_indexed": len(docs),
        "chunks_indexed": chunk_count,
    }


# --------------------------------------------------
# Delete Company Knowledge Base
# --------------------------------------------------

@app.delete("/api/companies/{name}")
def delete_company(name: str):

    deleted = retriever.delete_company(name)

    if deleted == 0:
        raise HTTPException(
            status_code=404,
            detail=f"No corpus found for '{name}'."
        )

    return {
        "company": retriever.slugify(name),
        "chunks_deleted": deleted
    }


# --------------------------------------------------
# Single Ticket Triage
# --------------------------------------------------

@app.post("/api/triage")
def triage(req: TriageRequest):

    if not req.issue.strip():
        raise HTTPException(
            status_code=400,
            detail="Issue text is required."
        )

    try:
        return process_ticket(
            req.issue,
            req.subject,
            req.company
        )

    except ValueError as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# --------------------------------------------------
# Batch CSV Triage
# --------------------------------------------------

@app.post("/api/triage/batch")
async def triage_batch(file: UploadFile = File(...)):

    """
    Accept a CSV containing:
    issue, subject, company

    Validate records and return a results CSV.
    """

    # Step 1: Validate file type
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload a CSV file."
        )

    # Step 2: Read uploaded CSV
    try:
        raw = (await file.read()).decode("utf-8-sig")
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=400,
            detail="Invalid CSV encoding. Please use UTF-8."
        )

    reader = csv.DictReader(io.StringIO(raw))

    # Step 3: Validate required columns
    required_columns = {
        "issue",
        "subject",
        "company"
    }

    actual_columns = set(reader.fieldnames or [])

    missing_columns = required_columns - actual_columns

    if missing_columns:
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid CSV. Missing required columns: "
                + ", ".join(sorted(missing_columns))
            )
        )

    # Step 4: Read all records
    tickets = list(reader)

    # Step 5: Reject empty CSV
    if not tickets:
        raise HTTPException(
            status_code=400,
            detail="Invalid CSV. No ticket records found."
        )

    # Step 6: Validate required fields
    required_fields = [
        "issue",
        "subject",
        "company"
    ]

    for row_number, ticket in enumerate(tickets, start=2):

        missing_fields = [
            field
            for field in required_fields
            if not isinstance(ticket.get(field), str)
            or not ticket[field].strip()
        ]

        if missing_fields:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Invalid CSV at row {row_number}. "
                    f"Missing or empty fields: "
                    f"{', '.join(missing_fields)}"
                )
            )

    # Step 7: Process valid tickets
    rows = []

    for index, row in enumerate(tickets, start=1):

        print(
            f"[BATCH] Processing ticket {index}/{len(tickets)}",
            flush=True
        )

        result = process_ticket(
            row["issue"],
            row["subject"],
            row["company"]
        )

        rows.append(result)

        print(
            f"[BATCH] Completed ticket {index}",
            flush=True
        )

    # Step 8: Generate output CSV
    output = io.StringIO()

    fieldnames = [
        "issue",
        "subject",
        "company",
        "response",
        "product_area",
        "status",
        "request_type",
        "justification",
        "confidence",
        "escalation_reason",
    ]

    writer = csv.DictWriter(
        output,
        fieldnames=fieldnames,
        extrasaction="ignore"
    )

    writer.writeheader()
    writer.writerows(rows)

    output.seek(0)

    # Step 9: Return downloadable CSV
    return StreamingResponse(
        io.BytesIO(
            output.getvalue().encode("utf-8")
        ),
        media_type="text/csv",
        headers={
            "Content-Disposition":
            "attachment; filename=triage_results.csv"
        },
    )
