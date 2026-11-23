# Multi-Domain AI Support Triage Agent

**A full-stack, RAG-powered customer support automation system with intelligent ticket classification, confidence-aware retrieval, and human escalation.**

Originally developed for **HackerRank Orchestrate 2026 (Top 2% — Rank 284 out of 12,885 participants)**, the project has been extended into a deployed, multi-tenant web application.

The system combines **Google Gemini, ChromaDB, Retrieval-Augmented Generation (RAG), FastAPI, and React** to process customer support tickets, retrieve company-specific knowledge, generate grounded responses, and escalate complex or uncertain queries to human agents.

### Live Application

- **Live Demo:** https://support-triage-agent-henna.vercel.app/
- **Backend API:** https://support-triage-backend.onrender.com
- **Interactive API Documentation:** https://support-triage-backend.onrender.com/docs
- **GitHub Repository:** https://github.com/vaanichhabra07/support-triage-agent

## Key Features

### 1. Intelligent Ticket Classification

- Analyzes incoming customer support requests.
- Detects safety-sensitive issues and escalation keywords.
- Identifies the relevant company knowledge base.
- Supports automatic company inference when no company is selected.

### 2. Retrieval-Augmented Generation (RAG)

- Uses **Gemini Embedding API** to generate semantic embeddings.
- Stores and retrieves document embeddings using **ChromaDB**.
- Retrieves relevant information from company-specific knowledge bases.
- Provides retrieved context to the Gemini language model to generate more grounded responses.

### 3. Confidence-Aware Human Escalation

- Uses retrieval confidence as a grounding signal.
- Supports a configurable confidence escalation threshold, documented as `0.35`.
- Routes insufficiently grounded or sensitive queries toward human assistance.
- Helps reduce unsupported AI-generated responses.

### 4. Multi-Tenant Knowledge Base Management

- Supports independent knowledge collections for multiple companies.
- Includes example corpora for **HackerRank, Claude, and Visa**.
- Allows uploading `.txt` and `.md` knowledge documents.
- Supports runtime corpus management without hardcoding every company.

### 5. AI-Powered Response Generation

- Integrates Google Gemini for contextual ticket processing.
- Combines ticket details with retrieved knowledge.
- Produces structured triage outcomes.
- Supports AI responses and human escalation decisions.

### 6. Batch Ticket Processing

- Supports processing multiple tickets through CSV uploads.
- Exposes batch-processing functionality through the FastAPI backend.
- Enables bulk evaluation and support workflow automation.

### 7. Full-Stack Web Application

- Responsive React interface built using Vite.
- FastAPI backend exposing REST endpoints.
- Interactive API testing through Swagger UI.
- Separate cloud deployments for frontend and backend.

## Technology Stack

| Category | Technologies |
|---|---|
| Frontend | React, JavaScript, Vite |
| Backend | Python, FastAPI |
| LLM | Google Gemini |
| Embedding Model | Gemini Embedding API (`gemini-embedding-001`) |
| Vector Database | ChromaDB |
| AI Architecture | Retrieval-Augmented Generation (RAG) |
| API Architecture | REST |
| Backend Hosting | Render |
| Frontend Hosting | Vercel |
| Version Control | Git, GitHub |

## System Architecture

```text
                   CUSTOMER SUPPORT TICKET
                             |
                             v
                   REACT FRONTEND (Vercel)
                             |
                             v
                    FASTAPI BACKEND (Render)
                             |
                             v
                      TICKET CLASSIFIER
                   /                    \
                  v                      v
         Safety / Request Type      Company Inference
                  \                      /
                   \                    /
                             v
                       RAG RETRIEVER
                             |
                             v
                    Gemini Embedding API
                             |
                             v
                       ChromaDB Search
                             |
                             v
                    Relevant Documents
                             |
                             v
                     Confidence Scoring
                             |
                             v
                        Gemini LLM
                             |
                             v
                     Response Validation
                             |
                   +---------+---------+
                   |                   |
                   v                   v
              AI RESPONSE        HUMAN ESCALATION
```

## Processing Workflow

**Step 1 — Ticket Submission**

The user submits a support ticket containing a subject, issue description, and optionally a company name.

**Step 2 — Classification**

The classifier examines the ticket, identifies the company context, and checks for conditions requiring escalation.

**Step 3 — Knowledge Retrieval**

The retriever generates a query embedding using Gemini and searches the appropriate ChromaDB collection for relevant document chunks.

**Step 4 — Confidence Evaluation**

The system evaluates retrieval confidence to estimate how well the retrieved documents support the ticket.

**Step 5 — AI Processing**

The Gemini language model processes the ticket using the retrieved context.

**Step 6 — Validation and Escalation**

The pipeline validates the result and determines whether an AI-generated response is appropriate or human intervention is required.

**Step 7 — Result Display**

The frontend displays the triage status, generated response, confidence information, and retrieved sources where available.

## Deployment Optimization

The original implementation used **Sentence-Transformers** for local embedding generation.

For deployment on Render's memory-constrained free tier, the embedding architecture was updated to use the **Gemini Embedding API**.

### Before vs. After

| Component | Original | Deployed |
|---|---|---|
| Embedding generation | Local Sentence-Transformers | Gemini Embedding API |
| Embedding inference | Local CPU and RAM | External API |
| Vector storage | ChromaDB | ChromaDB |
| RAG retrieval | Supported | Retained |
| Confidence-based escalation | Supported | Retained |
| Backend hosting | Local / cloud deployment configuration | Render |
| Frontend hosting | Local / deployable | Vercel |

This change reduces local model-loading overhead while retaining the core RAG workflow.

## Project Structure

```text
support-triage-agent/
|
|-- backend/
|   |-- app/
|   |   |-- main.py          # FastAPI routes
|   |   |-- pipeline.py      # Triage orchestration
|   |   |-- classifier.py    # Ticket classification
|   |   |-- retriever.py     # ChromaDB + Gemini embeddings
|   |   |-- agent.py         # Gemini LLM integration
|   |   |-- seed_data/       # Sample company knowledge bases
|   |
|   |-- requirements.txt
|   |-- .env.example
|
|-- frontend/
|   |-- src/                # React application
|   |-- package.json
|   |-- vercel.json
|   |-- .env.example
|
|-- code/                   # Original project code
|-- data/                   # Project data
|-- support_tickets/        # Ticket-related files
|-- render.yaml             # Render configuration
|-- README.md
```

## Local Installation

### Prerequisites

- Python 3.11
- Node.js and npm
- Git
- Google Gemini API key

### 1. Clone the Repository

```bash
git clone https://github.com/vaanichhabra07/support-triage-agent.git
cd support-triage-agent
```

### 2. Backend Setup

Navigate to the backend:

```bash
cd backend
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the virtual environment.

**Windows PowerShell:**

```powershell
.\venv\Scripts\Activate.ps1
```

**macOS/Linux:**

```bash
source venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Create a `.env` file using `.env.example` as a reference:

```env
GEMINI_API_KEY=your_gemini_api_key
ALLOWED_ORIGINS=http://localhost:5173
CHROMA_DATA_DIR=./chroma_data
```

Start the FastAPI server:

```bash
uvicorn app.main:app --reload
```

Backend:

http://localhost:8000

Swagger API documentation:

http://localhost:8000/docs

The application initializes the bundled sample company knowledge bases when needed.

### 3. Frontend Setup

Open a second terminal from the project root:

```bash
cd frontend
npm install
```

Create a `.env.local` file:

```env
VITE_API_URL=http://localhost:8000
```

Start the development server:

```bash
npm run dev
```

Open the URL displayed by Vite, typically:

http://localhost:5173

## REST API Reference

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | Backend health check |
| GET | `/api/companies` | List indexed company corpora |
| POST | `/api/companies/upload` | Upload company knowledge documents |
| GET | `/api/companies/{name}` | Retrieve company corpus information |
| DELETE | `/api/companies/{name}` | Delete a company corpus |
| POST | `/api/triage` | Process a single support ticket |
| POST | `/api/triage/batch` | Process tickets from CSV |

### Example: Submit a Support Ticket

**Endpoint:**

```http
POST /api/triage
Content-Type: application/json
```

**Request Body:**

```json
{
  "subject": "Assessment Time Extension",
  "issue": "How can I extend the assessment time for a candidate?",
  "company": "hackerrank"
}
```

The API processes the request through the classification, retrieval, and AI triage pipeline, returning a structured result.

For the exact response schema, refer to the interactive Swagger documentation.

## Production Deployment

### Backend — Render

The FastAPI application is hosted on Render.

**Production URL:**

https://support-triage-backend.onrender.com

Deployment configuration:

- Python backend service
- Dependencies installed from `backend/requirements.txt`
- Gemini API key stored securely as an environment variable
- ChromaDB knowledge base initialization
- CORS configured for the frontend domain

Example environment variables:

```env
GEMINI_API_KEY=your_secret_key
ALLOWED_ORIGINS=https://support-triage-agent-henna.vercel.app
```

### Frontend — Vercel

The React application is hosted on Vercel.

**Production URL:**

https://support-triage-agent-henna.vercel.app/

Build configuration:

```text
Framework: Vite
Root Directory: frontend
Build Command: npm run build
Output Directory: dist
```

Frontend environment variable:

```env
VITE_API_URL=https://support-triage-backend.onrender.com
```

**Security:** Gemini API credentials are stored only on the backend. They must not be exposed in the React frontend or committed to version control.

## Deployment Limitations

The application currently uses free cloud hosting and an external AI API.

- **Cold starts:** Render may temporarily suspend inactive services, increasing the first request's response time.
- **Gemini availability:** API quotas, rate limits, and temporary service failures can interrupt AI processing.
- **Ephemeral storage:** Uploaded company documents and ChromaDB indexes may be lost after a Render restart or redeployment. Bundled example corpora can be reindexed.
- **Human escalation:** Escalation is represented as a triage outcome; direct integration with an external support-ticketing platform is not yet implemented.

## Future Enhancements

- **Feedback Loop:** Collect user and support-agent feedback to improve classification and response quality.
- **Analytics Dashboard:** Track ticket categories, confidence scores, escalation rates, and processing trends.
- **Persistent Vector Storage:** Preserve uploaded company knowledge across deployments.
- **Support Platform Integration:** Connect to Zendesk, Freshdesk, or similar systems for real human-agent handoff.
- **Authentication:** Introduce secure user accounts and tenant-level access controls.
- **Reliability Improvements:** Add retry handling, quota-aware request management, and automated integration tests.

## Project Highlights

This project demonstrates practical experience in:

- Designing and deploying full-stack applications
- Building Retrieval-Augmented Generation systems
- Integrating large language models and embedding APIs
- Implementing semantic search with vector databases
- Developing confidence-aware AI workflows
- Designing REST APIs using FastAPI
- Building interactive interfaces with React
- Deploying applications using Render and Vercel
- Managing API credentials and cross-origin communication

## Project Owner

**Vaani Chhabra**

GitHub: https://github.com/vaanichhabra07

Project Repository: https://github.com/vaanichhabra07/support-triage-agent

Live Demo: https://support-triage-agent-henna.vercel.app/

---

**Multi-Domain AI Support Triage Agent** — Combining Retrieval-Augmented Generation, semantic search, and confidence-aware escalation to build more reliable AI-powered customer support workflows.