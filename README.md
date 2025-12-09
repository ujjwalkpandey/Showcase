# 🚀 Showcase

<div align="center">

**An Intelligent Portfolio Generator Powered by AI**

*Transforming resumes into stunning portfolio websites—effortlessly.*

[![MLSA](https://img.shields.io/badge/MLSA-KIIT%20Chapter-blue?style=for-the-badge)](https://studentambassadors.microsoft.com/)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)](https://reactjs.org/)

</div>

---

## 📖 About

**Showcase** is an intelligent web application that transforms your resume, project descriptions, and skill inventories into a sleek, personalized portfolio website. Powered by state-of-the-art LLMs, Showcase automatically curates your best work, suggests visually impactful layouts, and optimizes content presentation—allowing users to effortlessly showcase their accomplishments with a modern, high-impact digital presence tailored to their target industry, **without requiring any coding or design expertise**.

### ✨ Key Features

- 📄 **Smart Resume Processing** - Upload PDFs, images, or DOCX files with advanced OCR extraction using Hugging Face models
- 🤖 **AI-Powered Enhancement** - LLM-driven content curation and optimization
- 🎨 **Auto-Generated Layouts** - Beautiful, responsive portfolio designs tailored to your industry
- ⚡ **Real-Time Processing** - Track your portfolio generation progress in real-time
- 🚀 **One-Click Deployment** - Deploy your portfolio to Vercel with a single click
- 🔄 **Intelligent Validation** - Automatic content validation and auto-fix capabilities

---

## 🏗️ Architecture

```
[User Uploads Resume] → OCR Service (Hugging Face/Tesseract) → 
Structured JSON → Gemini Content Pass → Gemini Frontend Pass → 
Validation & Auto-fix → Preview / Download / Vercel Deploy
```

---

## 🛠️ Tech Stack

### Backend
- **FastAPI** - Modern, fast web framework for building APIs
- **PostgreSQL** - Robust relational database with SQLAlchemy ORM
- **Celery + Redis** - Distributed task queue for async processing
- **Alembic** - Database migration management

### Frontend
- **React 18** - Modern UI library
- **Vite** - Lightning-fast build tool
- **Tailwind CSS** - Utility-first CSS framework

### AI & Processing
- **Google Gemini** - State-of-the-art LLM for content generation
- **Hugging Face OCR Models** - Advanced OCR models for text extraction (primary)
- **Pytesseract** - OCR engine fallback for local processing
- **Agno Agents** - Intelligent orchestration framework

### Infrastructure
- **Docker Compose** - Containerized development environment
- **Vercel** - Deployment platform for generated portfolios

---

## 📋 Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.11+** - [Download Python](https://www.python.org/downloads/)
- **uv** (Recommended) - Fast Python package installer - [Install uv](https://github.com/astral-sh/uv#installation)
  - **Windows**: `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
  - **macOS/Linux**: `curl -LsSf https://astral.sh/uv/install.sh | sh`
  - Or use pip: `pip install uv`
- **Node.js 18+** - [Download Node.js](https://nodejs.org/)
- **Docker & Docker Compose** - [Install Docker](https://www.docker.com/get-started)
- **Tesseract OCR** - Optional fallback for local OCR processing (recommended)
  - **Windows**: `choco install tesseract` or [Download](https://github.com/UB-Mannheim/tesseract/wiki)
  - **macOS**: `brew install tesseract`
  - **Linux**: `sudo apt-get install tesseract-ocr`
  
> **Note**: Showcase primarily uses Hugging Face OCR models for text extraction. Tesseract is used as a fallback when Hugging Face models are unavailable or for offline processing.

> **Note**: `uv` is required for dependency installation. The setup scripts will automatically install `uv` if it's not found.

---

## 🚀 Quick Start

### 1️⃣ Clone the Repository

```bash
git clone <repository-url>
cd showcase
```

### 2️⃣ Install Dependencies

**Using uv (Automatic Installation):**
```bash
# Install Python dependencies with uv (uv will be installed automatically if needed)
make install
# Or manually:
uv pip install -e .
```

**Using pip (Fallback - Not Recommended):**
```bash
# Install Python dependencies with pip
pip install -e .
# Or: make install-pip
```

**Install Frontend Dependencies:**
```bash
# Install frontend generator dependencies
cd frontend_generator && npm install && cd ..

# Install frontend dependencies
make install-frontend
# Or: cd frontend && npm install && cd ..
```

### 3️⃣ Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env and add your API keys
# Required: GEMINI_API_KEY (get from https://makersuite.google.com/app/apikey)
# Optional: VERCEL_TOKEN, VERCEL_ORG_ID, VERCEL_PROJECT_ID
```

### 4️⃣ Start Infrastructure

```bash
# Start PostgreSQL and Redis
make dev-up

# Run database migrations
make upgrade
```

### 5️⃣ Start Services

Open **4 separate terminals**:

**Terminal 1 - Backend API:**
```bash
make run-backend
# API available at http://localhost:8000
```

**Terminal 2 - Celery Worker:**
```bash
make run-celery
# Processes background jobs
```

**Terminal 3 - Frontend UI:**
```bash
make run-frontend
# UI available at http://localhost:3001
```

**Terminal 4 - (Optional) Agent:**
```bash
make run-agent <resume_file.pdf>
# Or use the web UI instead
```

### 6️⃣ Access the Application

- **Web UI**: Open [http://localhost:3001](http://localhost:3001) in your browser
- **API Docs**: Visit [http://localhost:8000/docs](http://localhost:8000/docs) for interactive API documentation

---

## 📚 Usage Guide

### Using the Web Interface

1. **Upload Resume**
   - Navigate to the web UI at `http://localhost:3001`
   - Click "Upload Resume" and select your PDF, image, or DOCX file
   - Wait for processing to complete

2. **View Progress**
   - Monitor job status in real-time
   - View processing logs and AI interactions
   - Check generated artifacts

3. **Preview & Deploy**
   - Preview your generated portfolio
   - Download artifacts (JSON, bundle)
   - Deploy to Vercel with one click

### Using the API

#### Upload Resume

```bash
curl -X POST http://localhost:8000/api/v1/resumes/upload \
  -F "file=@your_resume.pdf"
```

Response:
```json
{
  "job_id": 1,
  "status": "pending",
  "message": "Resume uploaded and processing started"
}
```

#### Check Job Status

```bash
curl http://localhost:8000/api/v1/jobs/1
```

#### Deploy to Vercel

```bash
curl -X POST http://localhost:8000/api/v1/jobs/1/deploy
```

### Using the Agent Script

```bash
python agents/pipeline_agent.py path/to/resume.pdf --deploy
```

---

## 📁 Project Structure

```
showcase/
├── app/                          # Backend application
│   ├── api/                      # API routes and handlers
│   ├── ai_providers/             # AI adapter interfaces
│   ├── ocr/                      # OCR processing adapters
│   ├── frontend_generator/       # Frontend bundle generation
│   ├── models.py                 # Database models
│   ├── tasks.py                  # Celery background tasks
│   └── ai_pipeline.py            # Main processing pipeline
├── frontend/                     # React web UI
│   ├── src/
│   │   ├── components/           # React components
│   │   └── api/                  # API client
├── frontend_generator/           # Portfolio generator
│   └── generate.js               # Next.js bundle generator
├── agents/                       # Agent orchestration
│   └── pipeline_agent.py         # Pipeline agent example
├── alembic/                     # Database migrations
└── docker-compose.yml            # Infrastructure setup
```

For detailed structure, see [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md).

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/resumes/upload` | Upload resume file |
| `GET` | `/api/v1/jobs/{job_id}` | Get job status and artifacts |
| `GET` | `/preview/{job_id}` | View generated portfolio preview |
| `POST` | `/api/v1/jobs/{job_id}/deploy` | Deploy portfolio to Vercel |

Full API documentation available at `/docs` when the backend is running.

---

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the root directory:

```env
# Database
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/showcase_db

# Redis
REDIS_URL=redis://localhost:6379/0

# AI Provider (Required for AI features)
GEMINI_API_KEY=your_gemini_api_key_here

# Hugging Face (Optional - for OCR models)
# Models are downloaded automatically, but you can set a token for private models
HUGGINGFACE_API_TOKEN=your_huggingface_token_here

# Vercel Deployment (Required for deployment)
VERCEL_TOKEN=your_vercel_token_here
VERCEL_ORG_ID=your_vercel_org_id_here
VERCEL_PROJECT_ID=your_vercel_project_id_here

# Application
SECRET_KEY=your_secret_key_here
DEBUG=True
```

### Getting API Keys

- **Gemini API Key**: [Get from Google AI Studio](https://makersuite.google.com/app/apikey)
- **Hugging Face Token** (Optional): [Get from Hugging Face](https://huggingface.co/settings/tokens) - Only needed for private models
- **Vercel Token**: [Get from Vercel Dashboard](https://vercel.com/account/tokens)

---

## 🧪 Development

### Running Tests

```bash
make test
# Or: pytest
```

### Database Migrations

```bash
# Create new migration
make migrate msg="description"

# Apply migrations
make upgrade

# Rollback
make downgrade
```

### Available Make Commands

```bash
make help              # Show all available commands
make install          # Install Python dependencies using uv (installs uv if needed)
make install-pip       # Install Python dependencies using pip explicitly
make install-frontend # Install frontend dependencies
make dev-up           # Start Docker services
make dev-down         # Stop Docker services
make run-backend      # Start FastAPI server
make run-celery       # Start Celery worker
make run-frontend     # Start Vite dev server
make run-agent        # Run pipeline agent
make clean            # Clean generated files
```

---

## 🐛 Troubleshooting

### Port Already in Use

If port 3001 is in use, change it in `frontend/vite.config.js`:

```js
server: {
  port: 3002, // Change to available port
}
```

### OCR Processing Issues

**Hugging Face Models:**
- Hugging Face OCR models are downloaded automatically on first use
- Ensure you have internet connection for initial model download
- Models are cached locally after first download
- If models fail to load, the system will automatically fallback to Tesseract

**Tesseract Fallback:**
- Tesseract is optional but recommended as a fallback
- If Tesseract is not installed, OCR will rely solely on Hugging Face models
- To install Tesseract (optional):
  - Ensure Tesseract is installed and in your PATH
  - Verify installation: `tesseract --version`
  - Windows: Add Tesseract installation directory to PATH

### Database Connection Errors

- Check Docker containers: `docker-compose ps`
- Verify `DATABASE_URL` in `.env`
- Restart services: `docker-compose restart postgres`

### Celery Worker Not Processing

- Verify Redis is running: `docker-compose ps`
- Check `REDIS_URL` in `.env`
- Review Celery logs for errors

### GEMINI_API_KEY Not Set

- The pipeline will use mock responses for testing
- For real AI features, add your Gemini API key to `.env`

### uv Installation Issues

**Automatic Installation:**
- Setup scripts automatically install `uv` if not found
- If automatic installation fails, install manually:

**Windows:**
- If PowerShell execution policy blocks installation, run:
  ```powershell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
  ```
- Then install: `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
- Restart terminal after installation

**macOS/Linux:**
- Ensure you have curl installed: `curl --version`
- Install: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Add to PATH: `export PATH="$HOME/.cargo/bin:$PATH"` (add to `~/.bashrc` or `~/.zshrc`)

**Verification:**
- Check if uv is installed: `uv --version`
- If issues persist, restart your terminal to refresh PATH

---

## 🚧 Roadmap

- [ ] **Enhanced OCR Models** - Support for additional Hugging Face OCR models
- [ ] **Enhanced AI Models** - Support for multiple LLM providers
- [ ] **Custom Themes** - User-selectable portfolio themes
- [ ] **Analytics Integration** - Track portfolio views and engagement
- [ ] **Multi-language Support** - Generate portfolios in multiple languages
- [ ] **Export Options** - PDF, static HTML, and more export formats
- [ ] **Collaboration Features** - Share and collaborate on portfolios
- [ ] **Template Marketplace** - Community-driven portfolio templates

---

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines for more details.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👥 Team

**MLSA KIIT Chapter**

Built with ❤️ by the Microsoft Learn Student Ambassadors community at KIIT University.

---

## 📞 Support

- 📧 **Email**: [Your Email]
- 💬 **Discord**: [Your Discord Server]
- 🐛 **Issues**: [GitHub Issues](https://github.com/your-repo/issues)

---

<div align="center">

**Made with ❤️ by MLSA KIIT Chapter**

[⬆ Back to Top](#-showcase)

</div>
