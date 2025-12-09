# PowerShell setup script for Windows

Write-Host "🚀 Setting up Resume Processing Pipeline..." -ForegroundColor Green

# Check Python version
$pythonVersion = python --version 2>&1
Write-Host "Python version: $pythonVersion"

# Check for uv and install if not found
$uvInstalled = Get-Command uv -ErrorAction SilentlyContinue
if (-not $uvInstalled) {
    Write-Host "📦 Installing uv..." -ForegroundColor Yellow
    try {
        powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
        # Refresh PATH to make uv available
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
        # Try to find uv in common locations
        $uvPath = "$env:USERPROFILE\.cargo\bin\uv.exe"
        if (Test-Path $uvPath) {
            $env:Path = "$env:USERPROFILE\.cargo\bin;$env:Path"
        }
        # Verify installation
        $uvInstalled = Get-Command uv -ErrorAction SilentlyContinue
        if (-not $uvInstalled) {
            Write-Host "⚠️  uv installed but not in PATH. Please restart your terminal." -ForegroundColor Yellow
            Write-Host "   Or add $env:USERPROFILE\.cargo\bin to your PATH manually." -ForegroundColor Yellow
            exit 1
        }
    } catch {
        Write-Host "❌ Failed to install uv. Please install manually:" -ForegroundColor Red
        Write-Host "   powershell -ExecutionPolicy ByPass -c `"irm https://astral.sh/uv/install.ps1 | iex`"" -ForegroundColor Yellow
        exit 1
    }
}

Write-Host "✅ Using uv for faster installation" -ForegroundColor Green
Write-Host "📦 Installing Python dependencies with uv..." -ForegroundColor Yellow
uv pip install -e .

# Install frontend generator dependencies
Write-Host "📦 Installing frontend generator dependencies..." -ForegroundColor Yellow
Set-Location frontend_generator
npm install
Set-Location ..

# Create .env if it doesn't exist
if (-not (Test-Path .env)) {
    Write-Host "📝 Creating .env from .env.example..." -ForegroundColor Yellow
    Copy-Item .env.example .env
    Write-Host "⚠️  Please edit .env and add your API keys!" -ForegroundColor Red
}

# Start Docker services
Write-Host "🐳 Starting Docker services..." -ForegroundColor Yellow
docker-compose up -d

# Wait for services to be ready
Write-Host "⏳ Waiting for services to be ready..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

# Run migrations
Write-Host "🗄️  Running database migrations..." -ForegroundColor Yellow
alembic upgrade head

Write-Host "✅ Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Edit .env and add your API keys"
Write-Host "  2. Run: make run-backend (in one terminal)"
Write-Host "  3. Run: make run-celery (in another terminal)"
Write-Host "  4. Test: python agents/pipeline_agent.py <resume_file.pdf>"


