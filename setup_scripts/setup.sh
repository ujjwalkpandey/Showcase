#!/bin/bash
# Setup script for local development

set -e

echo "🚀 Setting up Resume Processing Pipeline..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Check for uv and install if not found
if ! command -v uv &> /dev/null; then
    echo "📦 Installing uv..."
    if curl -LsSf https://astral.sh/uv/install.sh | sh; then
        # Add uv to PATH for current session
        export PATH="$HOME/.cargo/bin:$PATH"
        # Verify installation
        if ! command -v uv &> /dev/null; then
            echo "⚠️  uv installed but not in PATH. Please restart your terminal or run:"
            echo "   export PATH=\"\$HOME/.cargo/bin:\$PATH\""
            exit 1
        fi
    else
        echo "❌ Failed to install uv. Please install manually:"
        echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
        exit 1
    fi
fi

echo "✅ Using uv for faster installation"
echo "📦 Installing Python dependencies with uv..."
uv pip install -e .

# Install frontend generator dependencies
echo "📦 Installing frontend generator dependencies..."
cd frontend_generator
npm install
cd ..

# Create .env if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env from .env.example..."
    cp .env.example .env
    echo "⚠️  Please edit .env and add your API keys!"
fi

# Start Docker services
echo "🐳 Starting Docker services..."
docker-compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
sleep 5

# Run migrations
echo "🗄️  Running database migrations..."
alembic upgrade head

echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Edit .env and add your API keys"
echo "  2. Run: make run-backend (in one terminal)"
echo "  3. Run: make run-celery (in another terminal)"
echo "  4. Test: python agents/pipeline_agent.py <resume_file.pdf>"


