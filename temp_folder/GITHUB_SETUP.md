# GitHub Setup Guide

This guide will help you set up your GitHub repository with the necessary secrets and configurations for the CI/CD pipeline.

## 🔧 Repository Setup

### 1. Create GitHub Repository

1. Go to [GitHub](https://github.com) and create a new repository
2. Name it `movie-recommender` (or your preferred name)
3. Make it public or private (your choice)
4. Don't initialize with README (we already have one)

### 2. Push Your Code

```bash
# Initialize git and push to GitHub
git init
git add .
git commit -m "Initial production-ready commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/movie-recommender.git
git push -u origin main
```

## 🔐 GitHub Secrets Configuration

### Required Secrets

These secrets are needed for the CI/CD pipeline to work properly:

#### 1. Docker Hub Credentials (Optional)

If you want to push Docker images to Docker Hub:

1. Go to your repository → Settings → Secrets and variables → Actions
2. Click "New repository secret"
3. Add these secrets:

| Secret Name | Description | Example |
|-------------|-------------|---------|
| `DOCKER_USERNAME` | Your Docker Hub username | `your-docker-username` |
| `DOCKER_PASSWORD` | Your Docker Hub password/token | `your-docker-password` |

**Note:** For Docker Hub, it's recommended to use an access token instead of your password:
1. Go to [Docker Hub](https://hub.docker.com) → Account Settings → Security
2. Create a new access token
3. Use the token as `DOCKER_PASSWORD`

#### 2. Security Scanning (Optional)

For security vulnerability scanning:

| Secret Name | Description | Example |
|-------------|-------------|---------|
| `SNYK_TOKEN` | Snyk API token for security scanning | `your-snyk-token` |

To get a Snyk token:
1. Sign up at [Snyk](https://snyk.io)
2. Go to Account Settings → API tokens
3. Create a new token

#### 3. Code Coverage (Optional)

For code coverage reporting:

| Secret Name | Description | Example |
|-------------|-------------|---------|
| `CODECOV_TOKEN` | Codecov token for coverage reporting | `your-codecov-token` |

To get a Codecov token:
1. Sign up at [Codecov](https://codecov.io)
2. Connect your GitHub repository
3. Get the token from your repository settings

## 🚀 CI/CD Pipeline Features

### What Works Without Secrets

The following features work immediately without any secrets:

- ✅ **Testing**: Runs on Python 3.8, 3.9, 3.10, 3.11
- ✅ **Linting**: Black, flake8, mypy code quality checks
- ✅ **Type Checking**: Static type analysis
- ✅ **Unit Tests**: Automated test execution
- ✅ **Docker Build**: Builds Docker image locally
- ✅ **Security Scan**: Basic security checks (if Snyk token provided)

### What Requires Secrets

- 🔐 **Docker Hub Push**: Requires `DOCKER_USERNAME` and `DOCKER_PASSWORD`
- 🔐 **Security Scanning**: Requires `SNYK_TOKEN`
- 🔐 **Code Coverage**: Requires `CODECOV_TOKEN`
- 🔐 **Staging Deployment**: Requires deployment credentials

## 📋 Workflow Triggers

The CI/CD pipeline runs on:

- **Push to `main`**: Full build, test, and optional Docker push
- **Push to `develop`**: Full build, test, and staging deployment
- **Pull Request to `main`**: Test and lint only

## 🔍 Monitoring the Pipeline

### View Workflow Runs

1. Go to your repository → Actions tab
2. Click on "CI/CD Pipeline" workflow
3. View individual job results

### Common Issues and Solutions

#### 1. "Context access might be invalid" Warning

This is normal for GitHub Actions YAML files. The linter doesn't recognize GitHub-specific contexts like `secrets`, `github`, etc. This warning can be ignored.

#### 2. Docker Build Fails

If Docker build fails:
- Check that your `Dockerfile` is in the root directory
- Ensure all required files are present
- Check the build logs for specific error messages

#### 3. Tests Fail

If tests fail:
- Check that all dependencies are in `requirements.txt`
- Ensure test files are in the `tests/` directory
- Look at the test output for specific failures

#### 4. Linting Fails

If linting fails:
- Run `black .` locally to format code
- Fix any flake8 violations
- Ensure type hints are correct for mypy

## 🛠️ Local Development Setup

### Pre-commit Hooks

Install pre-commit hooks to catch issues before pushing:

```bash
# Install pre-commit
pip install pre-commit

# Install the git hook scripts
pre-commit install

# Run against all files
pre-commit run --all-files
```

### Running Tests Locally

```bash
# Install test dependencies
pip install pytest pytest-cov black flake8 mypy

# Run tests
pytest tests/

# Run linting
black --check .
flake8 .
mypy api/ database.py streamlit_app*.py
```

## 📊 GitHub Insights

Once your repository is set up, you can view:

- **Code**: Repository contents and history
- **Issues**: Bug reports and feature requests
- **Pull Requests**: Code reviews and contributions
- **Actions**: CI/CD pipeline runs
- **Security**: Vulnerability alerts
- **Insights**: Repository analytics

## 🔄 Continuous Deployment

### Automatic Deployment Setup

To enable automatic deployment:

1. **For AWS**: Configure AWS credentials in GitHub secrets
2. **For Docker**: Set up Docker Hub credentials
3. **For Kubernetes**: Configure kubectl access

### Manual Deployment

For manual deployment:

```bash
# Deploy locally
python scripts/deploy.py --environment development --target local

# Deploy with Docker
python scripts/deploy.py --environment staging --target docker

# Deploy to AWS
python scripts/deploy.py --environment production --target aws
```

## 🆘 Getting Help

### GitHub Issues

- Create an issue for bugs or feature requests
- Use the issue templates if available
- Provide detailed information about the problem

### GitHub Discussions

- Use Discussions for questions and general help
- Share your setup and configuration
- Ask for best practices and recommendations

### Documentation

- Check the main `README.md` for project overview
- Review `DEPLOYMENT_GUIDE.md` for deployment instructions
- Look at inline code documentation

## 🎉 Success Checklist

- [ ] Repository created and code pushed
- [ ] GitHub Actions workflow runs successfully
- [ ] Tests pass on all Python versions
- [ ] Linting passes without errors
- [ ] Docker image builds successfully
- [ ] Optional secrets configured (if needed)
- [ ] Pre-commit hooks installed locally
- [ ] Documentation updated

Your Movie Recommender System is now ready for collaborative development! 🚀 