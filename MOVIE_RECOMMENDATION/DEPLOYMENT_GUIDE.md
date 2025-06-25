# 🚀 Movie Recommender Deployment Guide

## 📋 Overview
This guide covers deploying your Movie Recommender System through three phases:
1. **Local Enhanced Version** (Week 1)
2. **GitHub + Streamlit Cloud** (Week 2)
3. **AWS Free Tier** (Week 3)

---

## 🏠 Phase 1: Enhanced Local Version

### ✅ What's New:
- **User Authentication** (SQLite database)
- **Personal Ratings System**
- **Watchlist Management**
- **User Preferences**
- **Enhanced UI with Dashboard**

### 🚀 Quick Start:
```bash
# Install new dependencies
pip install -r requirements.txt

# Run the enhanced version
streamlit run streamlit_app_enhanced.py
```

### 🎯 Features Available:
- ✅ User registration/login
- ✅ Rate movies (1-5 stars)
- ✅ Add movies to watchlist
- ✅ Set genre preferences
- ✅ Personalized recommendations
- ✅ AI Assistant (DeepSeek-Coder)
- ✅ User dashboard with stats

---

## ☁️ Phase 2: GitHub + Streamlit Cloud

### 📦 Preparation:
1. **Create GitHub Repository**
```bash
git init
git add .
git commit -m "Initial commit: Movie Recommender Pro"
git branch -M main
git remote add origin https://github.com/yourusername/movie-recommender.git
git push -u origin main
```

2. **Deploy to Streamlit Cloud**
- Go to [share.streamlit.io](https://share.streamlit.io)
- Connect your GitHub account
- Select your repository
- Deploy automatically

### 🎯 Benefits:
- ✅ **Free hosting** (unlimited)
- ✅ **Automatic deployments** on git push
- ✅ **Public URL** for sharing
- ✅ **No server management**

### 📝 Files Added:
- `.github/workflows/deploy.yml` - CI/CD pipeline
- `Dockerfile` - Container configuration
- `docker-compose.yml` - Local container setup

---

## ☁️ Phase 3: AWS Free Tier

### 🆓 AWS Free Tier Services Used:
- **EC2** (t2.micro) - 750 hours/month
- **ECR** - Container registry
- **ECS Fargate** - Container orchestration
- **Application Load Balancer** - Traffic distribution
- **CloudWatch** - Monitoring

### 🚀 Deployment Steps:

#### 1. **AWS Setup**
```bash
# Install AWS CLI
aws configure  # Set your credentials

# Set environment variables
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export AWS_REGION="us-east-1"
```

#### 2. **Create ECR Repository**
```bash
aws ecr create-repository --repository-name movie-recommender --region $AWS_REGION
```

#### 3. **Deploy with Script**
```bash
# Make script executable
chmod +x aws-deploy.sh

# Run deployment
./aws-deploy.sh
```

### 💰 Cost Management:
- **Free Tier Limits:**
  - EC2: 750 hours/month
  - ECR: 500MB storage
  - ALB: 750 hours/month
  - Data transfer: 15GB/month

- **Monitoring:**
  - Set up billing alerts
  - Monitor usage in AWS Console
  - Use AWS Cost Explorer

---

## 🔧 Configuration Files

### `requirements.txt`
```txt
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.1.0
requests>=2.28.0
streamlit>=1.25.0
ollama>=0.5.0
sqlalchemy>=1.4.0
fastapi>=0.68.0
uvicorn>=0.15.0
python-multipart>=0.0.5
python-jose[cryptography]>=3.3.0
passlib[bcrypt]>=1.7.4
```

### `Dockerfile`
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

---

## 🎯 Feature Comparison

| Feature | Local | Streamlit Cloud | AWS |
|---------|-------|-----------------|-----|
| User Auth | ✅ SQLite | ✅ SQLite | ✅ RDS |
| AI Assistant | ✅ Ollama | ❌ No Ollama | ✅ ECS |
| Database | ✅ SQLite | ✅ SQLite | ✅ RDS |
| Cost | Free | Free | Free Tier |
| Scalability | ❌ | ⚠️ Limited | ✅ High |
| Custom Domain | ❌ | ❌ | ✅ |

---

## 🚨 Important Notes

### For Streamlit Cloud:
- **No Ollama support** - AI features will use fallback
- **SQLite database** - Data resets on redeploy
- **Public access** - Anyone can use your app

### For AWS:
- **Monitor costs** - Stay within free tier
- **Backup data** - Use RDS for persistence
- **Security groups** - Configure properly

---

## 🎉 Success Metrics

### Phase 1 ✅
- [ ] Enhanced app running locally
- [ ] User registration working
- [ ] Ratings system functional
- [ ] AI Assistant responding

### Phase 2 ✅
- [ ] GitHub repository created
- [ ] Streamlit Cloud deployment
- [ ] Public URL accessible
- [ ] CI/CD pipeline working

### Phase 3 ✅
- [ ] AWS account configured
- [ ] ECR repository created
- [ ] ECS service running
- [ ] Load balancer accessible
- [ ] Monitoring set up

---

## 🆘 Troubleshooting

### Common Issues:

1. **Import Errors**
   ```bash
   pip install -r requirements.txt
   ```

2. **Database Issues**
   ```bash
   # Reset database
   rm movie_recommender.db
   ```

3. **Ollama Not Working**
   ```bash
   # Check Ollama status
   ollama list
   ```

4. **AWS Deployment Fails**
   ```bash
   # Check AWS credentials
   aws sts get-caller-identity
   ```

---

## 📞 Support

- **Local Issues**: Check logs in terminal
- **Streamlit Cloud**: Check deployment logs
- **AWS Issues**: Check CloudWatch logs

**Happy Deploying! 🚀** 