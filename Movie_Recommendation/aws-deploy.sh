#!/bin/bash

# AWS Free Tier Deployment Script for Movie Recommender
# This script deploys the application to AWS using free tier services

echo "🚀 Deploying Movie Recommender to AWS Free Tier..."

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo "❌ AWS CLI is not installed. Please install it first."
    exit 1
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install it first."
    exit 1
fi

# Set variables
PROJECT_NAME="movie-recommender"
AWS_REGION="us-east-1"
ECR_REPOSITORY_NAME="$PROJECT_NAME"
EC2_INSTANCE_TYPE="t2.micro"  # Free tier eligible

echo "📦 Building Docker image..."
docker build -t $PROJECT_NAME .

echo "🏷️ Tagging Docker image..."
docker tag $PROJECT_NAME:latest $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPOSITORY_NAME:latest

echo "📤 Pushing to ECR..."
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPOSITORY_NAME:latest

echo "☁️ Creating ECS cluster..."
aws ecs create-cluster --cluster-name $PROJECT_NAME-cluster --region $AWS_REGION

echo "📋 Creating ECS task definition..."
cat > task-definition.json << EOF
{
    "family": "$PROJECT_NAME",
    "networkMode": "awsvpc",
    "requiresCompatibilities": ["FARGATE"],
    "cpu": "256",
    "memory": "512",
    "executionRoleArn": "arn:aws:iam::$AWS_ACCOUNT_ID:role/ecsTaskExecutionRole",
    "containerDefinitions": [
        {
            "name": "$PROJECT_NAME",
            "image": "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPOSITORY_NAME:latest",
            "portMappings": [
                {
                    "containerPort": 8501,
                    "protocol": "tcp"
                }
            ],
            "essential": true,
            "logConfiguration": {
                "logDriver": "awslogs",
                "options": {
                    "awslogs-group": "/ecs/$PROJECT_NAME",
                    "awslogs-region": "$AWS_REGION",
                    "awslogs-stream-prefix": "ecs"
                }
            }
        }
    ]
}
EOF

aws ecs register-task-definition --cli-input-json file://task-definition.json --region $AWS_REGION

echo "🌐 Creating Application Load Balancer..."
aws elbv2 create-load-balancer \
    --name $PROJECT_NAME-alb \
    --subnets subnet-12345678 subnet-87654321 \
    --security-groups sg-12345678 \
    --region $AWS_REGION

echo "🎯 Creating target group..."
aws elbv2 create-target-group \
    --name $PROJECT_NAME-tg \
    --protocol HTTP \
    --port 8501 \
    --vpc-id vpc-12345678 \
    --target-type ip \
    --region $AWS_REGION

echo "✅ Deployment completed!"
echo "🌍 Your application will be available at: http://your-load-balancer-url"
echo "💰 This deployment uses AWS Free Tier services"
echo "📊 Monitor usage in AWS Billing Dashboard" 