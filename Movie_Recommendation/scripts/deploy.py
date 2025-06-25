#!/usr/bin/env python3
"""
Deployment script for Movie Recommender System
"""
import argparse
import os
import sys
import subprocess
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional

# Add the parent directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from config import get_config, reload_config
from utils.logger import get_logger


class Deployer:
    """Main deployment class"""
    
    def __init__(self, environment: str, target: str = "local"):
        self.environment = environment
        self.target = target
        self.config = reload_config(environment)
        self.logger = get_logger("deployer")
        self.project_root = Path(__file__).parent.parent
        
    def deploy(self):
        """Main deployment method"""
        self.logger.info(f"Starting deployment to {self.target} ({self.environment})")
        
        try:
            # Validate environment
            self._validate_environment()
            
            # Run pre-deployment checks
            self._pre_deployment_checks()
            
            # Deploy based on target
            if self.target == "local":
                self._deploy_local()
            elif self.target == "docker":
                self._deploy_docker()
            elif self.target == "aws":
                self._deploy_aws()
            else:
                raise ValueError(f"Unknown deployment target: {self.target}")
            
            # Run post-deployment checks
            self._post_deployment_checks()
            
            self.logger.info("Deployment completed successfully")
            
        except Exception as e:
            self.logger.error(f"Deployment failed: {e}")
            raise
    
    def _validate_environment(self):
        """Validate deployment environment"""
        self.logger.info("Validating environment configuration")
        
        # Check required environment variables
        if self.environment == "production":
            required_vars = ["SECRET_KEY", "DATABASE_URL"]
            missing_vars = [var for var in required_vars if not os.getenv(var)]
            
            if missing_vars:
                raise ValueError(f"Missing required environment variables: {missing_vars}")
        
        # Validate configuration
        if not self.config.validate():
            raise ValueError("Configuration validation failed")
    
    def _pre_deployment_checks(self):
        """Run pre-deployment checks"""
        self.logger.info("Running pre-deployment checks")
        
        # Check if model exists
        model_path = Path(self.config.model.model_path)
        if not model_path.exists():
            self.logger.warning("Model not found, training new model...")
            self._train_model()
        
        # Check database
        self._check_database()
        
        # Check dependencies
        self._check_dependencies()
    
    def _train_model(self):
        """Train the recommendation model"""
        self.logger.info("Training recommendation model")
        
        try:
            result = subprocess.run(
                [sys.executable, "api/main.py"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True
            )
            self.logger.info("Model training completed")
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Model training failed: {e.stderr}")
            raise
    
    def _check_database(self):
        """Check database connectivity and setup"""
        self.logger.info("Checking database")
        
        try:
            from database import MovieDatabase
            db = MovieDatabase()
            
            # Test basic operations
            db.create_user("test_user", "test@example.com", "test_hash")
            user = db.get_user_by_username("test_user")
            
            if user:
                self.logger.info("Database check passed")
            else:
                raise Exception("Database user creation failed")
                
        except Exception as e:
            self.logger.error(f"Database check failed: {e}")
            raise
    
    def _check_dependencies(self):
        """Check if all dependencies are installed"""
        self.logger.info("Checking dependencies")
        
        required_packages = [
            "streamlit", "pandas", "numpy", "scikit-learn", 
            "requests", "ollama"
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            raise ValueError(f"Missing required packages: {missing_packages}")
        
        self.logger.info("Dependencies check passed")
    
    def _deploy_local(self):
        """Deploy locally"""
        self.logger.info("Deploying locally")
        
        # Start Ollama if not running
        self._ensure_ollama_running()
        
        # Start Streamlit
        self._start_streamlit()
    
    def _deploy_docker(self):
        """Deploy using Docker"""
        self.logger.info("Deploying with Docker")
        
        # Build Docker image
        self._build_docker_image()
        
        # Run Docker container
        self._run_docker_container()
    
    def _deploy_aws(self):
        """Deploy to AWS"""
        self.logger.info("Deploying to AWS")
        
        # Check AWS credentials
        self._check_aws_credentials()
        
        # Run AWS deployment script
        self._run_aws_deployment()
    
    def _ensure_ollama_running(self):
        """Ensure Ollama is running"""
        self.logger.info("Checking Ollama status")
        
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                self.logger.info("Ollama is running")
                return
        except:
            pass
        
        self.logger.info("Starting Ollama...")
        try:
            subprocess.Popen(["ollama", "serve"], start_new_session=True)
            time.sleep(5)  # Wait for Ollama to start
            self.logger.info("Ollama started")
        except Exception as e:
            self.logger.error(f"Failed to start Ollama: {e}")
            raise
    
    def _start_streamlit(self):
        """Start Streamlit application"""
        self.logger.info("Starting Streamlit application")
        
        cmd = [
            sys.executable, "-m", "streamlit", "run", 
            "streamlit_app_enhanced.py",
            "--server.port", str(self.config.app.port),
            "--server.address", self.config.app.host
        ]
        
        if self.config.app.debug:
            cmd.extend(["--logger.level", "debug"])
        
        try:
            subprocess.run(cmd, cwd=self.project_root, check=True)
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to start Streamlit: {e}")
            raise
    
    def _build_docker_image(self):
        """Build Docker image"""
        self.logger.info("Building Docker image")
        
        try:
            subprocess.run(
                ["docker", "build", "-t", "movie-recommender", "."],
                cwd=self.project_root,
                check=True
            )
            self.logger.info("Docker image built successfully")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Docker build failed: {e}")
            raise
    
    def _run_docker_container(self):
        """Run Docker container"""
        self.logger.info("Running Docker container")
        
        try:
            subprocess.run([
                "docker", "run", "-d",
                "-p", f"{self.config.app.port}:8501",
                "--name", "movie-recommender",
                "movie-recommender"
            ], cwd=self.project_root, check=True)
            
            self.logger.info("Docker container started successfully")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to run Docker container: {e}")
            raise
    
    def _check_aws_credentials(self):
        """Check AWS credentials"""
        self.logger.info("Checking AWS credentials")
        
        try:
            result = subprocess.run(
                ["aws", "sts", "get-caller-identity"],
                capture_output=True,
                text=True,
                check=True
            )
            
            identity = json.loads(result.stdout)
            self.logger.info(f"AWS credentials valid for account: {identity['Account']}")
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"AWS credentials check failed: {e}")
            raise
    
    def _run_aws_deployment(self):
        """Run AWS deployment script"""
        self.logger.info("Running AWS deployment")
        
        deploy_script = self.project_root / "aws-deploy.sh"
        
        if not deploy_script.exists():
            raise FileNotFoundError("AWS deployment script not found")
        
        try:
            subprocess.run(
                ["bash", str(deploy_script)],
                cwd=self.project_root,
                check=True
            )
            self.logger.info("AWS deployment completed")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"AWS deployment failed: {e}")
            raise
    
    def _post_deployment_checks(self):
        """Run post-deployment checks"""
        self.logger.info("Running post-deployment checks")
        
        # Check if application is responding
        self._check_application_health()
        
        # Check system resources
        self._check_system_resources()
        
        self.logger.info("Post-deployment checks completed")
    
    def _check_application_health(self):
        """Check if application is healthy"""
        self.logger.info("Checking application health")
        
        try:
            import requests
            response = requests.get(
                f"http://{self.config.app.host}:{self.config.app.port}",
                timeout=10
            )
            
            if response.status_code == 200:
                self.logger.info("Application health check passed")
            else:
                raise Exception(f"Application returned status {response.status_code}")
                
        except Exception as e:
            self.logger.error(f"Application health check failed: {e}")
            raise
    
    def _check_system_resources(self):
        """Check system resource usage"""
        self.logger.info("Checking system resources")
        
        try:
            import psutil
            
            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > 90:
                self.logger.warning(f"High CPU usage: {cpu_percent}%")
            
            # Check memory usage
            memory = psutil.virtual_memory()
            if memory.percent > 90:
                self.logger.warning(f"High memory usage: {memory.percent}%")
            
            # Check disk usage
            disk = psutil.disk_usage('/')
            if disk.percent > 90:
                self.logger.warning(f"High disk usage: {disk.percent}%")
            
            self.logger.info("System resource check completed")
            
        except Exception as e:
            self.logger.error(f"System resource check failed: {e}")
            raise


def main():
    """Main deployment function"""
    parser = argparse.ArgumentParser(description="Deploy Movie Recommender System")
    parser.add_argument(
        "--environment", "-e",
        choices=["development", "staging", "production"],
        default="development",
        help="Deployment environment"
    )
    parser.add_argument(
        "--target", "-t",
        choices=["local", "docker", "aws"],
        default="local",
        help="Deployment target"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = get_logger("deployer")
    if args.verbose:
        logger.logger.setLevel(logging.DEBUG)
    
    # Run deployment
    deployer = Deployer(args.environment, args.target)
    deployer.deploy()


if __name__ == "__main__":
    main() 