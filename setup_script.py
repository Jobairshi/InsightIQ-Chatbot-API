#!/usr/bin/env python3
"""
Setup and Deployment Script for Advanced RAG-Powered FAQ Chatbot
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
import json
from typing import Dict, Any
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChatbotSetup:
    """Comprehensive setup and deployment manager"""
    
    def __init__(self):
        self.base_dir = Path.cwd()
        self.requirements_installed = False
        self.env_configured = False
        self.data_validated = False
        self.vector_db_ready = False
    
    def check_python_version(self) -> bool:
        """Check if Python version is compatible"""
        version = sys.version_info
        if version.major == 3 and version.minor >= 8:
            logger.info(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
            return True
        else:
            logger.error(f"❌ Python {version.major}.{version.minor}.{version.micro} is not compatible. Requires Python 3.8+")
            return False
    
    def install_requirements(self) -> bool:
        """Install required packages"""
        try:
            logger.info("📦 Installing required packages...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
            self.requirements_installed = True
            logger.info("✅ All requirements installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Error installing requirements: {e}")
            return False
    
    def create_env_file(self) -> bool:
        """Create .env file if it doesn't exist"""
        env_file = self.base_dir / ".env"
        env_example = self.base_dir / ".env.example"
        
        if env_file.exists():
            logger.info("✅ .env file already exists")
            return True
        
        if env_example.exists():
            # Copy example to .env
            with open(env_example, 'r') as src, open(env_file, 'w') as dst:
                dst.write(src.read())
            logger.info("✅ Created .env file from .env.example")
            logger.warning("⚠️  Please update .env file with your actual API keys and endpoints")
            return True
        else:
            logger.error("❌ .env.example file not found. Please create environment configuration manually.")
            return False
    
    def validate_data_files(self) -> bool:
        """Validate required CSV data files"""
        required_files = ["new_features.csv", "new_feature_categories.csv"]
        missing_files = []
        
        for file_name in required_files:
            file_path = self.base_dir / file_name
            if not file_path.exists():
                missing_files.append(file_name)
        
        if missing_files:
            logger.error(f"❌ Missing required data files: {missing_files}")
            logger.info("Please ensure the following CSV files are present:")
            for file_name in required_files:
                logger.info(f"   - {file_name}")
            return False
        
        # Validate CSV structure
        try:
            faq_df = pd.read_csv("new_features.csv")
            cat_df = pd.read_csv("new_feature_categories.csv")
            
            # Check required columns
            required_faq_cols = ["id", "title", "short_desc", "desc", "cat_id"]
            required_cat_cols = ["id", "category_name"]
            
            missing_faq_cols = [col for col in required_faq_cols if col not in faq_df.columns]
            missing_cat_cols = [col for col in required_cat_cols if col not in cat_df.columns]
            
            if missing_faq_cols:
                logger.error(f"❌ Missing columns in new_features.csv: {missing_faq_cols}")
                return False
            
            if missing_cat_cols:
                logger.error(f"❌ Missing columns in new_feature_categories.csv: {missing_cat_cols}")
                return False
            
            logger.info(f"✅ Data validation passed - {len(faq_df)} FAQ entries, {len(cat_df)} categories")
            self.data_validated = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Error validating data files: {e}")
            return False
    
    def check_environment_variables(self) -> bool:
        """Check if all required environment variables are set"""
        from dotenv import load_dotenv
        load_dotenv()
        
        required_vars = [
            "AZURE_OPENAI_ENDPOINT",
            "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_API_VERSION",
            "AZURE_DEPLOYMENT_NAME"
        ]
        
        missing_vars = []
        placeholder_vars = []
        
        for var in required_vars:
            value = os.getenv(var)
            if not value:
                missing_vars.append(var)
            elif value.startswith("your-") or value == "your-azure-openai-api-key":
                placeholder_vars.append(var)
        
        if missing_vars:
            logger.error(f"❌ Missing environment variables: {missing_vars}")
            return False
        
        if placeholder_vars:
            logger.error(f"❌ Environment variables contain placeholder values: {placeholder_vars}")
            logger.info("Please update your .env file with actual values")
            return False
        
        logger.info("✅ All required environment variables are configured")
        self.env_configured = True
        return True
    
    def test_connections(self) -> Dict[str, bool]:
        """Test connections to external services"""
        results = {}
        
        # Test Azure OpenAI
        try:
            from langchain_openai import AzureChatOpenAI
            from dotenv import load_dotenv
            load_dotenv()
            
            llm = AzureChatOpenAI(
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                deployment_name=os.getenv("AZURE_DEPLOYMENT_NAME"),
                temperature=0.1,
                max_tokens=10
            )
            
            # Test with simple query
            response = llm.invoke([{"role": "user", "content": "Hello"}])
            results["azure_openai"] = True
            logger.info("✅ Azure OpenAI connection successful")
            
        except Exception as e:
            results["azure_openai"] = False
            logger.error(f"❌ Azure OpenAI connection failed: {e}")
        
        # Test Pinecone
        try:
            from pinecone import Pinecone
            
            pc = Pinecone(api_key="pcsk_4BnTBd_MWr6WTcLR1FH7MCATTLbKWMBah8becVf6KUecVUmzc5usoNsTjY6gQd2EGqNvVC")
            indexes = pc.list_indexes()
            results["pinecone"] = True
            logger.info("✅ Pinecone connection successful")
            
        except Exception as e:
            results["pinecone"] = False
            logger.error(f"❌ Pinecone connection failed: {e}")
        
        return results
    
    def initialize_vector_database(self) -> bool:
        """Initialize and populate vector database"""
        try:
            logger.info("🚀 Initializing vector database...")
            
            # Import the main application to trigger initialization
            from main import startup
            
            if startup():
                self.vector_db_ready = True
                logger.info("✅ Vector database initialized successfully")
                return True
            else:
                logger.error("❌ Failed to initialize vector database")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error initializing vector database: {e}")
            return False
    
    def create_startup_script(self) -> bool:
        """Create startup script for easy deployment"""
        startup_script = """#!/bin/bash
# Startup script for RAG-Powered FAQ Chatbot

echo "🚀 Starting RAG-Powered FAQ Chatbot..."

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "📦 Activating virtual environment..."
    source venv/bin/activate
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "❌ .env file not found. Please create it from .env.example"
    exit 1
fi

# Install/update dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Start the application
echo "🌐 Starting FastAPI server..."
python main.py

echo "✅ Application started successfully!"
echo "🌐 API available at: http://localhost:8001"
echo "📚 API documentation: http://localhost:8001/docs"
"""
        
        try:
            with open("start.sh", "w") as f:
                f.write(startup_script)
            
            # Make executable
            os.chmod("start.sh", 0o755)
            
            logger.info("✅ Created startup script: start.sh")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error creating startup script: {e}")
            return False
    
    def generate_setup_report(self) -> Dict[str, Any]:
        """Generate comprehensive setup report"""
        return {
            "setup_status": {
                "python_compatible": self.check_python_version(),
                "requirements_installed": self.requirements_installed,
                "env_configured": self.env_configured,
                "data_validated": self.data_validated,
                "vector_db_ready": self.vector_db_ready
            },
            "next_steps": self._get_next_steps(),
            "useful_commands": {
                "start_server": "python main.py",
                "run_setup": "python setup.py",
                "test_api": "curl http://localhost:8001/health",
                "view_docs": "http://localhost:8001/docs"
            }
        }
    
    def _get_next_steps(self) -> List[str]:
        """Get next steps based on current setup status"""
        steps = []
        
        if not self.requirements_installed:
            steps.append("Install requirements: pip install -r requirements.txt")
        
        if not self.env_configured:
            steps.append("Configure environment variables in .env file")
        
        if not self.data_validated:
            steps.append("Ensure CSV data files are present and properly formatted")
        
        if not self.vector_db_ready:
            steps.append("Initialize vector database by running the application")
        
        if all([self.requirements_installed, self.env_configured, self.data_validated]):
            steps.append("Start the application: python main.py")
            steps.append("Access API documentation: http://localhost:8001/docs")
        
        return steps
    
    def run_complete_setup(self):
        """Run complete setup process"""
        logger.info("🚀 Starting comprehensive setup process...")
        
        # Step 1: Check Python version
        if not self.check_python_version():
            return False
        
        # Step 2: Install requirements
        if not self.install_requirements():
            return False
        
        # Step 3: Create environment file
        if not self.create_env_file():
            return False
        
        # Step 4: Validate data files
        if not self.validate_data_files():
            return False
        
        # Step 5: Check environment variables
        if not self.check_environment_variables():
            logger.warning("⚠️  Environment variables need to be configured manually")
            logger.info("Please update your .env file and run setup again")
            return False
        
        # Step 6: Test connections
        logger.info("🔌 Testing external service connections...")
        connection_results = self.test_connections()
        
        if not all(connection_results.values()):
            logger.warning("⚠️  Some connections failed. Please check your API keys and endpoints")
            failed_services = [service for service, status in connection_results.items() if not status]
            logger.warning(f"Failed services: {failed_services}")
            return False
        
        # Step 7: Initialize vector database
        if not self.initialize_vector_database():
            logger.error("❌ Failed to initialize vector database")
            return False
        
        # Step 8: Create startup script
        self.create_startup_script()
        
        # Step 9: Generate final report
        report = self.generate_setup_report()
        
        logger.info("=" * 60)
        logger.info("🎉 SETUP COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info("📊 Setup Report:")
        for key, value in report["setup_status"].items():
            status = "✅" if value else "❌"
            logger.info(f"   {status} {key.replace('_', ' ').title()}")
        
        logger.info("\n🚀 You can now start the application with:")
        logger.info("   python main.py")
        logger.info("   or")
        logger.info("   ./start.sh")
        
        logger.info("\n📚 API Documentation will be available at:")
        logger.info("   http://localhost:8001/docs")
        
        return True

def main():
    """Main setup function"""
    print("🚀 Advanced RAG-Powered FAQ Chatbot Setup")
    print("=" * 50)
    
    setup = ChatbotSetup()
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "check":
            # Quick check mode
            logger.info("🔍 Running quick system check...")
            setup.check_python_version()
            setup.validate_data_files()
            setup.check_environment_variables()
            connections = setup.test_connections() 
            
            print("\n📊 System Status:")
            print(f"Python Compatible: {'✅' if setup.check_python_version() else '❌'}")
            print(f"Data Files: {'✅' if setup.data_validated else '❌'}")
            print(f"Environment: {'✅' if setup.env_configured else '❌'}")
            print(f"Azure OpenAI: {'✅' if connections.get('azure_openai', False) else '❌'}")
            print(f"Pinecone: {'✅' if connections.get('pinecone', False) else '❌'}")
            
        elif command == "install":
            # Install requirements only
            setup.install_requirements()
            
        elif command == "init":
            # Initialize vector database only
            setup.initialize_vector_database()
            
        elif command == "test":
            # Test connections only
            connections = setup.test_connections()
            print("\n🔌 Connection Test Results:")
            for service, status in connections.items():
                print(f"{service}: {'✅ Connected' if status else '❌ Failed'}")
                
        else:
            print(f"Unknown command: {command}")
            print("Available commands: check, install, init, test")
    else:
        # Full setup
        if setup.run_complete_setup():
            print("\n🎉 Setup completed successfully!")
            print("You can now start your RAG-powered chatbot!")
        else:
            print("\n❌ Setup failed. Please check the logs and try again.")
            sys.exit(1)

if __name__ == "__main__":
    main()