from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, model_validator
from typing import Dict, Optional, List
import httpx
import asyncio
import docker
import os
import json
from datetime import datetime
from enum import Enum
from pathlib import Path
import logging
import traceback
import uuid

app = FastAPI(title="Pulsr Core Service", version="2.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Docker client
docker_client = docker.from_env()

# Service configuration
SERVICES = {
    "atlas": {"port": 8001, "host": "localhost", "internal_host": "atlas"},
    "census": {"port": 8002, "host": "localhost", "internal_host": "census"},
    "hermes": {"port": 8003, "host": "localhost", "internal_host": "hermes"},
    #"lambda": {"port": 8004, "host": "localhost", "internal_host": "lambda"},
    "lingua": {"port": 8005, "host": "localhost", "internal_host": "lingua"},
    "nexus": {"port": 8006, "host": "localhost", "internal_host": "nexus"},
}

# Network name for all services
NETWORK_NAME = "pulsr-network"

# Configuration storage path (will be volume mounted)
CONFIG_STORAGE_PATH = "/app/data/deployment_config.json"

class DatabaseType(str, Enum):
    MANAGED = "managed"  # We create and manage the PostgreSQL container
    EXTERNAL = "external"  # User provides DATABASE_URL

class MongoType(str, Enum):
    MANAGED = "managed"    # We create and manage the MongoDB container
    EXTERNAL = "external"  # User provides MONGODB_URL

class DeploymentConfig(BaseModel):
    tag: str = "latest"
    database_type: DatabaseType = DatabaseType.MANAGED
    database_url: Optional[str] = None  # Required if database_type is EXTERNAL
    postgres_password: Optional[str] = None  # Required if database_type is MANAGED
    scaleway_secret: Optional[str] = None  # For Scaleway registry auth
    
    # Shared environment variables
    secret_key: str
    
    # Atlas-specific
    mongo_type: MongoType = MongoType.MANAGED
    mongodb_url: Optional[str] = None  # Required if mongo_type is EXTERNAL
    mongodb_db: str = "atlas_documents"
    
    # Census-specific
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    otp_expire_minutes: int = 5
    admin_email: str = "admin@example.com"
    hermes_from_email: str = "noreply@pulsr.one"
    
    # Hermes-specific
    smtp_host: str = "0.0.0.0"
    smtp_port: int = 25
    smtp_domain: str = "pulsr.one"
    outbound_smtp_host: Optional[str] = None
    outbound_smtp_port: int = 587
    outbound_smtp_user: Optional[str] = None
    outbound_smtp_password: Optional[str] = None
    outbound_smtp_use_tls: bool = True
    dkim_selector: str = "default"
    hermes_api_key: str  # Merged from api_key - used for both Census->Hermes and Hermes API auth
    
    @model_validator(mode='after')
    def validate_config(self):
        # Validate database configuration
        if self.database_type == DatabaseType.EXTERNAL and not self.database_url:
            raise ValueError("database_url is required when database_type is EXTERNAL")
        if self.database_type == DatabaseType.MANAGED and not self.postgres_password:
            raise ValueError("postgres_password is required when database_type is MANAGED")
            
        # Validate MongoDB configuration
        if self.mongo_type == MongoType.EXTERNAL and not self.mongodb_url:
            raise ValueError("mongodb_url is required when mongo_type is EXTERNAL")
            
        return self
    
    # Lingua-specific
    redis_url: Optional[str] = "redis://pulsr_redis:6379"
    openai_api_key: Optional[str] = None
    openai_api_base_url: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    default_model: Optional[str] = None
    local_llm_endpoint: Optional[str] = None
    private_cloud_endpoint: Optional[str] = None
    private_cloud_api_key: Optional[str] = None
    default_llm_provider: str = "openai"
    otel_exporter_otlp_endpoint: Optional[str] = None
    otel_service_name: str = "llm-wrapper-service"

class DeploymentStatus(BaseModel):
    deployed: bool
    deployment_time: Optional[datetime] = None
    configuration: Optional[Dict] = None
    database_type: Optional[DatabaseType] = None
    database_status: Optional[str] = None
    services_deployed: List[str] = []
    error: Optional[str] = None

class ServiceStatus(BaseModel):
    name: str
    port: int
    host: str
    status: str
    container_id: Optional[str] = None
    health: Optional[Dict] = None
    last_check: Optional[datetime] = None
    error: Optional[str] = None

class DeploymentStep(BaseModel):
    name: str
    status: str  # "pending", "in_progress", "completed", "failed"
    message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None

class DeploymentProgress(BaseModel):
    deployment_id: str
    status: str  # "starting", "in_progress", "completed", "failed"
    current_step: Optional[str] = None
    steps: List[DeploymentStep]
    started_at: datetime
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    config: Optional[Dict] = None

class SystemStatus(BaseModel):
    deployed: bool
    deployment_config: Optional[Dict] = None
    database_type: Optional[DatabaseType] = None
    database_status: Optional[str] = None
    mongo_type: Optional[MongoType] = None
    mongo_status: Optional[str] = None
    services: Dict[str, ServiceStatus]
    healthy_count: int
    unhealthy_count: int
    total_services: int

# Store deployment state
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

deployment_state = {
    "deployed": False,
    "deployment_time": None,
    "configuration": None,
    "database_type": None,
    "mongo_type": None
}

# Deployment progress tracking
deployment_progress = {}

def save_deployment_config(config: DeploymentConfig):
    """Save deployment configuration to persistent storage."""
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(CONFIG_STORAGE_PATH), exist_ok=True)
        
        # Save config with metadata
        config_data = {
            "deployment_time": datetime.utcnow().isoformat(),
            "config": config.dict(exclude={"postgres_password", "scaleway_secret"}),
            "database_type": config.database_type,
            "mongo_type": config.mongo_type
        }
        
        with open(CONFIG_STORAGE_PATH, 'w') as f:
            json.dump(config_data, f, indent=2)
            
        logger.info(f"Deployment config saved to {CONFIG_STORAGE_PATH}")
        
    except Exception as e:
        logger.error(f"Failed to save deployment config: {e}")

def load_deployment_config():
    """Load deployment configuration from persistent storage."""
    try:
        if not os.path.exists(CONFIG_STORAGE_PATH):
            return None
            
        with open(CONFIG_STORAGE_PATH, 'r') as f:
            config_data = json.load(f)
            
        logger.info(f"Deployment config loaded from {CONFIG_STORAGE_PATH}")
        return config_data
        
    except Exception as e:
        logger.error(f"Failed to load deployment config: {e}")
        return None

def clear_deployment_config():
    """Clear stored deployment configuration."""
    try:
        if os.path.exists(CONFIG_STORAGE_PATH):
            os.remove(CONFIG_STORAGE_PATH)
            logger.info("Deployment config cleared")
    except Exception as e:
        logger.error(f"Failed to clear deployment config: {e}")

def detect_existing_deployment():
    """Detect if services are already deployed and update state accordingly."""
    try:
        # Check if any pulsr containers are running
        running_services = []
        for service_name in SERVICES.keys():
            container = get_container_by_name(f"pulsr_{service_name}")
            if container and container.status == "running":
                running_services.append(service_name)
        
        # Check infrastructure containers
        postgres_running = get_container_by_name("pulsr_postgres") is not None
        mongo_running = get_container_by_name("pulsr_mongo") is not None
        
        if running_services and postgres_running:
            logger.info(f"Detected existing deployment with services: {running_services}")
            
            # Try to load stored configuration
            stored_config = load_deployment_config()
            
            deployment_state["deployed"] = True
            deployment_state["database_type"] = DatabaseType.MANAGED
            deployment_state["mongo_type"] = MongoType.MANAGED if mongo_running else MongoType.EXTERNAL
            
            if stored_config:
                # Use stored configuration
                deployment_state["deployment_time"] = datetime.fromisoformat(stored_config["deployment_time"])
                deployment_state["configuration"] = stored_config["config"]
                deployment_state["database_type"] = stored_config.get("database_type", DatabaseType.MANAGED)
                deployment_state["mongo_type"] = stored_config.get("mongo_type", MongoType.MANAGED)
                logger.info("Deployment state recovered from stored configuration")
            else:
                # Fallback to basic detection
                deployment_state["deployment_time"] = datetime.utcnow()
                deployment_state["configuration"] = {
                    "detected": True,
                    "services": running_services,
                    "note": "Deployment state recovered from running containers (no stored config)"
                }
                logger.info("Deployment state recovered from containers only (no stored config found)")
        else:
            logger.info("No existing deployment detected")
            
    except Exception as e:
        logger.error(f"Failed to detect existing deployment: {e}")

def get_container_by_name(name: str):
    """Get container by name."""
    try:
        return docker_client.containers.get(name)
    except docker.errors.NotFound:
        pass
    except Exception:
        pass
    return None

async def check_service_health(name: str, config: dict) -> ServiceStatus:
    """Check the health of a single service."""
    service_status = ServiceStatus(
        name=name,
        port=config["port"],
        host="localhost",  # Always use localhost for health checks from core service
        status="unknown",
        last_check=datetime.utcnow()
    )
    
    # Check if container exists
    container = get_container_by_name(f"pulsr_{name}")
    if container:
        service_status.container_id = container.short_id
        if container.status != "running":
            service_status.status = "stopped"
            service_status.error = f"Container status: {container.status}"
            return service_status
    else:
        service_status.status = "not_deployed"
        return service_status
    
    # Check health endpoint
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Try to connect to service via container network, fallback to host network
            health_urls = [
                f"http://pulsr_{name}:8000/health",  # Container network
                f"http://172.17.0.1:{config['port']}/health",  # Docker bridge gateway
                f"http://host.docker.internal:{config['port']}/health",  # Docker Desktop
            ]
            
            response = None
            last_error = None
            
            for health_url in health_urls:
                try:
                    logger.info(f"Trying health check for {name} at {health_url}")
                    response = await client.get(health_url, timeout=5.0)
                    logger.info(f"Successfully connected to {name} at {health_url}")
                    break
                except (httpx.ConnectError, httpx.TimeoutException) as e:
                    last_error = e
                    logger.debug(f"Failed to connect to {name} at {health_url}: {e}")
                    continue
                except Exception as e:
                    last_error = e
                    logger.warning(f"Unexpected error connecting to {name} at {health_url}: {e}")
                    continue
            
            if response is None:
                raise Exception(f"All health check attempts failed. Last error: {last_error}")
            
            if response.status_code == 200:
                service_status.status = "healthy"
                try:
                    service_status.health = response.json()
                except:
                    service_status.health = {"raw": response.text}
                logger.info(f"Service {name} is healthy")
            else:
                service_status.status = "unhealthy"
                service_status.error = f"HTTP {response.status_code}"
                logger.warning(f"Service {name} returned HTTP {response.status_code}")
                
    except httpx.ConnectError as e:
        service_status.status = "unreachable"
        service_status.error = f"Connection failed: {str(e)}"
        logger.error(f"Service {name} connection failed: {str(e)}")
    except httpx.TimeoutException as e:
        service_status.status = "timeout"
        service_status.error = f"Request timeout: {str(e)}"
        logger.error(f"Service {name} timeout: {str(e)}")
    except Exception as e:
        service_status.status = "error"
        service_status.error = f"Health check error: {str(e)}"
        logger.error(f"Service {name} health check error: {str(e)}")
    
    # Add container diagnostic info if container exists but service is unhealthy
    if container and service_status.status in ["unreachable", "timeout", "unhealthy", "error"]:
        try:
            # Get container logs (last 50 lines)
            logs = container.logs(tail=50).decode('utf-8', errors='ignore')
            service_status.health = {
                "container_status": container.status,
                "container_created": container.attrs.get('Created'),
                "container_started": container.attrs.get('State', {}).get('StartedAt'),
                "restart_count": container.attrs.get('RestartCount', 0),
                "exit_code": container.attrs.get('State', {}).get('ExitCode'),
                "last_logs": logs[-1000:] if logs else "No logs available"  # Last 1000 chars
            }
        except Exception as log_error:
            logger.error(f"Failed to get container diagnostics for {name}: {log_error}")
    
    return service_status

def create_network():
    """Create Docker network for services."""
    try:
        return docker_client.networks.get(NETWORK_NAME)
    except docker.errors.NotFound:
        return docker_client.networks.create(NETWORK_NAME, driver="bridge")

async def self_connect_to_network():
    """Connect the core service container to the pulsr network."""
    try:
        # Get our own container
        import socket
        hostname = socket.gethostname()
        core_container = docker_client.containers.get(hostname)
        
        # Get the pulsr network
        network = docker_client.networks.get(NETWORK_NAME)
        
        # Connect to network if not already connected
        if NETWORK_NAME not in [net['Name'] for net in core_container.attrs['NetworkSettings']['Networks']]:
            network.connect(core_container)
            logger.info(f"Connected core container {hostname} to {NETWORK_NAME}")
        else:
            logger.info(f"Core container already connected to {NETWORK_NAME}")
            
    except Exception as e:
        logger.error(f"Failed to connect core to network: {e}")
        raise

def create_postgres_init_script():
    """Create PostgreSQL initialization script."""
    init_sql = """#!/bin/bash
set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    -- Create separate databases for each microservice
    CREATE DATABASE atlas;
    CREATE DATABASE census;
    CREATE DATABASE hermes;
    CREATE DATABASE lingua;
    CREATE DATABASE nexus;

    -- Grant all privileges
    GRANT ALL PRIVILEGES ON DATABASE atlas TO pulsr;
    GRANT ALL PRIVILEGES ON DATABASE census TO pulsr;
    GRANT ALL PRIVILEGES ON DATABASE hermes TO pulsr;
    GRANT ALL PRIVILEGES ON DATABASE lingua TO pulsr;
    GRANT ALL PRIVILEGES ON DATABASE nexus TO pulsr;
EOSQL

echo "Databases created successfully"
"""
    
    # Create temp file for init script
    init_path = "/tmp/init-databases.sh"
    with open(init_path, "w") as f:
        f.write(init_sql)
    
    # Make the script executable
    import os
    os.chmod(init_path, 0o755)
    return init_path

async def verify_databases_created(postgres_password: str):
    """Verify that all required databases exist using docker exec."""
    container = get_container_by_name("pulsr_postgres")
    if not container:
        raise Exception("PostgreSQL container not found")
    
    # Wait a bit more for PostgreSQL to be fully ready
    await asyncio.sleep(5)
    
    # Check if databases exist
    result = container.exec_run([
        "psql", "-U", "pulsr", "-d", "postgres", "-t", "-c",
        "SELECT datname FROM pg_database WHERE datistemplate = false AND datname NOT IN ('postgres');"
    ], environment={"PGPASSWORD": postgres_password})
    
    logger.info(f"Database check result - Exit code: {result.exit_code}, Output: {result.output.decode()}")
    
    if result.exit_code != 0:
        raise Exception(f"Failed to check databases: {result.output.decode()}")
    
    existing_dbs = [db.strip() for db in result.output.decode().strip().split('\n') if db.strip()]
    required_dbs = ["atlas", "census", "hermes", "lingua", "nexus"]  # Removed lambda since it's commented out
    missing_dbs = [db for db in required_dbs if db not in existing_dbs]
    
    logger.info(f"Existing databases: {existing_dbs}")
    logger.info(f"Required databases: {required_dbs}")
    
    if missing_dbs:
        logger.warning(f"Missing databases: {missing_dbs}")
        raise Exception(f"Missing databases: {missing_dbs}")
        
    logger.info(f"All required databases exist: {existing_dbs}")

async def create_databases_manually(postgres_password: str):
    """Create databases manually using docker exec."""
    container = get_container_by_name("pulsr_postgres")
    if not container:
        raise Exception("PostgreSQL container not found")
    
    required_dbs = ["atlas", "census", "hermes", "lingua", "nexus"]  # Removed lambda since it's commented out
    
    for db_name in required_dbs:
        # Create database
        result = container.exec_run([
            "psql", "-U", "pulsr", "-d", "postgres", "-c",
            f"CREATE DATABASE {db_name};"
        ], environment={"PGPASSWORD": postgres_password})
        
        if result.exit_code == 0:
            logger.info(f"Created database: {db_name}")
        elif "already exists" in result.output.decode():
            logger.info(f"Database already exists: {db_name}")
        else:
            logger.error(f"Failed to create database {db_name}: {result.output.decode()}")
            
        # Grant privileges
        result = container.exec_run([
            "psql", "-U", "pulsr", "-d", "postgres", "-c",
            f"GRANT ALL PRIVILEGES ON DATABASE {db_name} TO pulsr;"
        ], environment={"PGPASSWORD": postgres_password})
        
        if result.exit_code == 0:
            logger.info(f"Granted privileges on {db_name}")
        else:
            logger.warning(f"Failed to grant privileges on {db_name}: {result.output.decode()}")

def get_service_environment(service_name: str, config: DeploymentConfig, db_url: str) -> dict:
    """Generate service-specific environment variables."""
    # Get the port for this service
    service_port = SERVICES[service_name]["port"] - 8000  # Convert external port to internal port (8000)
    
    # Base environment for all services
    env = {
        "DATABASE_URL": db_url,
        "SECRET_KEY": config.secret_key,
        "HOST": "0.0.0.0",
        "PORT": "8000",  # All services run on 8000 internally
    }
    
    if service_name == "atlas":
        # Set MongoDB URL based on mongo_type
        mongodb_url = config.mongodb_url if config.mongo_type == MongoType.EXTERNAL else "mongodb://pulsr_mongo:27017"
        env.update({
            "MONGODB_URL": mongodb_url,
            "MONGODB_DB": config.mongodb_db,
            "LINGUA_API_URL": "http://pulsr_lingua:8000"
        })
    
    elif service_name == "census":
        env.update({
            "ALGORITHM": config.algorithm,
            "ACCESS_TOKEN_EXPIRE_MINUTES": str(config.access_token_expire_minutes),
            "OTP_EXPIRE_MINUTES": str(config.otp_expire_minutes),
            "ADMIN_EMAIL": config.admin_email,
            "HERMES_BASE": "http://pulsr_hermes:8000",
            "HERMES_API_KEY": config.hermes_api_key,
            "HERMES_FROM_EMAIL": config.hermes_from_email
        })
    
    elif service_name == "hermes":
        env.update({
            "SMTP_HOST": config.smtp_host,
            "SMTP_PORT": str(config.smtp_port),
            "SMTP_DOMAIN": config.smtp_domain,
            "DKIM_SELECTOR": config.dkim_selector,
            "API_KEY": config.hermes_api_key
        })
        
        # Optional SMTP settings
        if config.outbound_smtp_host:
            env["OUTBOUND_SMTP_HOST"] = config.outbound_smtp_host
        if config.outbound_smtp_user:
            env["OUTBOUND_SMTP_USER"] = config.outbound_smtp_user
        if config.outbound_smtp_password:
            env["OUTBOUND_SMTP_PASSWORD"] = config.outbound_smtp_password
        
        env.update({
            "OUTBOUND_SMTP_PORT": str(config.outbound_smtp_port),
            "OUTBOUND_SMTP_USE_TLS": str(config.outbound_smtp_use_tls).lower()
        })
    
    elif service_name == "lingua":
        env.update({
            "REDIS_URL": config.redis_url,
            "DEFAULT_LLM_PROVIDER": config.default_llm_provider,
            "OTEL_SERVICE_NAME": config.otel_service_name,
            "RELOAD": "false"
        })
        
        # Optional LLM provider settings
        if config.openai_api_key:
            env["OPENAI_API_KEY"] = config.openai_api_key
        if config.openai_api_base_url:
            env["OPENAI_API_BASE_URL"] = config.openai_api_base_url
        if config.anthropic_api_key:
            env["ANTHROPIC_API_KEY"] = config.anthropic_api_key
        if config.default_model:
            env["DEFAULT_MODEL"] = config.default_model
        if config.local_llm_endpoint:
            env["LOCAL_LLM_ENDPOINT"] = config.local_llm_endpoint
        if config.private_cloud_endpoint:
            env["PRIVATE_CLOUD_ENDPOINT"] = config.private_cloud_endpoint
        if config.private_cloud_api_key:
            env["PRIVATE_CLOUD_API_KEY"] = config.private_cloud_api_key
        if config.otel_exporter_otlp_endpoint:
            env["OTEL_EXPORTER_OTLP_ENDPOINT"] = config.otel_exporter_otlp_endpoint
    
    # Lambda and Nexus use the base environment
    
    return env

def update_deployment_step(deployment_id: str, step_name: str, status: str, message: str = None, error: str = None):
    """Update deployment step status."""
    if deployment_id not in deployment_progress:
        return
    
    progress = deployment_progress[deployment_id]
    
    # Find and update the step
    for step in progress["steps"]:
        if step["name"] == step_name:
            step["status"] = status
            step["message"] = message
            if status == "in_progress" and not step.get("started_at"):
                step["started_at"] = datetime.utcnow()
            elif status in ["completed", "failed"]:
                step["completed_at"] = datetime.utcnow()
            if error:
                step["error"] = error
            break
    
    # Update overall progress
    if status == "in_progress":
        progress["current_step"] = step_name
        progress["status"] = "in_progress"
    elif status == "failed":
        progress["status"] = "failed"
        progress["completed_at"] = datetime.utcnow()
        progress["error"] = error
    elif status == "completed":
        # Check if all steps are completed
        if all(s["status"] == "completed" for s in progress["steps"]):
            progress["status"] = "completed"
            progress["completed_at"] = datetime.utcnow()
            progress["current_step"] = None

async def deploy_services_background(config: DeploymentConfig, deployment_id: str):
    """Deploy all services using Docker SDK directly with progress tracking."""
    try:
        logger.info(f"Starting deployment {deployment_id}")
        
        # Fixed registry URL
        registry_url = "rg.nl-ams.scw.cloud/pulsr-core"
    
        # Step 1: Registry login
        update_deployment_step(deployment_id, "registry_login", "in_progress", "Logging into container registry")
        if config.scaleway_secret:
            try:
                docker_client.login(
                    username="nologin",
                    password=config.scaleway_secret,
                    registry="rg.nl-ams.scw.cloud"
                )
                update_deployment_step(deployment_id, "registry_login", "completed", "Registry login successful")
            except Exception as e:
                error_msg = f"Registry login failed: {str(e)}"
                logger.error(error_msg)
                update_deployment_step(deployment_id, "registry_login", "failed", error=error_msg)
                raise Exception(error_msg)
        else:
            update_deployment_step(deployment_id, "registry_login", "completed", "No registry authentication required")
        
        # Step 2: Network setup
        update_deployment_step(deployment_id, "network_setup", "in_progress", "Creating Docker network")
        try:
            network = create_network()
            update_deployment_step(deployment_id, "network_setup", "completed", "Network created successfully")
        except Exception as e:
            error_msg = f"Network creation failed: {str(e)}"
            logger.error(error_msg)
            update_deployment_step(deployment_id, "network_setup", "failed", error=error_msg)
            raise Exception(error_msg)
        
        # Step 3: Database setup
        postgres_container = None
        if config.database_type == DatabaseType.MANAGED:
            update_deployment_step(deployment_id, "database_setup", "in_progress", "Setting up PostgreSQL database")
            try:
                # Create volume
                try:
                    docker_client.volumes.get("postgres_data")
                except docker.errors.NotFound:
                    docker_client.volumes.create("postgres_data")
                
                # Create init script
                init_script_path = create_postgres_init_script()
                
                # Pull PostgreSQL image
                logger.info("Pulling PostgreSQL image")
                docker_client.images.pull("postgres:15")
                
                # Start PostgreSQL container
                postgres_container = docker_client.containers.run(
                    "postgres:15",
                    name="pulsr_postgres",
                    environment={
                        "POSTGRES_USER": "pulsr",
                        "POSTGRES_PASSWORD": config.postgres_password,
                        "POSTGRES_DB": "postgres"
                    },
                    ports={"5432/tcp": 5432},
                    volumes={
                        "postgres_data": {"bind": "/var/lib/postgresql/data", "mode": "rw"},
                        init_script_path: {"bind": "/docker-entrypoint-initdb.d/init.sh", "mode": "ro"}
                    },
                    network=NETWORK_NAME,
                    restart_policy={"Name": "unless-stopped"},
                    detach=True
                )
                
                # Wait for PostgreSQL to be ready and verify databases
                logger.info("Waiting for PostgreSQL to be ready...")
                await asyncio.sleep(15)  # Give more time for init scripts
                
                # Verify that databases were created
                try:
                    await verify_databases_created(config.postgres_password)
                    update_deployment_step(deployment_id, "database_setup", "completed", "PostgreSQL databases ready")
                except Exception as db_error:
                    # If databases weren't created by init script, create them manually
                    logger.warning(f"Init script may have failed, creating databases manually: {db_error}")
                    await create_databases_manually(config.postgres_password)
                    update_deployment_step(deployment_id, "database_setup", "completed", "PostgreSQL databases created manually")
                
            except Exception as e:
                error_msg = f"PostgreSQL setup failed: {str(e)}"
                logger.error(error_msg)
                update_deployment_step(deployment_id, "database_setup", "failed", error=error_msg)
                raise Exception(error_msg)
        else:
            update_deployment_step(deployment_id, "database_setup", "completed", "Using external database")
        
        # Step 4: MongoDB setup
        if config.mongo_type == MongoType.MANAGED:
            update_deployment_step(deployment_id, "mongodb_setup", "in_progress", "Setting up MongoDB")
            try:
                # Create MongoDB volume
                try:
                    docker_client.volumes.get("mongo_data")
                except docker.errors.NotFound:
                    docker_client.volumes.create("mongo_data")
                
                # Pull and start MongoDB
                logger.info("Pulling MongoDB image")
                docker_client.images.pull("mongo:7")
                docker_client.containers.run(
                    "mongo:7",
                    name="pulsr_mongo",
                    ports={"27017/tcp": 27017},
                    volumes={"mongo_data": {"bind": "/data/db", "mode": "rw"}},
                    network=NETWORK_NAME,
                    restart_policy={"Name": "unless-stopped"},
                    detach=True
                )
                update_deployment_step(deployment_id, "mongodb_setup", "completed", "MongoDB ready")
            except Exception as e:
                error_msg = f"MongoDB setup failed: {str(e)}"
                logger.error(error_msg)
                update_deployment_step(deployment_id, "mongodb_setup", "failed", error=error_msg)
                raise Exception(error_msg)
        else:
            update_deployment_step(deployment_id, "mongodb_setup", "completed", "Using external MongoDB")
        
        # Step 5: Redis setup
        if config.redis_url and "pulsr_redis" in config.redis_url:
            update_deployment_step(deployment_id, "redis_setup", "in_progress", "Setting up Redis")
            try:
                # Pull and start Redis
                logger.info("Pulling Redis image")
                docker_client.images.pull("redis:7-alpine")
                docker_client.containers.run(
                    "redis:7-alpine",
                    name="pulsr_redis",
                    ports={"6379/tcp": 6379},
                    network=NETWORK_NAME,
                    restart_policy={"Name": "unless-stopped"},
                    detach=True
                )
                update_deployment_step(deployment_id, "redis_setup", "completed", "Redis ready")
            except Exception as e:
                error_msg = f"Redis setup failed: {str(e)}"
                logger.error(error_msg)
                update_deployment_step(deployment_id, "redis_setup", "failed", error=error_msg)
                raise Exception(error_msg)
        else:
            update_deployment_step(deployment_id, "redis_setup", "completed", "Using external Redis")
        
        # Wait a bit for infrastructure services to start
        logger.info("Waiting for infrastructure services...")
        await asyncio.sleep(5)
        
        # Step 6: Deploy microservices
        update_deployment_step(deployment_id, "services_deployment", "in_progress", "Deploying microservices")
        deployed_services = []
        
        try:
            for service_name, service_config in SERVICES.items():
                logger.info(f"Deploying service: {service_name}")
                
                # Construct database URL
                if config.database_type == DatabaseType.MANAGED:
                    db_url = f"postgresql://pulsr:{config.postgres_password}@pulsr_postgres:5432/{service_name}"
                else:
                    db_url = config.database_url
                
                # Get service-specific environment variables
                environment = get_service_environment(service_name, config, db_url)
                
                # Pull service image
                image_name = f"{registry_url}/{service_name}:{config.tag}"
                logger.info(f"Pulling image: {image_name}")
                docker_client.images.pull(image_name)
                
                # Start service container
                logger.info(f"Starting container: pulsr_{service_name}")
                container = docker_client.containers.run(
                    image_name,
                    name=f"pulsr_{service_name}",
                    environment=environment,
                    ports={"8000/tcp": service_config["port"]},
                    network=NETWORK_NAME,
                    restart_policy={"Name": "unless-stopped"},
                    detach=True
                )
                deployed_services.append(service_name)
                logger.info(f"Service {service_name} deployed successfully")
                
            update_deployment_step(deployment_id, "services_deployment", "completed", f"All {len(deployed_services)} services deployed")
            
        except Exception as e:
            error_msg = f"Service deployment failed: {str(e)}"
            logger.error(error_msg)
            logger.error(f"Traceback: {traceback.format_exc()}")
            update_deployment_step(deployment_id, "services_deployment", "failed", error=error_msg)
            raise Exception(error_msg)
        
        # Step 7: Finalize deployment
        update_deployment_step(deployment_id, "finalization", "in_progress", "Finalizing deployment")
        
        # Update deployment state
        deployment_state["deployed"] = True
        deployment_state["deployment_time"] = datetime.utcnow()
        deployment_state["configuration"] = config.dict(exclude={"postgres_password", "scaleway_secret"})
        deployment_state["database_type"] = config.database_type
        deployment_state["mongo_type"] = config.mongo_type
        
        # Save configuration to persistent storage
        save_deployment_config(config)
        
        update_deployment_step(deployment_id, "finalization", "completed", "Deployment completed successfully")
        logger.info(f"Deployment {deployment_id} completed successfully")
        
    except Exception as e:
        error_msg = f"Deployment failed: {str(e)}"
        logger.error(f"Deployment {deployment_id} failed: {error_msg}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        
        # Update progress with failure
        if deployment_id in deployment_progress:
            deployment_progress[deployment_id]["status"] = "failed"
            deployment_progress[deployment_id]["error"] = error_msg
            deployment_progress[deployment_id]["completed_at"] = datetime.utcnow()
        
        # Cleanup on failure
        try:
            await cleanup_deployment()
            logger.info("Cleanup completed after deployment failure")
        except Exception as cleanup_error:
            logger.error(f"Cleanup failed: {str(cleanup_error)}")
        
        # Don't raise HTTPException here since this runs in background

async def cleanup_deployment():
    """Stop and remove all deployed containers."""
    container_names = (
        ["pulsr_postgres", "pulsr_mongo", "pulsr_redis"] + 
        [f"pulsr_{name}" for name in SERVICES.keys()]
    )
    
    for container_name in container_names:
        try:
            container = docker_client.containers.get(container_name)
            container.stop()
            container.remove()
        except docker.errors.NotFound:
            pass
        except Exception:
            pass
    
    # Remove network
    try:
        network = docker_client.networks.get(NETWORK_NAME)
        network.remove()
    except docker.errors.NotFound:
        pass
    except Exception:
        pass

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Pulsr One Core",
        "version": "2.0.0",
        "endpoints": ["/deploy", "/undeploy", "/status", "/health", "/docs"],
        "deployed": deployment_state["deployed"]
    }

@app.get("/health")
async def health():
    """Health check for the core service itself."""
    return {"status": "healthy", "service": "core"}

@app.post("/deploy")
async def deploy(config: DeploymentConfig, background_tasks: BackgroundTasks):
    """Start deployment in background."""
    if deployment_state["deployed"]:
        raise HTTPException(status_code=400, detail="Services already deployed. Use /undeploy first.")
    
    # Generate deployment ID
    deployment_id = str(uuid.uuid4())
    
    # Initialize deployment progress
    steps = [
        {"name": "registry_login", "status": "pending", "message": "Authenticate with container registry", "started_at": None, "completed_at": None, "error": None},
        {"name": "network_setup", "status": "pending", "message": "Create Docker network", "started_at": None, "completed_at": None, "error": None},
        {"name": "database_setup", "status": "pending", "message": "Set up database", "started_at": None, "completed_at": None, "error": None},
        {"name": "mongodb_setup", "status": "pending", "message": "Set up MongoDB", "started_at": None, "completed_at": None, "error": None},
        {"name": "redis_setup", "status": "pending", "message": "Set up Redis", "started_at": None, "completed_at": None, "error": None},
        {"name": "services_deployment", "status": "pending", "message": "Deploy microservices", "started_at": None, "completed_at": None, "error": None},
        {"name": "finalization", "status": "pending", "message": "Finalize deployment", "started_at": None, "completed_at": None, "error": None}
    ]
    
    deployment_progress[deployment_id] = {
        "deployment_id": deployment_id,
        "status": "starting",
        "current_step": None,
        "steps": steps,
        "started_at": datetime.utcnow(),
        "completed_at": None,
        "error": None,
        "config": config.dict(exclude={"postgres_password", "scaleway_secret"})
    }
    
    # Start deployment in background
    background_tasks.add_task(deploy_services_background, config, deployment_id)
    
    return {
        "status": "started",
        "deployment_id": deployment_id,
        "message": "Deployment started in background",
        "track_progress": f"/deployment/{deployment_id}",
        "check_status": "/status"
    }

@app.get("/deployment/{deployment_id}", response_model=DeploymentProgress)
async def get_deployment_progress(deployment_id: str):
    """Get deployment progress by ID."""
    if deployment_id not in deployment_progress:
        raise HTTPException(status_code=404, detail="Deployment not found")
    
    progress = deployment_progress[deployment_id]
    return DeploymentProgress(**progress)

@app.get("/deployments")
async def list_deployments():
    """List all deployment attempts."""
    return {
        "deployments": [
            {
                "deployment_id": dep_id,
                "status": progress["status"],
                "started_at": progress["started_at"],
                "completed_at": progress.get("completed_at"),
                "current_step": progress.get("current_step")
            }
            for dep_id, progress in deployment_progress.items()
        ]
    }

@app.delete("/deployment/{deployment_id}")
async def delete_deployment_record(deployment_id: str):
    """Delete deployment progress record."""
    if deployment_id not in deployment_progress:
        raise HTTPException(status_code=404, detail="Deployment not found")
    
    del deployment_progress[deployment_id]
    return {"message": "Deployment record deleted"}

@app.post("/cleanup")
async def force_cleanup():
    """Force cleanup of all Pulsr containers, networks, and configuration."""
    try:
        await cleanup_deployment()
        
        # Clear deployment state completely (unlike undeploy)
        deployment_state["deployed"] = False
        deployment_state["deployment_time"] = None
        deployment_state["configuration"] = None
        deployment_state["database_type"] = None
        deployment_state["mongo_type"] = None
        
        # Clear stored configuration
        clear_deployment_config()
        
        return {
            "status": "success",
            "message": "Complete cleanup completed",
            "note": "All containers, networks, and configuration removed. Volumes preserved."
        }
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        return {
            "status": "partial",
            "message": f"Cleanup partially failed: {str(e)}",
            "suggestion": "Try manual cleanup commands"
        }

@app.post("/undeploy")
async def undeploy(remove_data: bool = False):
    """Stop and remove all deployed services."""
    # Check if any containers are actually running
    running_containers = []
    for service_name in SERVICES.keys():
        container = get_container_by_name(f"pulsr_{service_name}")
        if container and container.status == "running":
            running_containers.append(service_name)
    
    # Check infrastructure containers
    infra_containers = []
    for name in ["pulsr_postgres", "pulsr_mongo", "pulsr_redis"]:
        container = get_container_by_name(name)
        if container:
            infra_containers.append(name)
    
    if not running_containers and not infra_containers:
        raise HTTPException(status_code=400, detail="No services or infrastructure containers found to undeploy")
    
    if running_containers or infra_containers:
        logger.info(f"Undeploying running services: {running_containers}, infrastructure: {infra_containers}")
    
    try:
        # Stop and remove containers
        await cleanup_deployment()
        
        # Remove data volumes if requested
        if remove_data:
            volume_names = ["postgres_data", "mongo_data"]
            for volume_name in volume_names:
                try:
                    volume = docker_client.volumes.get(volume_name)
                    volume.remove()
                except docker.errors.NotFound:
                    pass
        
        # Reset state (but keep configuration for redeployment)
        deployment_state["deployed"] = False
        deployment_state["deployment_time"] = None
        # Keep configuration, database_type, and mongo_type for redeployment
        
        return {
            "status": "undeployed",
            "message": "All services stopped and removed",
            "data_removed": remove_data
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Undeploy failed: {str(e)}"
        )

@app.get("/status", response_model=SystemStatus)
async def status():
    """Get status and health of all microservices."""
    tasks = [
        check_service_health(name, config) 
        for name, config in SERVICES.items()
    ]
    
    results = await asyncio.gather(*tasks)
    
    services_dict = {service.name: service for service in results}
    healthy_count = sum(1 for s in results if s.status == "healthy")
    unhealthy_count = len(results) - healthy_count
    
    # Check database status if deployed
    db_status = None
    mongo_status = None
    if deployment_state["deployed"]:
        if deployment_state["database_type"] == DatabaseType.MANAGED:
            pg_container = get_container_by_name("pulsr_postgres")
            if pg_container and pg_container.status == "running":
                db_status = "healthy"
            else:
                db_status = "unhealthy"
        else:
            db_status = "external"
            
        # Check MongoDB status
        if deployment_state["mongo_type"] == MongoType.MANAGED:
            mongo_container = get_container_by_name("pulsr_mongo")
            if mongo_container and mongo_container.status == "running":
                mongo_status = "healthy"
            else:
                mongo_status = "unhealthy"
        else:
            mongo_status = "external"
    
    return SystemStatus(
        deployed=deployment_state["deployed"],
        deployment_config=deployment_state["configuration"],
        database_type=deployment_state["database_type"],
        database_status=db_status,
        mongo_type=deployment_state["mongo_type"],
        mongo_status=mongo_status,
        services=services_dict,
        healthy_count=healthy_count,
        unhealthy_count=unhealthy_count,
        total_services=len(SERVICES)
    )

@app.get("/status/{service_name}")
async def service_status(service_name: str):
    """Get status of a specific service."""
    if service_name not in SERVICES:
        raise HTTPException(status_code=404, detail=f"Service '{service_name}' not found")
    
    service_info = await check_service_health(service_name, SERVICES[service_name])
    return service_info

@app.get("/logs/{service_name}")
async def get_service_logs(service_name: str, lines: int = 100):
    """Get container logs for a service."""
    if service_name not in SERVICES:
        raise HTTPException(status_code=404, detail="Service not found")
    
    container = get_container_by_name(f"pulsr_{service_name}")
    if not container:
        raise HTTPException(status_code=404, detail="Container not found")
    
    try:
        logs = container.logs(tail=lines).decode('utf-8', errors='ignore')
        return {
            "service": service_name,
            "container_id": container.short_id,
            "container_status": container.status,
            "logs": logs,
            "lines_requested": lines
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get logs: {str(e)}")

@app.get("/diagnostics/{service_name}")
async def get_service_diagnostics(service_name: str):
    """Get detailed diagnostics for a service container."""
    if service_name not in SERVICES:
        raise HTTPException(status_code=404, detail="Service not found")
    
    container = get_container_by_name(f"pulsr_{service_name}")
    if not container:
        raise HTTPException(status_code=404, detail="Container not found")
    
    try:
        # Get container details
        container.reload()  # Refresh container state
        attrs = container.attrs
        
        return {
            "service": service_name,
            "container_id": container.short_id,
            "container_name": container.name,
            "status": container.status,
            "image": attrs.get('Config', {}).get('Image'),
            "created": attrs.get('Created'),
            "started_at": attrs.get('State', {}).get('StartedAt'),
            "finished_at": attrs.get('State', {}).get('FinishedAt'),
            "exit_code": attrs.get('State', {}).get('ExitCode'),
            "error": attrs.get('State', {}).get('Error'),
            "restart_count": attrs.get('RestartCount', 0),
            "ports": attrs.get('NetworkSettings', {}).get('Ports', {}),
            "environment": attrs.get('Config', {}).get('Env', []),
            "mounts": [
                {
                    "source": mount.get('Source'),
                    "destination": mount.get('Destination'),
                    "mode": mount.get('Mode')
                }
                for mount in attrs.get('Mounts', [])
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get diagnostics: {str(e)}")

@app.get("/deployment")
async def get_deployment_info():
    """Get current deployment information."""
    return DeploymentStatus(
        deployed=deployment_state["deployed"],
        deployment_time=deployment_state["deployment_time"],
        configuration=deployment_state["configuration"],
        database_type=deployment_state["database_type"],
        database_status="managed" if deployment_state["database_type"] == DatabaseType.MANAGED else "external",
        services_deployed=list(SERVICES.keys()) if deployment_state["deployed"] else []
    )

# Detect existing deployment on startup
detect_existing_deployment()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)