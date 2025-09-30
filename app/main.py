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
# Note: In production with HTTPS, update allow_origins to include your domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://*.pulsr.one",  # Wildcard for subdomains (note: may need specific domains in production)
    ],
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
    #"maestro": {"port": 8004, "host": "localhost", "internal_host": "maestro"},
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

    # Traefik/HTTPS configuration
    domain: Optional[str] = None  # Base domain (e.g., pulsr.example.com)
    enable_https: bool = True  # Enable HTTPS with Let's Encrypt
    letsencrypt_email: Optional[str] = None  # Defaults to admin_email if not provided
    traefik_dashboard: bool = False  # Enable Traefik dashboard

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

        # Set letsencrypt_email to admin_email if not provided
        if not self.letsencrypt_email:
            self.letsencrypt_email = self.admin_email

        # Validate HTTPS configuration
        if self.enable_https and not self.domain:
            raise ValueError("domain is required when enable_https is True")

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

# Dynamic deployment state - computed from containers and stored config
def get_deployment_state():
    """
    Compute deployment state dynamically from running containers and stored config.
    This is the single source of truth for deployment status.
    """
    # Check for running service containers
    running_services = []
    for service_name in SERVICES.keys():
        container = get_container_by_name(f"pulsr_{service_name}")
        if container and container.status == "running":
            running_services.append(service_name)
    
    # Check infrastructure containers
    postgres_running = False
    mongo_running = False
    redis_running = False
    
    postgres_container = get_container_by_name("pulsr_postgres")
    if postgres_container and postgres_container.status == "running":
        postgres_running = True
    
    mongo_container = get_container_by_name("pulsr_mongo")
    if mongo_container and mongo_container.status == "running":
        mongo_running = True
        
    redis_container = get_container_by_name("pulsr_redis")
    if redis_container and redis_container.status == "running":
        redis_running = True
    
    # Determine if we have a deployment (services + some infrastructure)
    deployed = len(running_services) > 0 and (postgres_running or mongo_running or redis_running)
    
    # Load stored configuration if available
    stored_config = load_deployment_config()
    configuration = None
    database_type = None
    mongo_type = None
    deployment_time = None
    
    if stored_config:
        configuration = stored_config.get("config", {})
        database_type = stored_config.get("database_type")
        mongo_type = stored_config.get("mongo_type") 
        deployment_time = stored_config.get("deployment_time")
        if deployment_time and isinstance(deployment_time, str):
            deployment_time = datetime.fromisoformat(deployment_time)
    else:
        # Try to recover configuration from container environment variables
        recovered_config = recover_config_from_containers()
        
        if recovered_config:
            configuration = recovered_config
            database_type = recovered_config.get("database_type", DatabaseType.MANAGED if postgres_running else DatabaseType.EXTERNAL)
            mongo_type = recovered_config.get("mongo_type", MongoType.MANAGED if mongo_running else MongoType.EXTERNAL)
            deployment_time = None  # Unknown without stored config
            logger.info("Using configuration recovered from container environment variables")
        else:
            # Final fallback: infer types from running containers
            database_type = DatabaseType.MANAGED if postgres_running else DatabaseType.EXTERNAL
            mongo_type = MongoType.MANAGED if mongo_running else MongoType.EXTERNAL
            deployment_time = None  # Unknown without stored config
            configuration = {
                "detected": True,
                "services": running_services,
                "note": "State computed from running containers (no stored config or recoverable config)"
            } if deployed else None
    
    return {
        "deployed": deployed,
        "deployment_time": deployment_time,
        "configuration": configuration,
        "database_type": database_type,
        "mongo_type": mongo_type,
        "running_services": running_services,
        "infrastructure": {
            "postgres": postgres_running,
            "mongo": mongo_running,
            "redis": redis_running
        }
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

def recover_config_from_containers():
    """Recover deployment configuration from running container environment variables."""
    try:
        recovered_config = {}
        
        # Try to get config from Census container (has most config values)
        census_container = get_container_by_name("pulsr_census")
        if census_container:
            env_vars = {}
            container_env = census_container.attrs.get('Config', {}).get('Env', [])
            for env_var in container_env:
                if '=' in env_var:
                    key, value = env_var.split('=', 1)
                    env_vars[key] = value
            
            # Extract config values from Census environment
            if 'SECRET_KEY' in env_vars:
                recovered_config['secret_key'] = env_vars['SECRET_KEY']
            if 'ADMIN_EMAIL' in env_vars:
                recovered_config['admin_email'] = env_vars['ADMIN_EMAIL']
            if 'ALGORITHM' in env_vars:
                recovered_config['algorithm'] = env_vars['ALGORITHM']
            if 'ACCESS_TOKEN_EXPIRE_MINUTES' in env_vars:
                recovered_config['access_token_expire_minutes'] = int(env_vars['ACCESS_TOKEN_EXPIRE_MINUTES'])
            if 'OTP_EXPIRE_MINUTES' in env_vars:
                recovered_config['otp_expire_minutes'] = int(env_vars['OTP_EXPIRE_MINUTES'])
            if 'HERMES_API_KEY' in env_vars:
                recovered_config['hermes_api_key'] = env_vars['HERMES_API_KEY']
            if 'HERMES_FROM_EMAIL' in env_vars:
                recovered_config['hermes_from_email'] = env_vars['HERMES_FROM_EMAIL']
        
        # Get config from Hermes container
        hermes_container = get_container_by_name("pulsr_hermes")
        if hermes_container:
            env_vars = {}
            container_env = hermes_container.attrs.get('Config', {}).get('Env', [])
            for env_var in container_env:
                if '=' in env_var:
                    key, value = env_var.split('=', 1)
                    env_vars[key] = value
            
            if 'SMTP_HOST' in env_vars:
                recovered_config['smtp_host'] = env_vars['SMTP_HOST']
            if 'SMTP_PORT' in env_vars:
                recovered_config['smtp_port'] = int(env_vars['SMTP_PORT'])
            if 'SMTP_DOMAIN' in env_vars:
                recovered_config['smtp_domain'] = env_vars['SMTP_DOMAIN']
            if 'DKIM_SELECTOR' in env_vars:
                recovered_config['dkim_selector'] = env_vars['DKIM_SELECTOR']
            if 'OUTBOUND_SMTP_HOST' in env_vars:
                recovered_config['outbound_smtp_host'] = env_vars['OUTBOUND_SMTP_HOST']
            if 'OUTBOUND_SMTP_PORT' in env_vars:
                recovered_config['outbound_smtp_port'] = int(env_vars['OUTBOUND_SMTP_PORT'])
            if 'OUTBOUND_SMTP_USER' in env_vars:
                recovered_config['outbound_smtp_user'] = env_vars['OUTBOUND_SMTP_USER']
            if 'OUTBOUND_SMTP_USE_TLS' in env_vars:
                recovered_config['outbound_smtp_use_tls'] = env_vars['OUTBOUND_SMTP_USE_TLS'].lower() == 'true'
        
        # Get config from Atlas container
        atlas_container = get_container_by_name("pulsr_atlas")
        if atlas_container:
            env_vars = {}
            container_env = atlas_container.attrs.get('Config', {}).get('Env', [])
            for env_var in container_env:
                if '=' in env_var:
                    key, value = env_var.split('=', 1)
                    env_vars[key] = value
            
            if 'MONGODB_URL' in env_vars:
                if 'pulsr_mongo' in env_vars['MONGODB_URL']:
                    recovered_config['mongo_type'] = 'managed'
                else:
                    recovered_config['mongo_type'] = 'external'
                    recovered_config['mongodb_url'] = env_vars['MONGODB_URL']
            if 'MONGODB_DB' in env_vars:
                recovered_config['mongodb_db'] = env_vars['MONGODB_DB']
        
        # Get config from Lingua container
        lingua_container = get_container_by_name("pulsr_lingua")
        if lingua_container:
            env_vars = {}
            container_env = lingua_container.attrs.get('Config', {}).get('Env', [])
            for env_var in container_env:
                if '=' in env_var:
                    key, value = env_var.split('=', 1)
                    env_vars[key] = value
            
            if 'REDIS_URL' in env_vars:
                recovered_config['redis_url'] = env_vars['REDIS_URL']
            if 'DEFAULT_LLM_PROVIDER' in env_vars:
                recovered_config['default_llm_provider'] = env_vars['DEFAULT_LLM_PROVIDER']
            if 'OTEL_SERVICE_NAME' in env_vars:
                recovered_config['otel_service_name'] = env_vars['OTEL_SERVICE_NAME']
            if 'DEFAULT_MODEL' in env_vars:
                recovered_config['default_model'] = env_vars['DEFAULT_MODEL']
            if 'LOCAL_LLM_ENDPOINT' in env_vars:
                recovered_config['local_llm_endpoint'] = env_vars['LOCAL_LLM_ENDPOINT']
            if 'PRIVATE_CLOUD_ENDPOINT' in env_vars:
                recovered_config['private_cloud_endpoint'] = env_vars['PRIVATE_CLOUD_ENDPOINT']
            if 'OTEL_EXPORTER_OTLP_ENDPOINT' in env_vars:
                recovered_config['otel_exporter_otlp_endpoint'] = env_vars['OTEL_EXPORTER_OTLP_ENDPOINT']
        
        # Determine database type from any service's DATABASE_URL
        for service_name in SERVICES.keys():
            container = get_container_by_name(f"pulsr_{service_name}")
            if container:
                env_vars = {}
                container_env = container.attrs.get('Config', {}).get('Env', [])
                for env_var in container_env:
                    if '=' in env_var:
                        key, value = env_var.split('=', 1)
                        env_vars[key] = value
                
                if 'DATABASE_URL' in env_vars:
                    db_url = env_vars['DATABASE_URL']
                    if 'pulsr_postgres' in db_url:
                        recovered_config['database_type'] = 'managed'
                    else:
                        recovered_config['database_type'] = 'external'
                        recovered_config['database_url'] = db_url
                    break
        
        # Set defaults for missing values
        if 'mongo_type' not in recovered_config:
            recovered_config['mongo_type'] = 'managed' if get_container_by_name("pulsr_mongo") else 'external'
        
        if 'database_type' not in recovered_config:
            recovered_config['database_type'] = 'managed' if get_container_by_name("pulsr_postgres") else 'external'
        
        # Add tag information (default to 'latest' since we can't determine it from containers)
        recovered_config['tag'] = 'latest'
        
        logger.info(f"Recovered configuration from containers: {list(recovered_config.keys())}")
        return recovered_config
        
    except Exception as e:
        logger.error(f"Failed to recover config from containers: {e}")
        return {}

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
            
            # Note: deployment state is now computed dynamically via get_deployment_state()
            logger.info("Deployment state will be computed dynamically from containers and stored config")
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
            # Always use container network for health checks (works with or without Traefik)
            health_urls = [
                f"http://pulsr_{name}:8000/health",  # Container network (primary)
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

        # Step 5.5: Traefik setup (if HTTPS enabled)
        if config.enable_https and config.domain:
            update_deployment_step(deployment_id, "traefik_setup", "in_progress", "Setting up Traefik reverse proxy")
            try:
                # Create volume for Let's Encrypt certificates
                try:
                    docker_client.volumes.get("traefik_certs")
                except docker.errors.NotFound:
                    docker_client.volumes.create("traefik_certs")

                # Pull Traefik image
                logger.info("Pulling Traefik image")
                docker_client.images.pull("traefik:v3.0")

                # Build Traefik command arguments
                traefik_command = [
                    "--api.dashboard=true" if config.traefik_dashboard else "--api.dashboard=false",
                    "--api.insecure=false",
                    "--providers.docker=true",
                    "--providers.docker.exposedbydefault=false",
                    "--providers.docker.network=" + NETWORK_NAME,
                    "--entrypoints.web.address=:80",
                    "--entrypoints.websecure.address=:443",
                    "--entrypoints.web.http.redirections.entrypoint.to=websecure",
                    "--entrypoints.web.http.redirections.entrypoint.scheme=https",
                    "--certificatesresolvers.letsencrypt.acme.tlschallenge=true",
                    f"--certificatesresolvers.letsencrypt.acme.email={config.letsencrypt_email}",
                    "--certificatesresolvers.letsencrypt.acme.storage=/letsencrypt/acme.json",
                    "--log.level=INFO",
                    "--accesslog=true"
                ]

                # Traefik labels for dashboard (if enabled)
                traefik_labels = {}
                if config.traefik_dashboard:
                    traefik_labels = {
                        "traefik.enable": "true",
                        f"traefik.http.routers.traefik-dashboard.rule": f"Host(`traefik.{config.domain}`)",
                        "traefik.http.routers.traefik-dashboard.entrypoints": "websecure",
                        "traefik.http.routers.traefik-dashboard.tls.certresolver": "letsencrypt",
                        "traefik.http.routers.traefik-dashboard.service": "api@internal"
                    }

                # Start Traefik container
                logger.info("Starting Traefik container")
                docker_client.containers.run(
                    "traefik:v3.0",
                    name="pulsr_traefik",
                    command=traefik_command,
                    ports={
                        "80/tcp": 80,
                        "443/tcp": 443
                    },
                    volumes={
                        "/var/run/docker.sock": {"bind": "/var/run/docker.sock", "mode": "ro"},
                        "traefik_certs": {"bind": "/letsencrypt", "mode": "rw"}
                    },
                    labels=traefik_labels,
                    network=NETWORK_NAME,
                    restart_policy={"Name": "unless-stopped"},
                    detach=True
                )

                logger.info("Traefik started successfully")
                update_deployment_step(deployment_id, "traefik_setup", "completed", "Traefik reverse proxy ready")
            except Exception as e:
                error_msg = f"Traefik setup failed: {str(e)}"
                logger.error(error_msg)
                update_deployment_step(deployment_id, "traefik_setup", "failed", error=error_msg)
                raise Exception(error_msg)
        else:
            update_deployment_step(deployment_id, "traefik_setup", "completed", "HTTPS disabled - no Traefik deployment")
        
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

                # Build Traefik labels if HTTPS is enabled
                labels = {}
                if config.enable_https and config.domain:
                    service_subdomain = f"{service_name}.{config.domain}"
                    labels = {
                        "traefik.enable": "true",
                        f"traefik.http.routers.{service_name}.rule": f"Host(`{service_subdomain}`)",
                        f"traefik.http.routers.{service_name}.entrypoints": "websecure",
                        f"traefik.http.routers.{service_name}.tls.certresolver": "letsencrypt",
                        f"traefik.http.services.{service_name}.loadbalancer.server.port": "8000"
                    }

                # Determine port mapping based on HTTPS mode
                if config.enable_https and config.domain:
                    # No direct port exposure - all traffic via Traefik
                    ports = {}
                else:
                    # Direct port exposure (legacy mode)
                    ports = {"8000/tcp": service_config["port"]}

                # Start service container
                logger.info(f"Starting container: pulsr_{service_name}")
                container = docker_client.containers.run(
                    image_name,
                    name=f"pulsr_{service_name}",
                    environment=environment,
                    ports=ports,
                    labels=labels,
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
        
        # Save configuration to persistent storage (state is now computed dynamically)
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
        ["pulsr_postgres", "pulsr_mongo", "pulsr_redis", "pulsr_traefik"] +
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
    state = get_deployment_state()
    return {
        "service": "Pulsr One Core",
        "version": "2.0.0",
        "endpoints": ["/deploy", "/undeploy", "/status", "/health", "/email-credentials", "/docs"],
        "deployed": state["deployed"]
    }

@app.get("/health")
async def health():
    """Health check for the core service itself."""
    return {"status": "healthy", "service": "core"}

@app.post("/deploy")
async def deploy(config: DeploymentConfig, background_tasks: BackgroundTasks):
    """Start deployment in background."""
    state = get_deployment_state()
    if state["deployed"]:
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
        {"name": "traefik_setup", "status": "pending", "message": "Set up Traefik reverse proxy", "started_at": None, "completed_at": None, "error": None},
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
        
        # Clear stored configuration (state is now computed dynamically)
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
    
    # If remove_data is requested, allow cleanup even with no running containers
    if not running_containers and not infra_containers and not remove_data:
        raise HTTPException(status_code=400, detail="No services or infrastructure containers found to undeploy")
    
    # Check if there are data volumes to remove when no containers are running
    volumes_exist = False
    if remove_data and not running_containers and not infra_containers:
        volume_names = ["postgres_data", "mongo_data"]
        for volume_name in volume_names:
            try:
                docker_client.volumes.get(volume_name)
                volumes_exist = True
                break
            except docker.errors.NotFound:
                continue
        
        if not volumes_exist:
            raise HTTPException(status_code=400, detail="No containers or data volumes found to undeploy")
    
    if running_containers or infra_containers:
        logger.info(f"Undeploying running services: {running_containers}, infrastructure: {infra_containers}")
    elif remove_data and volumes_exist:
        logger.info("No containers running, but removing data volumes as requested")
    
    try:
        # Stop and remove containers (if any exist)
        if running_containers or infra_containers:
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
        
        # State is now computed dynamically from containers and stored config
        # No need to manually update deployment state
        
        # Determine what was actually done
        containers_removed = bool(running_containers or infra_containers)
        volumes_removed = remove_data
        
        if containers_removed and volumes_removed:
            message = "All services stopped and removed, data volumes deleted"
        elif containers_removed:
            message = "All services stopped and removed, data volumes preserved"
        elif volumes_removed:
            message = "No containers to remove, data volumes deleted"
        else:
            message = "No action taken"
        
        return {
            "status": "undeployed",
            "message": message,
            "containers_removed": containers_removed,
            "data_removed": volumes_removed
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
    
    # Get current deployment state
    state = get_deployment_state()
    
    # Check database status if deployed
    db_status = None
    mongo_status = None
    if state["deployed"]:
        if state["database_type"] == DatabaseType.MANAGED:
            if state["infrastructure"]["postgres"]:
                db_status = "healthy"
            else:
                db_status = "unhealthy"
        else:
            db_status = "external"
            
        # Check MongoDB status
        if state["mongo_type"] == MongoType.MANAGED:
            if state["infrastructure"]["mongo"]:
                mongo_status = "healthy"
            else:
                mongo_status = "unhealthy"
        else:
            mongo_status = "external"
    
    return SystemStatus(
        deployed=state["deployed"],
        deployment_config=state["configuration"],
        database_type=state["database_type"],
        database_status=db_status,
        mongo_type=state["mongo_type"],
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
async def get_service_logs(
    service_name: str, 
    lines: int = 100,
    exclude_endpoints: Optional[str] = None,
    include_endpoints: Optional[str] = None,
    exclude_patterns: Optional[str] = None,
    include_patterns: Optional[str] = None
):
    """Get container logs for a service with optional filtering.
    
    Args:
        service_name: Name of the service
        lines: Number of log lines to retrieve
        exclude_endpoints: Comma-separated list of endpoints to exclude (e.g., "/health,/metrics")
        include_endpoints: Comma-separated list of endpoints to include only
        exclude_patterns: Comma-separated list of patterns to exclude from logs
        include_patterns: Comma-separated list of patterns to include only
    """
    if service_name not in SERVICES:
        raise HTTPException(status_code=404, detail="Service not found")
    
    container = get_container_by_name(f"pulsr_{service_name}")
    if not container:
        raise HTTPException(status_code=404, detail="Container not found")
    
    try:
        logs = container.logs(tail=lines).decode('utf-8', errors='ignore')
        original_line_count = len(logs.split('\n'))
        
        # Apply filtering if any filters are specified
        if exclude_endpoints or include_endpoints or exclude_patterns or include_patterns:
            log_lines = logs.split('\n')
            filtered_lines = []
            
            # Parse filter parameters
            exclude_endpoint_list = [ep.strip() for ep in exclude_endpoints.split(',')] if exclude_endpoints else []
            include_endpoint_list = [ep.strip() for ep in include_endpoints.split(',')] if include_endpoints else []
            exclude_pattern_list = [pat.strip() for pat in exclude_patterns.split(',')] if exclude_patterns else []
            include_pattern_list = [pat.strip() for pat in include_patterns.split(',')] if include_patterns else []
            
            for line in log_lines:
                should_include = True
                
                # Check exclude endpoints
                if exclude_endpoint_list:
                    for endpoint in exclude_endpoint_list:
                        if endpoint in line:
                            should_include = False
                            break
                
                # Check include endpoints (if specified, line must contain at least one)
                if include_endpoint_list and should_include:
                    found_include_endpoint = False
                    for endpoint in include_endpoint_list:
                        if endpoint in line:
                            found_include_endpoint = True
                            break
                    if not found_include_endpoint:
                        should_include = False
                
                # Check exclude patterns
                if exclude_pattern_list and should_include:
                    for pattern in exclude_pattern_list:
                        if pattern in line:
                            should_include = False
                            break
                
                # Check include patterns (if specified, line must contain at least one)
                if include_pattern_list and should_include:
                    found_include_pattern = False
                    for pattern in include_pattern_list:
                        if pattern in line:
                            found_include_pattern = True
                            break
                    if not found_include_pattern:
                        should_include = False
                
                if should_include:
                    filtered_lines.append(line)
            
            logs = '\n'.join(filtered_lines)
        
        return {
            "service": service_name,
            "container_id": container.short_id,
            "container_status": container.status,
            "logs": logs,
            "lines_requested": lines,
            "original_line_count": original_line_count,
            "filtered_line_count": len(logs.split('\n')) if logs else 0,
            "filters_applied": {
                "exclude_endpoints": exclude_endpoints,
                "include_endpoints": include_endpoints,
                "exclude_patterns": exclude_patterns,
                "include_patterns": include_patterns
            }
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
    state = get_deployment_state()
    return DeploymentStatus(
        deployed=state["deployed"],
        deployment_time=state["deployment_time"],
        configuration=state["configuration"],
        database_type=state["database_type"],
        database_status="managed" if state["database_type"] == DatabaseType.MANAGED else "external",
        services_deployed=list(SERVICES.keys()) if state["deployed"] else []
    )

@app.post("/recover-config")
async def recover_deployment_config():
    """Manually recover and save deployment configuration from running containers."""
    try:
        # Check if containers are running
        running_services = []
        for service_name in SERVICES.keys():
            container = get_container_by_name(f"pulsr_{service_name}")
            if container and container.status == "running":
                running_services.append(service_name)
        
        if not running_services:
            raise HTTPException(status_code=400, detail="No running services found to recover configuration from")
        
        # Recover configuration from containers
        recovered_config = recover_config_from_containers()
        
        if not recovered_config:
            raise HTTPException(status_code=500, detail="Failed to recover configuration from containers")
        
        # Create a temporary DeploymentConfig object to validate the recovered config
        try:
            # Add required fields with defaults if missing
            if 'secret_key' not in recovered_config:
                raise HTTPException(status_code=400, detail="Could not recover SECRET_KEY from containers")
            if 'hermes_api_key' not in recovered_config:
                raise HTTPException(status_code=400, detail="Could not recover HERMES_API_KEY from containers")
            
            # Create config object (this will validate the structure)
            deployment_config = DeploymentConfig(**recovered_config)
            
            # Save the recovered configuration
            save_deployment_config(deployment_config)
            
            return {
                "status": "success",
                "message": "Configuration recovered and saved from running containers",
                "recovered_fields": list(recovered_config.keys()),
                "running_services": running_services,
                "config_preview": {k: v for k, v in recovered_config.items() if k not in ['secret_key', 'hermes_api_key', 'postgres_password', 'scaleway_secret']}
            }
            
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Recovered configuration is invalid: {str(e)}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to recover configuration: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to recover configuration: {str(e)}")

# Detect existing deployment on startup
detect_existing_deployment()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)