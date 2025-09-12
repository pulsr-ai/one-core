from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Optional, List
import httpx
import asyncio
import docker
import os
import json
from datetime import datetime
from enum import Enum
from pathlib import Path

app = FastAPI(title="Pulsr Core Service", version="2.0.0")

# Docker client
docker_client = docker.from_env()

# Service configuration
SERVICES = {
    "atlas": {"port": 8001, "host": "atlas"},
    "census": {"port": 8002, "host": "census"},
    "hermes": {"port": 8003, "host": "hermes"},
    "lambda": {"port": 8004, "host": "lambda"},
    "lingua": {"port": 8005, "host": "lingua"},
    "nexus": {"port": 8006, "host": "nexus"},
}

# Network name for all services
NETWORK_NAME = "pulsr-network"

class DatabaseType(str, Enum):
    MANAGED = "managed"  # We create and manage the PostgreSQL container
    EXTERNAL = "external"  # User provides DATABASE_URL

class DeploymentConfig(BaseModel):
    tag: str = "latest"
    database_type: DatabaseType = DatabaseType.MANAGED
    database_url: Optional[str] = None  # Required if database_type is EXTERNAL
    postgres_password: Optional[str] = None  # Required if database_type is MANAGED
    scaleway_secret: Optional[str] = None  # For Scaleway registry auth
    
    # Shared environment variables
    secret_key: str
    
    # Atlas-specific
    mongodb_url: Optional[str] = "mongodb://pulsr_mongo:27017"
    mongodb_db: str = "atlas_documents"
    
    # Census-specific
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    otp_expire_minutes: int = 5
    admin_email: str = "admin@example.com"
    hermes_api_key: str
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
    api_key: str
    
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

class SystemStatus(BaseModel):
    deployed: bool
    deployment_config: Optional[Dict] = None
    database_type: Optional[DatabaseType] = None
    database_status: Optional[str] = None
    services: Dict[str, ServiceStatus]
    healthy_count: int
    unhealthy_count: int
    total_services: int

# Store deployment state
deployment_state = {
    "deployed": False,
    "deployment_time": None,
    "configuration": None,
    "database_type": None
}

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
        host=config["host"],
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
        async with httpx.AsyncClient(timeout=5.0) as client:
            health_url = f"http://{config['host']}:{config['port']}/health"
            response = await client.get(health_url)
            
            if response.status_code == 200:
                service_status.status = "healthy"
                try:
                    service_status.health = response.json()
                except:
                    service_status.health = {"raw": response.text}
            else:
                service_status.status = "unhealthy"
                service_status.error = f"HTTP {response.status_code}"
                
    except httpx.ConnectError:
        service_status.status = "unreachable"
        service_status.error = "Connection failed"
    except httpx.TimeoutException:
        service_status.status = "timeout"
        service_status.error = "Request timeout"
    except Exception as e:
        service_status.status = "error"
        service_status.error = str(e)
    
    return service_status

def create_network():
    """Create Docker network for services."""
    try:
        return docker_client.networks.get(NETWORK_NAME)
    except docker.errors.NotFound:
        return docker_client.networks.create(NETWORK_NAME, driver="bridge")

def create_postgres_init_script():
    """Create PostgreSQL initialization script."""
    init_sql = """-- Create separate databases for each microservice
CREATE DATABASE atlas;
CREATE DATABASE census;
CREATE DATABASE hermes;
CREATE DATABASE lambda;
CREATE DATABASE lingua;
CREATE DATABASE nexus;

-- Grant all privileges
GRANT ALL PRIVILEGES ON DATABASE atlas TO pulsr;
GRANT ALL PRIVILEGES ON DATABASE census TO pulsr;
GRANT ALL PRIVILEGES ON DATABASE hermes TO pulsr;
GRANT ALL PRIVILEGES ON DATABASE lambda TO pulsr;
GRANT ALL PRIVILEGES ON DATABASE lingua TO pulsr;
GRANT ALL PRIVILEGES ON DATABASE nexus TO pulsr;"""
    
    # Create temp file for init script
    init_path = "/tmp/init-databases.sql"
    with open(init_path, "w") as f:
        f.write(init_sql)
    return init_path

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
        env.update({
            "MONGODB_URL": config.mongodb_url,
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
            "API_KEY": config.api_key
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

async def deploy_services(config: DeploymentConfig):
    """Deploy all services using Docker SDK directly."""
    # Fixed registry URL
    registry_url = "rg.nl-ams.scw.cloud/pulsr-core"
    
    # Login to registry if needed
    if config.scaleway_secret:
        try:
            docker_client.login(
                username="nologin",
                password=config.scaleway_secret,
                registry="rg.nl-ams.scw.cloud"
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Registry login failed: {str(e)}")
    
    try:
        # Create network
        network = create_network()
        
        # Create volume for PostgreSQL if managed
        postgres_container = None
        if config.database_type == DatabaseType.MANAGED:
            # Create volume
            try:
                docker_client.volumes.get("postgres_data")
            except docker.errors.NotFound:
                docker_client.volumes.create("postgres_data")
            
            # Create init script
            init_script_path = create_postgres_init_script()
            
            # Pull PostgreSQL image
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
                    init_script_path: {"bind": "/docker-entrypoint-initdb.d/init.sql", "mode": "ro"}
                },
                network=NETWORK_NAME,
                restart_policy={"Name": "unless-stopped"},
                detach=True
            )
            
            # Wait for PostgreSQL to be ready
            await asyncio.sleep(10)
        
        # Deploy MongoDB for Atlas
        if config.mongodb_url and "pulsr_mongo" in config.mongodb_url:
            # Create MongoDB volume
            try:
                docker_client.volumes.get("mongo_data")
            except docker.errors.NotFound:
                docker_client.volumes.create("mongo_data")
            
            # Pull and start MongoDB
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
        
        # Deploy Redis for Lingua
        if config.redis_url and "pulsr_redis" in config.redis_url:
            # Pull and start Redis
            docker_client.images.pull("redis:7-alpine")
            docker_client.containers.run(
                "redis:7-alpine",
                name="pulsr_redis",
                ports={"6379/tcp": 6379},
                network=NETWORK_NAME,
                restart_policy={"Name": "unless-stopped"},
                detach=True
            )
        
        # Wait a bit for services to start
        await asyncio.sleep(5)
        
        # Deploy each microservice
        deployed_services = []
        for service_name, service_config in SERVICES.items():
            # Construct database URL
            if config.database_type == DatabaseType.MANAGED:
                db_url = f"postgresql://pulsr:{config.postgres_password}@pulsr_postgres:5432/{service_name}"
            else:
                db_url = config.database_url
            
            # Get service-specific environment variables
            environment = get_service_environment(service_name, config, db_url)
            
            # Pull service image
            image_name = f"{registry_url}/{service_name}:{config.tag}"
            docker_client.images.pull(image_name)
            
            # Start service container
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
        
        # Update deployment state
        deployment_state["deployed"] = True
        deployment_state["deployment_time"] = datetime.utcnow()
        deployment_state["configuration"] = config.dict(exclude={"postgres_password", "scaleway_secret"})
        deployment_state["database_type"] = config.database_type
        
    except Exception as e:
        # Cleanup on failure
        try:
            await cleanup_deployment()
        except:
            pass
        raise HTTPException(status_code=500, detail=f"Deployment failed: {str(e)}")

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
    """Deploy all microservices."""
    if deployment_state["deployed"]:
        raise HTTPException(status_code=400, detail="Services already deployed. Use /undeploy first.")
    
    # Validate configuration
    if config.database_type == DatabaseType.EXTERNAL and not config.database_url:
        raise HTTPException(status_code=400, detail="database_url required for external database")
    
    if config.database_type == DatabaseType.MANAGED and not config.postgres_password:
        raise HTTPException(status_code=400, detail="postgres_password required for managed database")
    
    # Start deployment
    try:
        await deploy_services(config)
        return {
            "status": "deployed",
            "message": "Services are being deployed",
            "database_type": config.database_type,
            "services": list(SERVICES.keys()),
            "check_status": "/status"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/undeploy")
async def undeploy(remove_data: bool = False):
    """Stop and remove all deployed services."""
    if not deployment_state["deployed"]:
        raise HTTPException(status_code=400, detail="No services deployed")
    
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
        
        # Reset state
        deployment_state["deployed"] = False
        deployment_state["deployment_time"] = None
        deployment_state["configuration"] = None
        deployment_state["database_type"] = None
        
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
    if deployment_state["deployed"]:
        if deployment_state["database_type"] == DatabaseType.MANAGED:
            pg_container = get_container_by_name("pulsr_postgres")
            if pg_container and pg_container.status == "running":
                db_status = "healthy"
            else:
                db_status = "unhealthy"
        else:
            db_status = "external"
    
    return SystemStatus(
        deployed=deployment_state["deployed"],
        deployment_config=deployment_state["configuration"],
        database_type=deployment_state["database_type"],
        database_status=db_status,
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)