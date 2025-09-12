# Pulsr One Core - One-Click Microservices Deployment

Deploy the complete Pulsr One microservices architecture with a single API call.

## 🚀 One-Click Deployment

Deploy the entire Pulsr One stack programmatically via API:

```bash
# Start the core service (no configuration needed)
docker run -d \
  -p 8000:8000 \
  -v /var/run/docker.sock:/var/run/docker.sock \
  --name pulsr-core \
  your-registry/core:latest

# Deploy all services with one API call
curl -X POST http://localhost:8000/deploy \
  -H "Content-Type: application/json" \
  -d '{
    "registry_url": "rg.fr-par.scw.cloud/your-namespace",
    "tag": "latest",
    "database_type": "managed",
    "postgres_password": "secure_password",
    "secret_key": "your-secret-key",
    "api_key": "your-api-key",
    "hermes_api_key": "your-hermes-api-key"
  }'
```

## 📋 API Endpoints

### Deployment Management
- `POST /deploy` - Deploy all microservices
- `POST /undeploy?remove_data=false` - Stop and remove services
- `GET /deployment` - Get deployment info

### Monitoring
- `GET /status` - Complete system status
- `GET /status/{service_name}` - Individual service status
- `GET /health` - Core service health

### Info
- `GET /` - Service information
- `GET /docs` - Interactive API documentation

## 🏗️ Deployment Configuration

All microservice configuration is done through the API payload. The Core service has no configuration files - everything is API-driven.

### Complete Configuration Example

```json
{
  "registry_url": "rg.fr-par.scw.cloud/your-namespace",
  "tag": "latest",
  "database_type": "managed",
  "postgres_password": "secure_password",
  "scaleway_secret": "your-scw-secret-key",
  
  "secret_key": "your-secret-key-here",
  "api_key": "your-api-key-here", 
  "hermes_api_key": "your-hermes-api-key",
  
  "admin_email": "admin@yourcompany.com",
  "hermes_from_email": "noreply@yourcompany.com",
  "smtp_domain": "yourcompany.com",
  
  "openai_api_key": "sk-...",
  "anthropic_api_key": "sk-ant-...",
  "default_llm_provider": "openai"
}
```

### Minimal Configuration (Managed Database)

```json
{
  "registry_url": "rg.fr-par.scw.cloud/your-namespace",
  "database_type": "managed",
  "postgres_password": "secure_password",
  "secret_key": "your-secret-key",
  "api_key": "your-api-key",
  "hermes_api_key": "your-hermes-api-key"
}
```

### External Database Configuration

```json
{
  "registry_url": "rg.fr-par.scw.cloud/your-namespace",
  "database_type": "external",
  "database_url": "postgresql://user:pass@host:5432/dbname",
  "secret_key": "your-secret-key",
  "api_key": "your-api-key",
  "hermes_api_key": "your-hermes-api-key"
}
```

## 🎯 Services Deployed

| Service | Port | Container Name | Database |
|---------|------|----------------|----------|
| Atlas | 8001 | pulsr_atlas | atlas |
| Census | 8002 | pulsr_census | census |
| Hermes | 8003 | pulsr_hermes | hermes |
| Lambda | 8004 | pulsr_lambda | lambda |
| Lingua | 8005 | pulsr_lingua | lingua |
| Nexus | 8006 | pulsr_nexus | nexus |
| PostgreSQL* | 5432 | pulsr_postgres | postgres |

*Only deployed with `database_type: "managed"`

## 🔍 Monitoring

Check deployment status:
```bash
curl http://localhost:8000/status | jq
```

Example response:
```json
{
  "deployed": true,
  "database_type": "managed",
  "database_status": "healthy",
  "healthy_count": 6,
  "unhealthy_count": 0,
  "total_services": 6,
  "services": {
    "atlas": {
      "name": "atlas",
      "status": "healthy",
      "port": 8001,
      "container_id": "abc123"
    }
  }
}
```

## 🛠️ Running Core Service

### Production (Registry)
```bash
docker run -d \
  -p 8000:8000 \
  -v /var/run/docker.sock:/var/run/docker.sock \
  --name pulsr-core \
  --restart unless-stopped \
  your-registry/core:latest
```

### Development (Build locally)
```bash
cd core
docker build -t pulsr-core .
docker run -d \
  -p 8000:8000 \
  -v /var/run/docker.sock:/var/run/docker.sock \
  --name pulsr-core \
  pulsr-core
```

## 🔧 Registry Authentication

For Scaleway Container Registry:
```json
{
  "registry_url": "rg.fr-par.scw.cloud/your-namespace",
  "scaleway_secret": "your-scw-secret-key"
}
```

For DockerHub:
```json
{
  "registry_url": "docker.io/your-username"
}
```

## 🗑️ Cleanup

Remove all services:
```bash
curl -X POST "http://localhost:8000/undeploy?remove_data=true"
```

This stops all containers and removes data volumes.

## 💡 Perfect for SaaS

This architecture is ideal for:
- **One-click deployments** in admin portals
- **Multi-tenant** microservice hosting
- **Instant provisioning** for customers
- **Automated scaling** and management

The Core service handles all orchestration complexity, exposing simple REST APIs for deployment management.