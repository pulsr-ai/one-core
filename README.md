# Pulsr One Core - One-Click Microservices Deployment

Deploy the complete Pulsr One microservices architecture with a single API call, secured with automatic HTTPS.

## 🚀 One-Click Deployment with HTTPS

Deploy the entire Pulsr One stack programmatically via API with automatic SSL/TLS:

```bash
# Start the core service (no configuration needed)
docker run -d \
  -p 8000:8000 \
  -v /var/run/docker.sock:/var/run/docker.sock \
  --name pulsr-core \
  your-registry/core:latest

# Deploy all services with HTTPS (Let's Encrypt) - Recommended
curl -X POST http://localhost:8000/deploy \
  -H "Content-Type: application/json" \
  -d '{
    "domain": "pulsr.example.com",
    "enable_https": true,
    "tag": "latest",
    "database_type": "managed",
    "postgres_password": "secure_password",
    "secret_key": "your-secret-key",
    "admin_email": "admin@example.com",
    "hermes_api_key": "your-hermes-api-key"
  }'
```

**Services will be available at:**
- `https://atlas.pulsr.example.com`
- `https://census.pulsr.example.com`
- `https://hermes.pulsr.example.com`
- `https://lingua.pulsr.example.com`
- `https://nexus.pulsr.example.com`

### Legacy Mode (Without HTTPS)

For development or internal use:

```bash
curl -X POST http://localhost:8000/deploy \
  -H "Content-Type: application/json" \
  -d '{
    "enable_https": false,
    "tag": "latest",
    "database_type": "managed",
    "postgres_password": "secure_password",
    "secret_key": "your-secret-key",
    "admin_email": "admin@example.com",
    "hermes_api_key": "your-hermes-api-key"
  }'
```

Services will be on ports 8001-8006.

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

### Complete Configuration Example (HTTPS)

```json
{
  "domain": "pulsr.example.com",
  "enable_https": true,
  "letsencrypt_email": "admin@example.com",
  "traefik_dashboard": false,

  "tag": "latest",
  "database_type": "managed",
  "postgres_password": "secure_password",
  "scaleway_secret": "your-scw-secret-key",

  "secret_key": "your-secret-key-here",
  "hermes_api_key": "your-hermes-api-key",

  "admin_email": "admin@yourcompany.com",
  "hermes_from_email": "noreply@yourcompany.com",
  "smtp_domain": "yourcompany.com",

  "openai_api_key": "sk-...",
  "anthropic_api_key": "sk-ant-...",
  "default_llm_provider": "openai"
}
```

### Minimal Configuration with HTTPS

```json
{
  "domain": "pulsr.example.com",
  "enable_https": true,
  "database_type": "managed",
  "postgres_password": "secure_password",
  "secret_key": "your-secret-key",
  "admin_email": "admin@example.com",
  "hermes_api_key": "your-hermes-api-key"
}
```

**Note:** `letsencrypt_email` defaults to `admin_email` if not specified.

### External Database Configuration

```json
{
  "domain": "pulsr.example.com",
  "enable_https": true,
  "database_type": "external",
  "database_url": "postgresql://user:pass@host:5432/dbname",
  "secret_key": "your-secret-key",
  "admin_email": "admin@example.com",
  "hermes_api_key": "your-hermes-api-key"
}
```

### HTTPS Configuration Options

| Field | Default | Description |
|-------|---------|-------------|
| `domain` | `null` | Base domain for services (e.g., `pulsr.example.com`) - **required for HTTPS** |
| `enable_https` | `true` | Enable automatic HTTPS with Let's Encrypt |
| `letsencrypt_email` | `admin_email` | Email for Let's Encrypt notifications |
| `traefik_dashboard` | `false` | Enable Traefik dashboard at `traefik.{domain}` |

## 🎯 Services Deployed

### With HTTPS Enabled

| Service | Subdomain | Container Name | Database |
|---------|-----------|----------------|----------|
| Atlas | `atlas.{domain}` | pulsr_atlas | atlas |
| Census | `census.{domain}` | pulsr_census | census |
| Hermes | `hermes.{domain}` | pulsr_hermes | hermes |
| Lingua | `lingua.{domain}` | pulsr_lingua | lingua |
| Nexus | `nexus.{domain}` | pulsr_nexus | nexus |
| Traefik | `traefik.{domain}`* | pulsr_traefik | - |
| PostgreSQL** | - | pulsr_postgres | postgres |
| MongoDB** | - | pulsr_mongo | - |
| Redis** | - | pulsr_redis | - |

\*Only if `traefik_dashboard: true`
\*\*Only deployed with `database_type: "managed"` or `mongo_type: "managed"`

### Without HTTPS (Legacy Mode)

| Service | Port | Container Name | Database |
|---------|------|----------------|----------|
| Atlas | 8001 | pulsr_atlas | atlas |
| Census | 8002 | pulsr_census | census |
| Hermes | 8003 | pulsr_hermes | hermes |
| Lingua | 8005 | pulsr_lingua | lingua |
| Nexus | 8006 | pulsr_nexus | nexus |
| PostgreSQL* | 5432 | pulsr_postgres | postgres |

\*Only deployed with `database_type: "managed"`

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

### Production (Registry) - With HTTPS Support
```bash
docker run -d \
  -p 80:80 \
  -p 443:443 \
  -p 8000:8000 \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v pulsr_core_data:/app/data \
  --name pulsr-core \
  --restart unless-stopped \
  your-registry/core:latest
```

**Note:** Ports 80 and 443 are only needed if you deploy with HTTPS enabled. The core service needs access to these ports to expose the Traefik reverse proxy.

### Development (Build locally)
```bash
cd core
docker build -t pulsr-core .
docker run -d \
  -p 80:80 \
  -p 443:443 \
  -p 8000:8000 \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v pulsr_core_data:/app/data \
  --name pulsr-core \
  pulsr-core
```

### DNS Configuration

For HTTPS to work, you need to configure DNS records for your domain:

```
A     pulsr.example.com          → your-server-ip
A     *.pulsr.example.com        → your-server-ip
```

Or individual A records:
```
A     atlas.pulsr.example.com    → your-server-ip
A     census.pulsr.example.com   → your-server-ip
A     hermes.pulsr.example.com   → your-server-ip
A     lingua.pulsr.example.com   → your-server-ip
A     nexus.pulsr.example.com    → your-server-ip
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
- **One-click deployments** in admin portals with automatic HTTPS
- **Multi-tenant** microservice hosting with SSL certificates
- **Instant provisioning** for customers with secure endpoints
- **Automated scaling** and management
- **Production-ready** with Let's Encrypt integration

The Core service handles all orchestration complexity, including:
- ✅ Automatic HTTPS certificate provisioning and renewal
- ✅ Reverse proxy configuration (Traefik)
- ✅ HTTP to HTTPS redirects
- ✅ Subdomain routing
- ✅ Container orchestration
- ✅ Health monitoring

## 🔒 Security Features

- **Automatic HTTPS**: Let's Encrypt certificates with auto-renewal
- **TLS Challenge**: No need to open extra ports (works with 80/443 only)
- **HTTP Redirect**: All HTTP traffic automatically redirects to HTTPS
- **No Direct Port Exposure**: Services only accessible via Traefik (when HTTPS enabled)
- **Internal Container Network**: Services communicate securely on internal network

## 🐛 Troubleshooting

### Let's Encrypt Certificate Issues

1. **DNS not propagated**: Wait for DNS records to propagate (up to 48 hours, usually minutes)
2. **Rate limits**: Let's Encrypt has rate limits (50 certs/week per domain)
3. **Staging environment**: For testing, use Let's Encrypt staging (modify Traefik command)

### Check Traefik Logs

```bash
docker logs pulsr_traefik
```

### Check Service Connectivity

```bash
# From within the network
docker exec pulsr_traefik wget -q -O- http://pulsr_atlas:8000/health
```