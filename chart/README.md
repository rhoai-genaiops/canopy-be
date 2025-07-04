# Canopy Backend Helm Chart

A Helm chart for deploying the Canopy FastAPI Backend application on Kubernetes/OpenShift.

## Overview

This chart deploys the Canopy Backend application, which is a FastAPI-based service that provides LLM integration capabilities with configurable feature flags for RAG and summarization.

## Prerequisites

- Kubernetes 1.19+
- Helm 3.0+
- OpenShift 4.x+ (for Route resource)

## Installation

### Install the chart

```bash
helm install canopy-backend ./chart
```

### Install with custom values

```bash
helm install canopy-backend ./chart -f custom-values.yaml
```

### Upgrade the chart

```bash
helm upgrade canopy-backend ./chart
```

## Configuration

The following table lists the configurable parameters of the Canopy Backend chart and their default values.

| Parameter | Description | Default |
|-----------|-------------|---------|
| `image` | Container image for the backend | `quay.io/rhoai-genaiops/canopy-be:0.1` |
| `env.llamaBaseUrl` | Base URL for the LLaMA service | `http://llama32:8000` |
| `env.cloudLlm` | Cloud LLM service identifier | `llama32` |
| `env.privateLlm` | Private LLM service identifier | `llama32` |
| `env.featureFlags.ragEnabled` | Enable RAG functionality | `false` |
| `env.featureFlags.summarizationEnabled` | Enable summarization functionality | `true` |

## Environment Variables

The application uses the following environment variables:

- `LLAMA_BASE_URL`: Base URL for the LLaMA service
- `CLOUD_LLM`: Cloud LLM service identifier
- `PRIVATE_LLM`: Private LLM service identifier
- `FEATURE_RAG_ENABLED`: Enable/disable RAG functionality
- `FEATURE_SUMMARIZATION_ENABLED`: Enable/disable summarization functionality

## Resources Created

This chart creates the following Kubernetes resources:

- **Deployment**: Main application deployment with configurable replicas
- **Service**: ClusterIP service exposing port 8000
- **Route**: OpenShift route with TLS edge termination for external access

## Examples

### Basic Installation

```bash
helm install canopy-backend ./chart
```

### Custom Configuration

```yaml
# custom-values.yaml
image: quay.io/rhoai-genaiops/canopy-be:0.2

env:
  llamaBaseUrl: "http://my-llama-service:8000"
  cloudLlm: "gpt-4"
  privateLlm: "llama-7b"
  featureFlags:
    ragEnabled: "true"
    summarizationEnabled: "true"
```

```bash
helm install canopy-backend ./chart -f custom-values.yaml
```

## Uninstallation

```bash
helm uninstall canopy-backend
```

## Chart Information

- **Chart Version**: 0.1.0
- **App Version**: 1.0.0
- **Chart Type**: application

## Support

For issues and questions, please refer to the project documentation or open an issue in the project repository.

Helm chart icon is from [here](https://www.deviantart.com/pratlegacy/art/Cute-Groot-Digital-Art-Vector-Icon-762435201)