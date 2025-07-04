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
| `LLAMA_STACK_URL` | Base URL for the LLaMA Stack service | `http://llama-stack` |
| `summarize.enabled` | Enable summarization functionality | `false` |
| `summarize.model` | Model to use for summarization | `llama32` |
| `summarize.prompt` | Prompt template for summarization | `Give me a good summary of the following text.` |
| `RAG.enabled` | Enable RAG functionality | `false` |
| `RAG.model` | Model to use for RAG | `llama32` |
| `RAG.prompt` | Prompt template for RAG | `You are a helpful assistant that answers questions based on the provided context. Use only the information from the context to answer.` |

## Values Structure

The chart uses a structured values file that gets mounted as a ConfigMap. The entire values structure is available to the application as a configuration file at `/canopy/canopy-config.yaml`.

### Environment Variables

The application uses the following environment variables:

- `LLAMA_BASE_URL`: Base URL for the LLaMA Stack service (sourced from `LLAMA_STACK_URL`)

## Resources Created

This chart creates the following Kubernetes resources:

- **Deployment**: Main application deployment with single replica
- **Service**: ClusterIP service exposing port 8000
- **Route**: OpenShift route for external access
- **ConfigMap**: Contains the entire values structure as `canopy-config.yaml`

## Examples

### Basic Installation

```bash
helm install canopy-backend ./chart
```

### Custom Configuration

```yaml
# custom-values.yaml
LLAMA_STACK_URL: "http://my-llama-service:8000"

summarize:
  enabled: true
  model: "llama3.2"
  prompt: |
    Provide a comprehensive summary of the following text,
    highlighting key points and main themes.

RAG:
  enabled: true
  model: "llama3.2"
  prompt: |
    Answer the following question based on the provided context.
    If the context doesn't contain relevant information, say so.
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