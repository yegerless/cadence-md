# Production Secrets

Create `cadence-md-secret` outside git before applying workloads. Keep real
values in your secret manager or pass them directly to `kubectl` from a secure
environment.

```bash
kubectl -n cadence-md create secret generic cadence-md-secret \
  --from-literal=POSTGRES_PASSWORD='<postgres-password>' \
  --from-literal=DATABASE_URL='postgresql+asyncpg://cadence_md:<postgres-password>@postgres:5432/cadence_md' \
  --from-literal=JWT_SECRET='<random-secret-at-least-32-characters>' \
  --from-literal=QDRANT__SERVICE__API_KEY='<qdrant-api-key>' \
  --from-literal=MODEL_INFERENCE_API_KEY='<inference-api-key>' \
  --from-literal=S3_KEY_ID='<yandex-s3-access-key-id>' \
  --from-literal=S3_KEY='<yandex-s3-secret-access-key>' \
  --from-literal=GRAFANA_ADMIN_PASSWORD='<grafana-admin-password>' \
  --from-literal=FLOWER_BASIC_AUTH='admin:<flower-password>' \
  --from-literal=LANGFUSE_PUBLIC_KEY='' \
  --from-literal=LANGFUSE_SECRET_KEY='' \
  --from-literal=HF_TOKEN=''
```

Do not commit generated `Secret` manifests with real production values.
