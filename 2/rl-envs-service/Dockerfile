FROM python:3.12.7
WORKDIR /app

COPY rl-envs-service/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY rl-envs-service/app /app/app
COPY rl-envs-service/openapi_spec /app/openapi_spec

# Copy environment library code (root-level packages)
COPY base /app/base
COPY envs /app/envs
COPY datasets /app/datasets

EXPOSE 8001
ENTRYPOINT ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8001"]
