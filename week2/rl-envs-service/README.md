# RL Envs Service (Logic Circuit)

FastAPI сервис для генерации и верификации one-shot задач из среды `logic_circuit`.

## Запуск (Docker)

```bash
cd rl-envs-service
docker compose up -d --build
```

Сервис доступен на `http://localhost:8001`.

## Эндпоинты

- `GET /health`
- `GET /api/v1/envs`
- `POST /api/v1/envs/{env_name}/generate`
- `POST /api/v1/envs/{env_name}/verify`
- `POST /api/v1/datasets/test` (генерация фиксированных тест-наборов в `rl-envs-service/artifacts/datasets/`)

## Примеры запросов

Generate:

```bash
curl -s http://localhost:8001/api/v1/envs/logic_circuit/generate \\
  -H 'Content-Type: application/json' \\
  -d '{"num_of_questions": 2, "difficulty": 3, "seed": 42}'
```

Verify:

```bash
curl -s http://localhost:8001/api/v1/envs/logic_circuit/verify \\
  -H 'Content-Type: application/json' \\
  -d '{"data": {"question":"...", "answer":"1", "difficulty":3, "metadata":{}}, "test_solution":"<answer>1</answer>"}'
```

