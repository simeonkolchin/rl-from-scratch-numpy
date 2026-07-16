# ДЗ2 (Неделя 2) — RL среда + LLM агент + сервис

Этот репозиторий — полностью воспроизводимое решение ДЗ2 по курсу “Избранные темы исследований в AI”.

## Что нужно было сделать (по заданию)
1) Придумать и реализовать **среду** для LLM-агента:
   - ответ верифицируемый,
   - сложность регулируется,
   - задача решается **за один запрос** (one-shot, без многошагового взаимодействия).
2) Реализовать `Env.generate`, `Env.extract_answer`, `Verifier.verify`.
3) Насэмплировать датасет для обучения и сделать **фиксированные** тестовые датасеты разных сложностей.
4) Обучить агента через RL (GRPO) и сравнить качество с базовой моделью.

## Что реализовано здесь
### Среда: `logic_circuit`
Задача: дано описание логической схемы (вентильная схема) + конкретные значения входов `x0..x{n-1}`.  
Нужно вернуть один бит `y ∈ {0,1}` — выход схемы.

Проверка: детерминированная симуляция схемы и сравнение с `data.answer`.

Сложность регулируется параметрами:
- `num_inputs` (2..6)
- `num_gates` (2..20)
- `allowed_gates` (набор операций)
- `require_non_constant` (по умолчанию `True`)

`difficulty ∈ [1..10]` маппится на эти гиперпараметры, но вы также можете передавать их напрямую (они имеют приоритет).

## Структура репозитория
- `base/` — базовые интерфейсы и `Data`:
  - `base/env.py`
  - `base/verifier.py`
  - `base/data.py`
- `envs/logic_circuit/` — среда:
  - `envs/logic_circuit/env.py` — генерация задач
  - `envs/logic_circuit/verifier.py` — симуляция + `extract_answer`
  - `envs/logic_circuit/prompt.py` — построение английского промпта
- `datasets/` — генерация датасетов:
  - `datasets/make_test_datasets.py` — фиксированные `test_d*.jsonl`
  - `datasets/make_train_iterable.py` — сэмплируемый поток (и debug-dump)
- `training/` — обучение/оценка (GPU окружение):
  - `training/train_grpo_unsloth.py`
  - `training/eval_vllm.py`
  - `training/plot_results.py`
  - `training/requirements-train.txt`
  - `training/SYSTEM_PROMPT.txt`
- `rl-envs-service/` — FastAPI сервис:
  - `rl-envs-service/app/app.py`
  - `rl-envs-service/app/routes/*`
  - `rl-envs-service/app/schemas/*`
- `tests/` — автотесты (pytest)

## Быстрый старт - поднять сервис и потыкать руками
Требования: установлен Docker + Docker Compose.

### 1) Запуск сервиса
Из корня репозитория:

```bash
docker compose up -d --build
```

Проверка:
- Swagger UI: `http://localhost:8001/docs`
- Health-check:
  ```bash
  curl http://localhost:8001/health
  ```

### 2) Как тестировать
1) Сгенерировать задачу:
   ```bash
   curl -s http://localhost:8001/api/v1/envs/logic_circuit/generate \
     -H 'Content-Type: application/json' \
     -d '{"num_of_questions": 1, "difficulty": 3, "seed": 42}'
   ```
   Ответ содержит `items[0]` — это JSON объекта `Data` (включая `answer` и `metadata`).

2) Проверить решение:
   - Возьмите `answer` из `items[0].answer`
   - Отправьте `verify` с `test_solution`, например:
   ```bash
   curl -s http://localhost:8001/api/v1/envs/logic_circuit/verify \
     -H 'Content-Type: application/json' \
     -d '{
       "data": {"question":"...","answer":"1","difficulty":3,"metadata":{}},
       "test_solution":"<answer>1</answer>"
     }'
   ```
   В ответе будет `correct: true/false` и `extracted`.

3) Сгенерировать фиксированные тест-датасеты через API:
   ```bash
   curl -s http://localhost:8001/api/v1/datasets/test \
     -H 'Content-Type: application/json' \
     -d '{"out_dir":"datasets","difficulties":[1,3,5,7,10],"n_per_difficulty":200,"seed":1234,"force":false}'
   ```
   Файлы будут сохранены в `rl-envs-service/artifacts/datasets/` (если `out_dir` относительный).

## Автотесты (pytest)
### В Docker (без локального Python)
```bash
docker compose --profile test run --rm tests
```

### Локально
```bash
python -m pytest
```

## Датасеты (CLI)
Фиксированные тест-датасеты (создаются один раз; для перегенерации нужен `--force`):
```bash
python datasets/make_test_datasets.py --out-dir datasets --n-per-difficulty 200 --seed 1234
```

Сэмпл из train-итератора (для дебага):
```bash
python datasets/make_train_iterable.py --out datasets/train_sample.jsonl --n 100 --seed 0
```

## Обучение и оценка (GPU окружение: Colab/Kaggle/локально)
Training не контейнеризован (много GPU-зависимостей).

1) Установка зависимостей:
```bash
pip install -r training/requirements-train.txt
```

2) Обучение GRPO (unsloth):
```bash
python training/train_grpo_unsloth.py --steps 500 --output-dir artifacts/model_grpo
```

3) Оценка baseline vs finetuned (vLLM):
```bash
python training/eval_vllm.py --finetuned artifacts/model_grpo --datasets-dir datasets
```

4) График:
```bash
python training/plot_results.py --metrics artifacts/metrics.json --out artifacts/accuracy_bars.png
```

## Отчёт
Шаблон отчёта: `reports/report_template.md`