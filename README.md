# GGUF Inference Rig

A persistent RunPod GGUF model library with an authenticated Gradio control panel and one managed `llama-server` OpenAI-compatible endpoint.

## What is implemented

- GGUF discovery under `/workspace/models/gguf/<org>/<repo>`
- Hugging Face file inspection and selected-file downloads directly to the network volume
- optional `mmproj` pairing for multimodal models
- quant, file-size, disk-space, GPU and process status in the panel
- single active `llama-server` with readiness checks and graceful `SIGTERM` → `SIGKILL`
- atomic active-model persistence and automatic restore after pod restart
- rollback to the previous model if a switch fails
- API Bearer key and Gradio Basic Auth required for public listeners
- HF token read only from the environment; no secrets are written to state or logs

## RunPod deployment

Use a CUDA/PyTorch Ubuntu template, attach a network volume at `/workspace`, and expose HTTP ports `7860` and `8000`.

Create these RunPod secrets/environment variables:

```text
GGUF_API_KEY=<long random token>
GGUF_PANEL_USER=<username>
GGUF_PANEL_PASSWORD=<strong password>
HF_TOKEN=<optional, needed for gated/private repositories>
```

Готовый полный шаблон для поля **Environment variables** на показанном экране RunPod находится в [`runpod_variables.template.env`](runpod_variables.template.env). Скопируйте его в приватный рабочий файл и замените плейсхолдеры:

```bash
cp runpod_variables.template.env runpod_variables.env
```

Файл `runpod_variables.env` уже добавлен в `.gitignore`, поэтому реальные ключи и пароли не попадут в Git. После заполнения скопируйте все строки из него в редактор RunPod и нажмите **Update Variables**. Сам `runpod_variables.template.env` безопасно хранить в репозитории, пока в нём остаются только плейсхолдеры.

Обязательно замените:

- `GGUF_API_KEY` — результат команды `openssl rand -hex 32`;
- `GGUF_PANEL_PASSWORD` — отдельный сложный пароль;
- `HF_TOKEN` — токен Hugging Face или пустое значение для публичных моделей.

Для шаблона Pod также укажите HTTP-порты `7860,8000` и mount path `/workspace`.

## FAQ: RunPod Secrets и Environment Variables

### Куда добавлять переменные?

Добавьте их в настройки Pod или его Template так, чтобы внутри контейнера они были доступны как обычные переменные окружения с теми же именами. Чувствительные значения следует создавать как **Secrets**, а обычные параметры — как **Environment Variables**.

Рекомендуемое разделение:

| Имя | Куда сохранить | Назначение |
| --- | --- | --- |
| `GGUF_API_KEY` | Secret | Защищает OpenAI-совместимый API на порту `8000` |
| `GGUF_PANEL_PASSWORD` | Secret | Пароль Gradio-панели |
| `HF_TOKEN` | Secret | Доступ к gated/private моделям Hugging Face; необязателен для публичных репозиториев |
| `GGUF_PANEL_USER` | Environment Variable | Логин Gradio-панели |
| `GGUF_SKIP_INSTALL` | Environment Variable | Значение `1` пропускает повторную установку после успешного первого запуска |

Не добавляйте значения Secrets в Git, `.env.example`, startup command, URL репозитория или скриншоты. После изменения секрета перезапустите Pod.

### Для чего нужен `GGUF_API_KEY`?

Порт `8000` предоставляет `llama-server` API, совместимый с OpenAI. Если порт опубликован через RunPod, без ключа посторонний пользователь мог бы отправлять запросы к модели, занимать GPU и расходовать оплачиваемое время.

Приложение передаёт `GGUF_API_KEY` в `llama-server` как API key. Клиент должен отправлять то же значение в каждом запросе:

```http
Authorization: Bearer <GGUF_API_KEY>
```

Это отдельный случайный секрет. Он не связан с Hugging Face, OpenAI или паролем Gradio и не должен совпадать с ними.

### Как сгенерировать `GGUF_API_KEY`?

Предпочтительный вариант через OpenSSL — 32 случайных байта, представленных 64 hexadecimal-символами:

```bash
openssl rand -hex 32
```

Либо с помощью Python:

```bash
python3 -c 'import secrets; print(secrets.token_urlsafe(48))'
```

Скопируйте полученную строку в RunPod Secret с именем `GGUF_API_KEY`. Кавычки вокруг значения в интерфейсе RunPod не нужны.

Для запроса с другого компьютера временно задайте то же значение в локальной оболочке:

```bash
export GGUF_API_KEY='вставьте-сюда-сгенерированное-значение'
```

Не вводите реальный ключ непосредственно в команду `curl`: так он с большей вероятностью попадёт в shell history. Если ключ стал известен посторонним, сгенерируйте новый, обновите RunPod Secret и перезапустите Pod.

### Чем `GGUF_API_KEY` отличается от логина панели?

- `GGUF_PANEL_USER` и `GGUF_PANEL_PASSWORD` открывают веб-панель Gradio на порту `7860`.
- `GGUF_API_KEY` разрешает программные inference-запросы к `llama-server` на порту `8000`.
- `HF_TOKEN` разрешает приложению скачивать модели с Hugging Face.

Это три независимых механизма доступа. Для каждого следует использовать отдельное значение.

Paste the command from `runpod_command.txt` into the pod startup command. The first boot installs build dependencies, compiles CUDA `llama-server`, creates a persistent venv, and starts the panel. Later boots use incremental installs/builds; set `GGUF_SKIP_INSTALL=1` after the first successful build to skip the installer entirely.

The API is available on port `8000` and uses the standard header:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer $GGUF_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"current","messages":[{"role":"user","content":"Hello"}]}'
```

## Local development

Install Python dependencies, point `LLAMA_SERVER_BIN` at a local llama.cpp build, then bind both services to localhost:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
GGUF_ALLOW_INSECURE=1 \
GGUF_API_HOST=127.0.0.1 \
GGUF_PANEL_HOST=127.0.0.1 \
.venv/bin/python app.py
```

Run backend tests without Gradio or a GPU:

```bash
python3 -m pytest -q
```

## Runtime variables

| Variable | Default |
| --- | --- |
| `GGUF_VOLUME_ROOT` | `/workspace` |
| `GGUF_MODELS_DIR` | `/workspace/models/gguf` |
| `LLAMA_SERVER_BIN` | `/workspace/llama.cpp/build/bin/llama-server` |
| `GGUF_API_PORT` | `8000` |
| `GGUF_PANEL_PORT` | `7860` |
| `GGUF_HEALTH_TIMEOUT` | `180` seconds |
| `GGUF_STOP_TIMEOUT` | `15` seconds |

Do not place secrets in `active_model.json`, shell history, repository files, or model metadata. Rotate credentials through RunPod Secrets and restart the pod.
