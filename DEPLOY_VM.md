# Деплой на VM Cloud.ru

Сценарий для Ubuntu 22.04+ и Docker Compose.

## 1. Подключиться

```bash
ssh user1@82.202.136.67
```

## 2. Установить Docker

```bash
sudo apt update
sudo apt install -y ca-certificates curl gnupg

sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker $USER
```

Переподключиться:

```bash
exit
ssh user1@82.202.136.67
```

## 3. Забрать проект

```bash
git clone https://github.com/RazhevDaniil/project_diploma.git
cd project_diploma
git checkout version_wno_rag
```

## 4. Создать `.env`

```bash
cp .env.example .env
nano .env
```

Минимально:

```env
BACKEND_API_URL=http://backend:8000

OPENAI_API_BASE=https://foundation-models.api.cloud.ru/v1
OPENAI_API_KEY=your_foundation_models_api_key_here
OPENAI_MODEL=openai/gpt-oss-120b
OPENAI_TEMPERATURE=0.05

MANAGED_RAG_URL=https://e424a162-618c-4862-b789-b089abd81b46.managed-rag.inference.cloud.ru/api/v2/retrieve_generate
MANAGED_RAG_KB_VERSION=eb73eb63-ec91-47c9-851e-1c14949b7a14
MANAGED_RAG_API_KEY=your_managed_rag_api_key_here
MANAGED_RAG_RESULTS=2
MANAGED_RAG_CONTEXT_CHUNKS=3
MANAGED_RAG_MAX_TOKENS=256
MANAGED_RAG_TEMPERATURE=0.01
MANAGED_RAG_CONCURRENCY=4
MANAGED_RAG_CACHE_ENABLED=true
```

## 5. Подготовить директории

```bash
mkdir -p uploads reports runs prompt_versions rag_cache
```

## 6. Запустить

```bash
docker compose up -d --build
docker compose ps
curl http://127.0.0.1:8000/health
```

## 7. Открыть UI

В Cloud.ru security group откройте входящий TCP `8501`, затем:

```text
http://82.202.136.67:8501
```

Для нормального доступа лучше открыть только `80` и поставить nginx.

## 8. Nginx

```bash
sudo apt install -y nginx
sudo nano /etc/nginx/sites-available/tz-analyzer
```

```nginx
server {
    listen 80;
    server_name 82.202.136.67;

    client_max_body_size 200M;

    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 86400;
    }
}
```

```bash
sudo ln -sf /etc/nginx/sites-available/tz-analyzer /etc/nginx/sites-enabled/tz-analyzer
sudo nginx -t
sudo systemctl reload nginx
```

Открыть в Cloud.ru TCP `80`, затем зайти:

```text
http://82.202.136.67
```

## 9. Обновление

```bash
cd ~/project_diploma
git pull
docker compose up -d --build
```

## 10. Проверка

```bash
docker compose ps
docker compose logs -f backend
curl http://127.0.0.1:8000/health
```
