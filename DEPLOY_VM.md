# Деплой на виртуальную машину Cloud.ru

Ниже сценарий для Ubuntu 22.04+, если проект хранится в GitHub, а UI и backend запускаются через Docker Compose.

## 1. Подготовить репозиторий локально

Проверьте, что в Git не попадают:

- `.env`
- `venv/`
- `crawl_cache/`
- `uploads/`
- `reports/`
- `faiss_index/`

Важно: `faiss_index/index.faiss` обычно больше лимита GitHub на обычные файлы. Эту папку лучше переносить отдельно через `rsync` или пересобирать на VM.

## 2. Запушить код в GitHub

```bash
cd /Users/daniil/Desktop/project_diploma
git init
git add .
git commit -m "Add dockerized backend and UI"
git branch -M main
git remote add origin https://github.com/<username>/<repo>.git
git push -u origin main
```

## 3. Подключиться к VM

```bash
ssh user1@82.202.136.67
```

## 4. Установить Docker и Compose plugin

```bash
sudo apt update
sudo apt install -y ca-certificates curl gnupg

sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo \"$VERSION_CODENAME\") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker $USER
```

После установки переподключитесь:

```bash
exit
ssh user1@82.202.136.67
```

Проверка:

```bash
docker --version
docker compose version
```

## 5. Склонировать проект

```bash
git clone https://github.com/<username>/<repo>.git
cd <repo>
```

## 6. Создать `.env`

```bash
cp .env.example .env
nano .env
```

Минимальный пример:

```env
BACKEND_API_URL=http://backend:8000
OPENAI_API_BASE=https://foundation-models.api.cloud.ru/v1
OPENAI_API_KEY=ваш_api_key
OPENAI_MODEL=GigaChat/GigaChat-2-Max
OPENAI_EMBEDDING_MODEL=BAAI/bge-m3
CHUNK_SIZE=500
CHUNK_OVERLAP=80
TOP_K_RESULTS=5
CRAWL_MAX_PAGES=0
CRAWL_CONCURRENCY=10
CRAWL_DELAY=0.2
```

## 7. Подготовить директории данных

```bash
mkdir -p faiss_index crawl_cache uploads reports knowledge_base_data
```

## 8. Вариант A: перенести готовый `faiss_index` с локальной машины

На локальной машине:

```bash
rsync -az --progress \
  /Users/daniil/Desktop/project_diploma/faiss_index/ \
  user1@82.202.136.67:/home/user1/<repo>/faiss_index/
```

Если индекс уже готов, это самый быстрый путь.

Если вы меняли `OPENAI_EMBEDDING_MODEL`, старый `faiss_index/` переносить не нужно: его надо пересобрать заново, потому что векторы разных embedding-моделей несовместимы.

## 9. Вариант B: собрать индекс уже на VM

Если папку `faiss_index/` не переносите, после запуска контейнеров заполните базу знаний через UI на вкладке «База знаний».

## 10. Собрать и запустить сервисы

```bash
docker compose up -d --build
```

Проверить статус:

```bash
docker compose ps
docker compose logs -f backend
docker compose logs -f ui
```

## 11. Открыть сервисы

Если security group разрешает входящий трафик:

- UI: `http://82.202.136.67:8501`
- Backend health: `http://82.202.136.67:8000/health`

Для постоянного доступа лучше завернуть UI в nginx.

## 12. Настроить nginx как reverse proxy

```bash
sudo apt install -y nginx
sudo nano /etc/nginx/sites-available/tz-analyzer
```

Вставьте:

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

Активировать:

```bash
sudo ln -s /etc/nginx/sites-available/tz-analyzer /etc/nginx/sites-enabled/tz-analyzer
sudo nginx -t
sudo systemctl reload nginx
```

После этого UI будет доступен по адресу:

```text
http://82.202.136.67
```

## 13. Открыть порты в Cloud.ru

Обычно достаточно:

- `22/tcp` — SSH
- `80/tcp` — nginx

Временный вариант для диагностики:

- `8501/tcp` — Streamlit UI
- `8000/tcp` — backend API

Для production лучше оставить наружу только `80`, а потом добавить `443`.

## 14. Обновление после нового push

На VM:

```bash
cd /home/user1/<repo>
git pull
docker compose up -d --build
```

## 15. Если база знаний пустая

Есть два пути:

1. Перенести готовую `faiss_index/` с локальной машины.
2. Зайти в UI и запустить краулинг на вкладке «База знаний».

Для этой VM быстрее и надёжнее обычно перенос готового индекса.

## 16. Что проверить после деплоя

1. Открывается UI.
2. `http://127.0.0.1:8000/health` отвечает на VM.
3. В UI видно количество векторов.
4. Загружается тестовый `.txt` или `.docx`.
5. Проходит извлечение требований.
6. Скачиваются отчёты в MD/DOCX/PDF/XLSX.
