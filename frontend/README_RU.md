# CADENCE-MD Frontend

React + Vite SPA для медицинского RAG-сервиса CADENCE-MD.

Frontend работает с versioned backend API под `/api/v1` и покрывает MVP
workflow:

- `/login` и `/register` для JWT-аутентификации.
- `/chat` для асинхронных RAG-запросов, polling, cancel, retry, отправки
  уточнения, отображения ответа, источников и скачивания PDF-источников.
- `/profile` для профиля авторизованного пользователя.

## Стек

- React 19 + TypeScript
- Vite dev server и same-origin API proxy
- React Router
- TanStack Query
- Vitest, Testing Library, Happy DOM, MSW

## Команды

Команды из `frontend/`:

```bash
npm install
npm run dev
npm test
npm run lint
npm run build
npm run preview
```

Команды из корня репозитория:

```bash
npm --prefix frontend test
npm --prefix frontend run lint
npm --prefix frontend run build
```

## Настройка API

Приложение использует `VITE_API_BASE_URL` для API-запросов.

Для локальной разработки через Vite proxy оставьте `VITE_API_BASE_URL` пустым и
укажите backend в `VITE_API_PROXY_TARGET`:

```bash
VITE_API_BASE_URL= VITE_API_PROXY_TARGET=http://127.0.0.1:8000 npm run dev
```

В Docker Compose `VITE_API_PROXY_TARGET` по умолчанию равен
`http://backend:8000`. Так браузер работает с same-origin `/api/v1/...`, а
backend не требует CORS middleware для локальной разработки.

## Docker Compose

Dev compose stack содержит сервис `frontend`:

```bash
docker compose --env-file ../.env.dev -f ../docker-compose-dev.yml up frontend
```

Из корня репозитория:

```bash
docker compose --env-file .env.dev -f docker-compose-dev.yml up frontend
docker compose --env-file .env.dev -f docker-compose-dev.yml watch frontend
```

Compose watch синхронизирует `./frontend` в `/app` и rebuild-ит контейнер при
изменениях `frontend/package*.json`.

## Политика хранения токенов

MVP хранит только короткоживущий JWT access token в `sessionStorage`. Backend не
предоставляет refresh-token endpoint, поэтому истечение токена или любой API
`401` очищает client auth state и переводит пользователя на `/login`.

`sessionStorage` сохраняет сессию при refresh страницы и не отправляется
автоматически как cookie. При этом токен доступен JavaScript, поэтому основной
риск — XSS. Frontend не логирует token values и не рендерит недоверенный HTML.

## Тесты и smoke-проверки

`npm test` запускает Vitest smoke coverage для:

- навигации по `/login`, `/register`, `/chat`, `/profile`;
- сохранения auth state, logout и автоматической очистки после API `401`;
- mocked chat lifecycle статусов: `queued`, `running`, `awaiting_clarification`,
  `succeeded`, `failed`, `cancelled`;
- отправки/отмены уточнения, retry, отображения ответа и источников.

Полный локальный dev flow описан в [`../docs/setup_ru.md`](../docs/setup_ru.md).
