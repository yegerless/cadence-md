# CADENCE-MD Frontend

React + Vite SPA for the CADENCE-MD medical RAG service.

The frontend talks to the versioned backend API under `/api/v1` and covers the
MVP clinical workflow:

- `/login` and `/register` for JWT authentication.
- `/chat` for persisted chat history, conversation list/open/create/delete,
  asynchronous RAG requests, polling, cancellation, retry, clarification
  submission, answer rendering, source rendering, and PDF source download.
- `/profile` for the authenticated user profile.

## Stack

- React 19 + TypeScript
- Vite dev server and same-origin API proxy
- React Router
- TanStack Query
- Vitest, Testing Library, Happy DOM, MSW

## Commands

Run commands from `frontend/`:

```bash
npm install
npm run dev
npm test
npm run lint
npm run build
npm run preview
```

Or from the repository root:

```bash
npm --prefix frontend test
npm --prefix frontend run lint
npm --prefix frontend run build
```

## API Configuration

The app uses `VITE_API_BASE_URL` for API requests.

For local development with the Vite proxy, keep `VITE_API_BASE_URL` empty and
point `VITE_API_PROXY_TARGET` at the backend:

```bash
VITE_API_BASE_URL= VITE_API_PROXY_TARGET=http://127.0.0.1:8000 npm run dev
```

In Docker Compose, `VITE_API_PROXY_TARGET` defaults to `http://backend:8000`.
This keeps browser traffic same-origin (`/api/v1/...`) and avoids requiring CORS
middleware in the backend.

## Docker Compose

The dev compose stack includes a `frontend` service:

```bash
docker compose --env-file ../.env.dev -f ../docker-compose-dev.yml up frontend
```

From the repository root:

```bash
docker compose --env-file .env.dev -f docker-compose-dev.yml up frontend
docker compose --env-file .env.dev -f docker-compose-dev.yml watch frontend
```

Compose watch syncs `./frontend` to `/app` and rebuilds when
`frontend/package*.json` changes.

## Auth Storage Policy

The MVP stores only the short-lived JWT access token in `sessionStorage`. The
backend does not expose a refresh-token endpoint, so token expiry or any API
`401` response clears client auth state and redirects the user to `/login`.

`sessionStorage` keeps the session through page refreshes and avoids automatic
cookie submission. It is still readable by JavaScript, so XSS remains the main
risk. The frontend does not log token values and does not render untrusted HTML.

## Tests And Smoke Checks

`npm test` runs Vitest smoke coverage for:

- route navigation across `/login`, `/register`, `/chat`, and `/profile`;
- auth persistence, logout, and automatic cleanup after API `401`;
- chat history loading, opening existing conversations, creating new chats, and
  soft deletion from the UI;
- mocked chat lifecycle statuses: `queued`, `running`, `awaiting_clarification`,
  `succeeded`, `failed`, and `cancelled`;
- clarification submit/cancel, retry, answer rendering, and source rendering.

For the full local development flow, see [`../docs/setup.md`](../docs/setup.md).
