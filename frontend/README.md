# CADENCE-MD Frontend

React SPA for the CADENCE-MD medical RAG service.

## Commands

```bash
npm install
npm run dev
npm run build
npm test
```

Smoke test scope in `npm test` covers route navigation (`/login`, `/register`,
`/chat`, `/profile`) and mocked chat lifecycle statuses (`queued`, `running`,
`succeeded`, `failed`, `cancelled`).

The app uses `VITE_API_BASE_URL` for API requests. In local development the
recommended value is empty, so requests to `/api/v1/...` go through the Vite
proxy and avoid browser CORS requirements. Set
`VITE_API_PROXY_TARGET=http://127.0.0.1:8000` for local host backend runs or
`http://backend:8000` in Docker Compose.

## Auth Storage Policy

The MVP stores only the short-lived JWT access token in `sessionStorage`. The
backend does not expose a refresh-token endpoint, so token expiry or any `401`
response clears client auth state and sends the user to `/login` for
re-authentication.

Trade-off: `sessionStorage` preserves the session through page refreshes and
avoids automatic cookie submission, reducing CSRF exposure. It is still readable
by JavaScript, so XSS is the main risk. The frontend does not log token values
and does not render untrusted HTML.

## Smoke Checks

- Login or register, refresh the page, and confirm `/chat` remains available
  during the same browser session.
- Click logout and confirm `sessionStorage` no longer contains the access token
  and `/chat` redirects to `/login`.
- Force a `401` from the API and confirm auth state is cleared and the user is
  redirected to `/login`.
- Submit a chat question, observe queued/running status, then check terminal
  answer, failure, cancel, and retry states.

# React + TypeScript + Vite

This template provides a minimal setup to get React working in Vite with HMR and
some ESLint rules.

Currently, two official plugins are available:

- [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react)
  uses [Oxc](https://oxc.rs)
- [@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react-swc)
  uses [SWC](https://swc.rs/)

## React Compiler

The React Compiler is not enabled on this template because of its impact on dev
& build performances. To add it, see
[this documentation](https://react.dev/learn/react-compiler/installation).

## Expanding the ESLint configuration

If you are developing a production application, we recommend updating the
configuration to enable type-aware lint rules:

```js
export default defineConfig([
  globalIgnores(["dist"]),
  {
    files: ["**/*.{ts,tsx}"],
    extends: [
      // Other configs...

      // Remove tseslint.configs.recommended and replace with this
      tseslint.configs.recommendedTypeChecked,
      // Alternatively, use this for stricter rules
      tseslint.configs.strictTypeChecked,
      // Optionally, add this for stylistic rules
      tseslint.configs.stylisticTypeChecked,

      // Other configs...
    ],
    languageOptions: {
      parserOptions: {
        project: ["./tsconfig.node.json", "./tsconfig.app.json"],
        tsconfigRootDir: import.meta.dirname,
      },
      // other options...
    },
  },
]);
```

You can also install
[eslint-plugin-react-x](https://github.com/Rel1cx/eslint-react/tree/main/packages/plugins/eslint-plugin-react-x)
and
[eslint-plugin-react-dom](https://github.com/Rel1cx/eslint-react/tree/main/packages/plugins/eslint-plugin-react-dom)
for React-specific lint rules:

```js
// eslint.config.js
import reactX from "eslint-plugin-react-x";
import reactDom from "eslint-plugin-react-dom";

export default defineConfig([
  globalIgnores(["dist"]),
  {
    files: ["**/*.{ts,tsx}"],
    extends: [
      // Other configs...
      // Enable lint rules for React
      reactX.configs["recommended-typescript"],
      // Enable lint rules for React DOM
      reactDom.configs.recommended,
    ],
    languageOptions: {
      parserOptions: {
        project: ["./tsconfig.node.json", "./tsconfig.app.json"],
        tsconfigRootDir: import.meta.dirname,
      },
      // other options...
    },
  },
]);
```
