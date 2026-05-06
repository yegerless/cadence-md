# CADENCE-MD Frontend

React SPA для медицинского RAG-сервиса CADENCE-MD.

## Команды

```bash
npm install
npm run dev
npm run build
npm test
```

Приложение использует `VITE_API_BASE_URL` для API-запросов. Для локальной
разработки рекомендуется оставить значение пустым, чтобы запросы к `/api/v1/...`
шли через Vite proxy и не требовали CORS в браузере. Укажите
`VITE_API_PROXY_TARGET=http://127.0.0.1:8000` для запуска с backend на хосте или
`http://backend:8000` в Docker Compose.

## Политика хранения токенов

В MVP хранится только короткоживущий JWT access token в `sessionStorage`.
Backend не предоставляет refresh-token endpoint, поэтому при истечении токена
или любом ответе `401` frontend очищает auth state и переводит пользователя на
`/login` для повторной авторизации.

Компромисс: `sessionStorage` сохраняет сессию при refresh страницы и не
отправляется автоматически браузером как cookie, что снижает CSRF-риск. При этом
токен доступен JavaScript, поэтому основной риск — XSS. Frontend не логирует
значения токенов и не рендерит недоверенный HTML.

## Smoke-проверки

- Выполните login или register, обновите страницу и убедитесь, что `/chat`
  остаётся доступным в рамках текущей браузерной сессии.
- Нажмите logout и убедитесь, что в `sessionStorage` больше нет access token, а
  `/chat` редиректит на `/login`.
- Смоделируйте `401` от API и проверьте, что auth state очищается, а
  пользователь переводится на `/login`.
- Отправьте вопрос в чат, проверьте статусы queued/running и терминальные
  состояния: answer, failed, cancelled и retry.

# React + TypeScript + Vite

Этот шаблон предоставляет минимальную настройку для React в Vite с HMR и
базовыми правилами ESLint.

Сейчас доступны два официальных плагина:

- [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react),
  использует [Oxc](https://oxc.rs)
- [@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react-swc),
  использует [SWC](https://swc.rs/)

## React Compiler

React Compiler в этом шаблоне не включён из-за влияния на производительность
dev/build. Для подключения см.
[документацию](https://react.dev/learn/react-compiler/installation).

## Расширение конфигурации ESLint

Если вы разрабатываете production-приложение, рекомендуется обновить
конфигурацию и включить type-aware правила:

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

Также можно установить
[eslint-plugin-react-x](https://github.com/Rel1cx/eslint-react/tree/main/packages/plugins/eslint-plugin-react-x)
и
[eslint-plugin-react-dom](https://github.com/Rel1cx/eslint-react/tree/main/packages/plugins/eslint-plugin-react-dom)
для дополнительных React-правил:

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
