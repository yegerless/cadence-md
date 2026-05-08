import { QueryClient } from '@tanstack/react-query'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, test, vi } from 'vitest'
import App from './App'
import { setAccessToken } from './auth/tokenStorage'
import type {
  ChatConversationResponse,
  RAGAnswerResponse,
  RAGRequestStatusResponse,
} from './api/types'

const profile = {
  id: 'user-1',
  email: 'doctor@example.org',
  first_name: 'Anna',
  last_name: 'Petrova',
}

function renderApp(path: string) {
  window.history.pushState({}, 'Test page', path)
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  })
  return render(<App queryClient={queryClient} />)
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

function emptyResponse(status = 204): Response {
  return new Response(null, { status })
}

function notFoundResponse(): Response {
  return jsonResponse({ code: 'not_found', message: 'Not found', details: {}, request_id: null }, 404)
}

function conversation(
  overrides: Partial<ChatConversationResponse> = {},
): ChatConversationResponse {
  return {
    id: 'chat-1',
    title: 'Клинический чат',
    created_at: '2026-05-08T12:00:00Z',
    updated_at: '2026-05-08T12:00:00Z',
    last_message_at: null,
    ...overrides,
  }
}

function statusResponse(
  overrides: Partial<RAGRequestStatusResponse> = {},
): RAGRequestStatusResponse {
  return {
    request_id: 'req-1',
    status: 'queued',
    chat_id: 'chat-1',
    original_request_id: null,
    answer: null,
    error: null,
    clarification: null,
    ...overrides,
  }
}

function answerResponse(overrides: Partial<RAGAnswerResponse> = {}): RAGAnswerResponse {
  return {
    answer: 'Ответ с цитатой [Doc 1].',
    sources: [{
      rank: 1,
      doc_ref: '[Doc 1]',
      filename: 'guideline.pdf',
      source_path: 'main_specialities/guideline.pdf',
      document_title: 'Клинические рекомендации',
      section_title: 'Терапия',
      section_id: 'section-1',
      chunk_id: 'chunk-1',
      content: 'Фрагмент рекомендации',
      score: 0.91,
    }],
    query_hash: 'hash',
    latency_ms: {},
    flags: {},
    error_type: null,
    error_message: null,
    langfuse_trace_id: null,
    ...overrides,
  }
}

function conversationList(items: ChatConversationResponse[]) {
  return { items, limit: 50, offset: 0 }
}

function historyItem(query: string, request: RAGRequestStatusResponse) {
  return { query, request }
}

function historyResponse(items: ReturnType<typeof historyItem>[] = []) {
  return { items }
}

function isConversationListRequest(url: string, init?: RequestInit): boolean {
  return (init?.method ?? 'GET') === 'GET'
    && /\/api\/v1\/chat\/conversations(?:\?|$)/.test(url)
}

function jsonBody(init?: RequestInit): Record<string, unknown> {
  return JSON.parse(String(init?.body ?? '{}')) as Record<string, unknown>
}

function commonResponse(
  input: RequestInfo | URL,
  init: RequestInit | undefined,
  conversations: ChatConversationResponse[] = [],
): Response | null {
  const url = String(input)
  if (url.endsWith('/api/v1/users/me')) {
    return jsonResponse(profile)
  }
  if (isConversationListRequest(url, init)) {
    return jsonResponse(conversationList(conversations))
  }
  return null
}

describe('CADENCE-MD frontend routes', () => {
  test('renders login and registration pages', async () => {
    const { unmount } = renderApp('/login')
    expect(await screen.findByRole('heading', { name: /добро пожаловать/i })).toBeInTheDocument()
    unmount()

    renderApp('/register')
    expect(await screen.findByRole('heading', { name: /создать аккаунт/i })).toBeInTheDocument()
  })

  test('redirects protected chat route to login without token', async () => {
    renderApp('/chat')
    expect(await screen.findByRole('heading', { name: /добро пожаловать/i })).toBeInTheDocument()
  })

  test('login stores token and opens chat', async () => {
    const fetchMock = vi.spyOn(window, 'fetch').mockImplementation(async (input, init) => {
      const url = String(input)
      if (url.endsWith('/api/v1/auth/login')) {
        return jsonResponse({ access_token: 'access-token', token_type: 'bearer', expires_in: 3600 })
      }
      const response = commonResponse(input, init)
      if (response) {
        return response
      }
      return notFoundResponse()
    })

    renderApp('/login')
    await userEvent.type(await screen.findByLabelText(/email/i), 'doctor@example.org')
    await userEvent.type(screen.getByLabelText(/пароль/i), 'correct-password')
    await userEvent.click(screen.getByRole('button', { name: /войти/i }))

    expect(await screen.findByRole('heading', { name: /задайте клинический вопрос/i }))
      .toBeInTheDocument()
    expect(window.sessionStorage.getItem('cadence_md_access_token')).toBe('access-token')
    expect(fetchMock).toHaveBeenCalled()
  })

  test('logout clears session storage and redirects to login', async () => {
    setAccessToken('access-token')
    vi.spyOn(window, 'fetch').mockResolvedValue(jsonResponse(profile))

    renderApp('/profile')
    expect(await screen.findByRole('heading', { name: /anna petrova/i })).toBeInTheDocument()

    await userEvent.click(screen.getByRole('button', { name: /выйти из аккаунта/i }))

    await waitFor(() => {
      expect(window.sessionStorage.getItem('cadence_md_access_token')).toBeNull()
    })
    expect(await screen.findByRole('heading', { name: /добро пожаловать/i })).toBeInTheDocument()
  })
})

describe('chat history UI', () => {
  test('loads the chat list after authentication', async () => {
    setAccessToken('access-token')
    const oldChat = conversation({ id: 'chat-old', title: 'АГ: старый чат' })
    vi.spyOn(window, 'fetch').mockImplementation(async (input, init) => (
      commonResponse(input, init, [oldChat]) ?? notFoundResponse()
    ))

    renderApp('/chat')

    expect(await screen.findByRole('button', { name: /^АГ: старый чат/i })).toBeInTheDocument()
  })

  test('creates a new chat and sends the message with chat_id', async () => {
    setAccessToken('access-token')
    const newChat = conversation({ id: 'chat-new', title: null })
    let created = false
    let messagePosted = false
    const fetchMock = vi.spyOn(window, 'fetch').mockImplementation(async (input, init) => {
      const url = String(input)
      if (url.endsWith('/api/v1/chat/conversations') && init?.method === 'POST') {
        created = true
        return jsonResponse(newChat, 201)
      }
      const response = commonResponse(input, init, created ? [newChat] : [])
      if (response) {
        return response
      }
      if (url.endsWith('/api/v1/chat/conversations/chat-new/messages')) {
        return jsonResponse(historyResponse(
          messagePosted
            ? [historyItem(
              'Как лечить АГ?',
              statusResponse({ request_id: 'req-new', chat_id: 'chat-new' }),
            )]
            : [],
        ))
      }
      if (url.endsWith('/api/v1/chat/messages') && init?.method === 'POST') {
        expect(jsonBody(init).chat_id).toBe('chat-new')
        messagePosted = true
        return jsonResponse(statusResponse({ request_id: 'req-new', chat_id: 'chat-new' }), 202)
      }
      if (url.endsWith('/api/v1/chat/messages/req-new')) {
        return jsonResponse(statusResponse({ request_id: 'req-new', chat_id: 'chat-new' }))
      }
      return notFoundResponse()
    })

    renderApp('/chat')
    await userEvent.click(await screen.findByRole('button', { name: /^новый чат$/i }))
    await userEvent.type(await screen.findByLabelText(/клинический вопрос/i), 'Как лечить АГ?')
    await userEvent.click(screen.getByRole('button', { name: /^отправить$/i }))

    await waitFor(() => {
      expect(fetchMock.mock.calls.some(
        ([url, options]) =>
          String(url).endsWith('/api/v1/chat/messages')
          && (options as RequestInit).method === 'POST',
      )).toBe(true)
    })
  })

  test('opens an old chat and restores persisted turns', async () => {
    setAccessToken('access-token')
    const oldChat = conversation({ id: 'chat-old', title: 'Старый чат' })
    vi.spyOn(window, 'fetch').mockImplementation(async (input, init) => {
      const url = String(input)
      const response = commonResponse(input, init, [oldChat])
      if (response) {
        return response
      }
      if (url.endsWith('/api/v1/chat/conversations/chat-old/messages')) {
        return jsonResponse(historyResponse([
          historyItem(
            'Исторический вопрос',
            statusResponse({
              request_id: 'req-old',
              chat_id: 'chat-old',
              status: 'succeeded',
              answer: answerResponse({ answer: 'Исторический ответ [Doc 1].' }),
            }),
          ),
        ]))
      }
      return notFoundResponse()
    })

    renderApp('/chat')
    await userEvent.click(await screen.findByRole('button', { name: /^старый чат/i }))

    expect(await screen.findByText(/исторический вопрос/i)).toBeInTheDocument()
    expect(screen.getByText(/исторический ответ/i)).toBeInTheDocument()
  })

  test('deletes a chat from the UI', async () => {
    setAccessToken('access-token')
    const removableChat = conversation({ id: 'chat-remove', title: 'Чат для удаления' })
    let conversations = [removableChat]
    vi.spyOn(window, 'fetch').mockImplementation(async (input, init) => {
      const url = String(input)
      const response = commonResponse(input, init, conversations)
      if (response) {
        return response
      }
      if (url.endsWith('/api/v1/chat/conversations/chat-remove/messages')) {
        return jsonResponse(historyResponse())
      }
      if (url.endsWith('/api/v1/chat/conversations/chat-remove') && init?.method === 'DELETE') {
        conversations = []
        return emptyResponse()
      }
      return notFoundResponse()
    })

    renderApp('/chat')
    await userEvent.click(await screen.findByRole('button', { name: /^чат для удаления/i }))
    await userEvent.click(screen.getByRole('button', { name: /удалить чат чат для удаления/i }))

    await waitFor(() => {
      expect(screen.queryByRole('button', { name: /чат для удаления/i })).not.toBeInTheDocument()
    })
    expect(await screen.findByText(/начните новый чат/i)).toBeInTheDocument()
  })
})

describe('chat lifecycle', () => {
  test('submits a question, polls result, and renders sources', async () => {
    setAccessToken('access-token')
    const createdChat = conversation({ id: 'chat-1', title: 'Какая терапия рекомендована?' })
    let currentStatus = statusResponse({ request_id: 'req-1' })
    vi.spyOn(window, 'fetch').mockImplementation(async (input, init) => {
      const url = String(input)
      if (url.endsWith('/api/v1/chat/conversations') && init?.method === 'POST') {
        return jsonResponse(createdChat, 201)
      }
      const response = commonResponse(input, init, [createdChat])
      if (response) {
        return response
      }
      if (url.endsWith('/api/v1/chat/conversations/chat-1/messages')) {
        return jsonResponse(historyResponse([
          historyItem('Какая терапия рекомендована?', currentStatus),
        ]))
      }
      if (url.endsWith('/api/v1/chat/messages') && init?.method === 'POST') {
        currentStatus = statusResponse({ request_id: 'req-1', status: 'queued' })
        return jsonResponse(currentStatus, 202)
      }
      if (url.endsWith('/api/v1/chat/messages/req-1')) {
        currentStatus = statusResponse({
          request_id: 'req-1',
          status: 'succeeded',
          answer: answerResponse(),
        })
        return jsonResponse(currentStatus)
      }
      return notFoundResponse()
    })

    renderApp('/chat')
    await userEvent.type(
      await screen.findByLabelText(/клинический вопрос/i),
      'Какая терапия рекомендована?',
    )
    await userEvent.click(screen.getByRole('button', { name: /^отправить$/i }))

    expect(await screen.findByText(/ответ с цитатой/i)).toBeInTheDocument()
    expect(screen.getByText('Клинические рекомендации')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /скачать pdf/i })).toBeInTheDocument()
  })

  test('clears auth state on API 401', async () => {
    setAccessToken('access-token')
    vi.spyOn(window, 'fetch').mockResolvedValue(
      jsonResponse({ code: 'invalid_token', message: 'Invalid token', details: {}, request_id: null }, 401),
    )

    renderApp('/chat')

    expect(await screen.findByRole('heading', { name: /добро пожаловать/i })).toBeInTheDocument()
    expect(window.sessionStorage.getItem('cadence_md_access_token')).toBeNull()
  })

  test('renders running status during polling', async () => {
    setAccessToken('access-token')
    const createdChat = conversation({ id: 'chat-running', title: 'Running' })
    let currentStatus = statusResponse({ request_id: 'req-running', chat_id: 'chat-running' })
    vi.spyOn(window, 'fetch').mockImplementation(async (input, init) => {
      const url = String(input)
      if (url.endsWith('/api/v1/chat/conversations') && init?.method === 'POST') {
        return jsonResponse(createdChat, 201)
      }
      const response = commonResponse(input, init, [createdChat])
      if (response) {
        return response
      }
      if (url.endsWith('/api/v1/chat/conversations/chat-running/messages')) {
        return jsonResponse(historyResponse([
          historyItem('Проверка running статуса', currentStatus),
        ]))
      }
      if (url.endsWith('/api/v1/chat/messages') && init?.method === 'POST') {
        currentStatus = statusResponse({ request_id: 'req-running', chat_id: 'chat-running' })
        return jsonResponse(
          currentStatus,
          202,
        )
      }
      if (url.endsWith('/api/v1/chat/messages/req-running')) {
        currentStatus = statusResponse({
          request_id: 'req-running',
          chat_id: 'chat-running',
          status: 'running',
        })
        return jsonResponse(currentStatus)
      }
      return notFoundResponse()
    })

    renderApp('/chat')
    await userEvent.type(await screen.findByLabelText(/клинический вопрос/i), 'Проверка running статуса')
    await userEvent.click(screen.getByRole('button', { name: /^отправить$/i }))

    expect(await screen.findByText(/выполняется/i)).toBeInTheDocument()
    expect(screen.getByText(/генерируем ответ/i)).toBeInTheDocument()
  })

  test('renders failed status with retry action', async () => {
    setAccessToken('access-token')
    const createdChat = conversation({ id: 'chat-failed', title: 'Failed' })
    let currentStatus = statusResponse({ request_id: 'req-failed', chat_id: 'chat-failed' })
    vi.spyOn(window, 'fetch').mockImplementation(async (input, init) => {
      const url = String(input)
      if (url.endsWith('/api/v1/chat/conversations') && init?.method === 'POST') {
        return jsonResponse(createdChat, 201)
      }
      const response = commonResponse(input, init, [createdChat])
      if (response) {
        return response
      }
      if (url.endsWith('/api/v1/chat/conversations/chat-failed/messages')) {
        return jsonResponse(historyResponse([
          historyItem('Проверка failed статуса', currentStatus),
        ]))
      }
      if (url.endsWith('/api/v1/chat/messages') && init?.method === 'POST') {
        currentStatus = statusResponse({ request_id: 'req-failed', chat_id: 'chat-failed' })
        return jsonResponse(
          currentStatus,
          202,
        )
      }
      if (url.endsWith('/api/v1/chat/messages/req-failed')) {
        currentStatus = statusResponse({
          request_id: 'req-failed',
          chat_id: 'chat-failed',
          status: 'failed',
          error: 'Ошибка worker',
        })
        return jsonResponse(currentStatus)
      }
      return notFoundResponse()
    })

    renderApp('/chat')
    await userEvent.type(await screen.findByLabelText(/клинический вопрос/i), 'Проверка failed статуса')
    await userEvent.click(screen.getByRole('button', { name: /^отправить$/i }))

    expect(await screen.findByText(/ошибка worker/i)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /повторить/i })).toBeInTheDocument()
  })

  test('renders cancelled status with retry action', async () => {
    setAccessToken('access-token')
    const createdChat = conversation({ id: 'chat-cancelled', title: 'Cancelled' })
    let currentStatus = statusResponse({ request_id: 'req-cancelled', chat_id: 'chat-cancelled' })
    vi.spyOn(window, 'fetch').mockImplementation(async (input, init) => {
      const url = String(input)
      if (url.endsWith('/api/v1/chat/conversations') && init?.method === 'POST') {
        return jsonResponse(createdChat, 201)
      }
      const response = commonResponse(input, init, [createdChat])
      if (response) {
        return response
      }
      if (url.endsWith('/api/v1/chat/conversations/chat-cancelled/messages')) {
        return jsonResponse(historyResponse([
          historyItem('Проверка cancelled статуса', currentStatus),
        ]))
      }
      if (url.endsWith('/api/v1/chat/messages') && init?.method === 'POST') {
        currentStatus = statusResponse({ request_id: 'req-cancelled', chat_id: 'chat-cancelled' })
        return jsonResponse(
          currentStatus,
          202,
        )
      }
      if (url.endsWith('/api/v1/chat/messages/req-cancelled')) {
        currentStatus = statusResponse({
          request_id: 'req-cancelled',
          chat_id: 'chat-cancelled',
          status: 'cancelled',
        })
        return jsonResponse(currentStatus)
      }
      return notFoundResponse()
    })

    renderApp('/chat')
    await userEvent.type(await screen.findByLabelText(/клинический вопрос/i), 'Проверка cancelled статуса')
    await userEvent.click(screen.getByRole('button', { name: /^отправить$/i }))

    expect(await screen.findByText(/запрос отменен/i)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /повторить/i })).toBeInTheDocument()
  })

  test('renders awaiting clarification status and question', async () => {
    setAccessToken('access-token')
    const createdChat = conversation({ id: 'chat-await', title: 'Await' })
    let currentStatus = statusResponse({ request_id: 'req-await', chat_id: 'chat-await' })
    vi.spyOn(window, 'fetch').mockImplementation(async (input, init) => {
      const url = String(input)
      if (url.endsWith('/api/v1/chat/conversations') && init?.method === 'POST') {
        return jsonResponse(createdChat, 201)
      }
      const response = commonResponse(input, init, [createdChat])
      if (response) {
        return response
      }
      if (url.endsWith('/api/v1/chat/conversations/chat-await/messages')) {
        return jsonResponse(historyResponse([
          historyItem('Как лечить АГ?', currentStatus),
        ]))
      }
      if (url.endsWith('/api/v1/chat/messages') && init?.method === 'POST') {
        currentStatus = statusResponse({ request_id: 'req-await', chat_id: 'chat-await' })
        return jsonResponse(currentStatus, 202)
      }
      if (url.endsWith('/api/v1/chat/messages/req-await')) {
        currentStatus = statusResponse({
          request_id: 'req-await',
          chat_id: 'chat-await',
          status: 'awaiting_clarification',
          clarification: {
            question: 'Уточните возраст и сопутствующие заболевания?',
            answered: false,
            requested_at: null,
          },
        })
        return jsonResponse(currentStatus)
      }
      return notFoundResponse()
    })

    renderApp('/chat')
    await userEvent.type(await screen.findByLabelText(/клинический вопрос/i), 'Как лечить АГ?')
    await userEvent.click(screen.getByRole('button', { name: /^отправить$/i }))

    expect(await screen.findByText(/требуется уточнение/i)).toBeInTheDocument()
    expect(screen.getByText(/уточните возраст и сопутствующие/i)).toBeInTheDocument()
    expect(screen.getByLabelText(/уточнение врача/i)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /отправить уточнение/i })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /отменить/i })).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /повторить/i })).not.toBeInTheDocument()
  })

  test('submit clarification calls endpoint and resumes queued/running polling', async () => {
    setAccessToken('access-token')
    const createdChat = conversation({ id: 'chat-submit-clarify', title: 'Clarify' })
    let clarificationPosted = false
    let currentStatus = statusResponse({
      request_id: 'req-submit-clarify',
      chat_id: 'chat-submit-clarify',
    })
    const fetchMock = vi.spyOn(window, 'fetch').mockImplementation(async (input, init) => {
      const url = String(input)
      if (url.endsWith('/api/v1/chat/conversations') && init?.method === 'POST') {
        return jsonResponse(createdChat, 201)
      }
      const response = commonResponse(input, init, [createdChat])
      if (response) {
        return response
      }
      if (url.endsWith('/api/v1/chat/conversations/chat-submit-clarify/messages')) {
        return jsonResponse(historyResponse([
          historyItem('Терапия АГ', currentStatus),
        ]))
      }
      if (url.endsWith('/api/v1/chat/messages') && init?.method === 'POST') {
        currentStatus = statusResponse({
          request_id: 'req-submit-clarify',
          chat_id: 'chat-submit-clarify',
        })
        return jsonResponse(currentStatus, 202)
      }
      if (url.endsWith('/api/v1/chat/messages/req-submit-clarify/clarification') && init?.method === 'POST') {
        clarificationPosted = true
        expect(jsonBody(init)).toEqual({ answer: 'Взрослый пациент 58 лет.' })
        currentStatus = statusResponse({
          request_id: 'req-submit-clarify',
          chat_id: 'chat-submit-clarify',
          status: 'queued',
        })
        return jsonResponse(currentStatus)
      }
      if (url.endsWith('/api/v1/chat/messages/req-submit-clarify')) {
        if (!clarificationPosted) {
          currentStatus = statusResponse({
            request_id: 'req-submit-clarify',
            chat_id: 'chat-submit-clarify',
            status: 'awaiting_clarification',
            clarification: {
              question: 'Уточните контекст пациента?',
              answered: false,
              requested_at: null,
            },
          })
          return jsonResponse(currentStatus)
        }
        currentStatus = statusResponse({
          request_id: 'req-submit-clarify',
          chat_id: 'chat-submit-clarify',
          status: 'running',
        })
        return jsonResponse(currentStatus)
      }
      return notFoundResponse()
    })

    renderApp('/chat')
    await userEvent.type(await screen.findByLabelText(/клинический вопрос/i), 'Терапия АГ')
    await userEvent.click(screen.getByRole('button', { name: /^отправить$/i }))

    expect(await screen.findByText(/уточните контекст пациента/i)).toBeInTheDocument()

    const user = userEvent.setup()
    await user.type(
      await screen.findByPlaceholderText(/Уточните контекст или выберите допущения/i),
      'Взрослый пациент 58 лет.',
    )
    await user.click(screen.getByRole('button', { name: /отправить уточнение/i }))

    await waitFor(() => {
      expect(
        fetchMock.mock.calls.some(
          ([url, options]) =>
            String(url).endsWith('/api/v1/chat/messages/req-submit-clarify/clarification')
            && (options as RequestInit).method === 'POST',
        ),
      ).toBe(true)
    })

    expect(await screen.findByText(/выполняется/i)).toBeInTheDocument()
    expect(screen.getByText(/генерируем ответ/i)).toBeInTheDocument()
  })

  test('cancel remains available for awaiting clarification', async () => {
    setAccessToken('access-token')
    const createdChat = conversation({ id: 'chat-await-cancel', title: 'Cancel' })
    let currentStatus = statusResponse({
      request_id: 'req-await-cancel',
      chat_id: 'chat-await-cancel',
    })
    vi.spyOn(window, 'fetch').mockImplementation(async (input, init) => {
      const url = String(input)
      if (url.endsWith('/api/v1/chat/conversations') && init?.method === 'POST') {
        return jsonResponse(createdChat, 201)
      }
      const response = commonResponse(input, init, [createdChat])
      if (response) {
        return response
      }
      if (url.endsWith('/api/v1/chat/conversations/chat-await-cancel/messages')) {
        return jsonResponse(historyResponse([
          historyItem('Вопрос для отмены', currentStatus),
        ]))
      }
      if (url.endsWith('/api/v1/chat/messages') && init?.method === 'POST') {
        currentStatus = statusResponse({
          request_id: 'req-await-cancel',
          chat_id: 'chat-await-cancel',
        })
        return jsonResponse(currentStatus, 202)
      }
      if (url.endsWith('/api/v1/chat/messages/req-await-cancel/cancel') && init?.method === 'POST') {
        currentStatus = statusResponse({
          request_id: 'req-await-cancel',
          chat_id: 'chat-await-cancel',
          status: 'cancelled',
        })
        return jsonResponse(currentStatus)
      }
      if (url.endsWith('/api/v1/chat/messages/req-await-cancel')) {
        currentStatus = statusResponse({
          request_id: 'req-await-cancel',
          chat_id: 'chat-await-cancel',
          status: 'awaiting_clarification',
          clarification: {
            question: 'Нужны ли детали коморбидности?',
            answered: false,
            requested_at: null,
          },
        })
        return jsonResponse(currentStatus)
      }
      return notFoundResponse()
    })

    renderApp('/chat')
    await userEvent.type(await screen.findByLabelText(/клинический вопрос/i), 'Вопрос для отмены')
    await userEvent.click(screen.getByRole('button', { name: /^отправить$/i }))

    expect(await screen.findByText(/требуется уточнение/i)).toBeInTheDocument()

    await userEvent.click(screen.getByRole('button', { name: /отменить/i }))

    expect(await screen.findByText(/запрос отменен/i)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /повторить/i })).toBeInTheDocument()
  })
})
