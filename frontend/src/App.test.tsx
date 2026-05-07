import { QueryClient } from '@tanstack/react-query'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, test, vi } from 'vitest'
import App from './App'
import { setAccessToken } from './auth/tokenStorage'

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
    const fetchMock = vi.spyOn(window, 'fetch').mockImplementation(async (input) => {
      const url = String(input)
      if (url.endsWith('/api/v1/auth/login')) {
        return jsonResponse({ access_token: 'access-token', token_type: 'bearer', expires_in: 3600 })
      }
      if (url.endsWith('/api/v1/users/me')) {
        return jsonResponse(profile)
      }
      return jsonResponse({ code: 'not_found', message: 'Not found', details: {}, request_id: null }, 404)
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

describe('chat lifecycle', () => {
  test('submits a question, polls result, and renders sources', async () => {
    setAccessToken('access-token')
    vi.spyOn(window, 'fetch').mockImplementation(async (input, init) => {
      const url = String(input)
      if (url.endsWith('/api/v1/users/me')) {
        return jsonResponse(profile)
      }
      if (url.endsWith('/api/v1/chat/messages') && init?.method === 'POST') {
        return jsonResponse(
          {
            request_id: 'req-1',
            status: 'queued',
            original_request_id: null,
            answer: null,
            error: null,
            clarification: null,
          },
          202,
        )
      }
      if (url.endsWith('/api/v1/chat/messages/req-1')) {
        return jsonResponse({
          request_id: 'req-1',
          status: 'succeeded',
          original_request_id: null,
          error: null,
          clarification: null,
          answer: {
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
          },
        })
      }
      return jsonResponse({ code: 'not_found', message: 'Not found', details: {}, request_id: null }, 404)
    })

    renderApp('/chat')
    await userEvent.type(
      await screen.findByLabelText(/клинический вопрос/i),
      'Какая терапия рекомендована?',
    )
    await userEvent.click(screen.getByRole('button', { name: /отправить/i }))

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
    vi.spyOn(window, 'fetch').mockImplementation(async (input, init) => {
      const url = String(input)
      if (url.endsWith('/api/v1/users/me')) {
        return jsonResponse(profile)
      }
      if (url.endsWith('/api/v1/chat/messages') && init?.method === 'POST') {
        return jsonResponse(
          {
            request_id: 'req-running',
            status: 'queued',
            original_request_id: null,
            answer: null,
            error: null,
            clarification: null,
          },
          202,
        )
      }
      if (url.endsWith('/api/v1/chat/messages/req-running')) {
        return jsonResponse({
          request_id: 'req-running',
          status: 'running',
          original_request_id: null,
          answer: null,
          error: null,
          clarification: null,
        })
      }
      return jsonResponse({ code: 'not_found', message: 'Not found', details: {}, request_id: null }, 404)
    })

    renderApp('/chat')
    await userEvent.type(await screen.findByLabelText(/клинический вопрос/i), 'Проверка running статуса')
    await userEvent.click(screen.getByRole('button', { name: /отправить/i }))

    expect(await screen.findByText(/выполняется/i)).toBeInTheDocument()
    expect(screen.getByText(/генерируем ответ/i)).toBeInTheDocument()
  })

  test('renders failed status with retry action', async () => {
    setAccessToken('access-token')
    vi.spyOn(window, 'fetch').mockImplementation(async (input, init) => {
      const url = String(input)
      if (url.endsWith('/api/v1/users/me')) {
        return jsonResponse(profile)
      }
      if (url.endsWith('/api/v1/chat/messages') && init?.method === 'POST') {
        return jsonResponse(
          {
            request_id: 'req-failed',
            status: 'queued',
            original_request_id: null,
            answer: null,
            error: null,
            clarification: null,
          },
          202,
        )
      }
      if (url.endsWith('/api/v1/chat/messages/req-failed')) {
        return jsonResponse({
          request_id: 'req-failed',
          status: 'failed',
          original_request_id: null,
          answer: null,
          error: 'Ошибка worker',
          clarification: null,
        })
      }
      return jsonResponse({ code: 'not_found', message: 'Not found', details: {}, request_id: null }, 404)
    })

    renderApp('/chat')
    await userEvent.type(await screen.findByLabelText(/клинический вопрос/i), 'Проверка failed статуса')
    await userEvent.click(screen.getByRole('button', { name: /отправить/i }))

    expect(await screen.findByText(/ошибка worker/i)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /повторить/i })).toBeInTheDocument()
  })

  test('renders cancelled status with retry action', async () => {
    setAccessToken('access-token')
    vi.spyOn(window, 'fetch').mockImplementation(async (input, init) => {
      const url = String(input)
      if (url.endsWith('/api/v1/users/me')) {
        return jsonResponse(profile)
      }
      if (url.endsWith('/api/v1/chat/messages') && init?.method === 'POST') {
        return jsonResponse(
          {
            request_id: 'req-cancelled',
            status: 'queued',
            original_request_id: null,
            answer: null,
            error: null,
            clarification: null,
          },
          202,
        )
      }
      if (url.endsWith('/api/v1/chat/messages/req-cancelled')) {
        return jsonResponse({
          request_id: 'req-cancelled',
          status: 'cancelled',
          original_request_id: null,
          answer: null,
          error: null,
          clarification: null,
        })
      }
      return jsonResponse({ code: 'not_found', message: 'Not found', details: {}, request_id: null }, 404)
    })

    renderApp('/chat')
    await userEvent.type(await screen.findByLabelText(/клинический вопрос/i), 'Проверка cancelled статуса')
    await userEvent.click(screen.getByRole('button', { name: /отправить/i }))

    expect(await screen.findByText(/запрос отменен/i)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /повторить/i })).toBeInTheDocument()
  })

  test('renders awaiting clarification status and question', async () => {
    setAccessToken('access-token')
    vi.spyOn(window, 'fetch').mockImplementation(async (input, init) => {
      const url = String(input)
      if (url.endsWith('/api/v1/users/me')) {
        return jsonResponse(profile)
      }
      if (url.endsWith('/api/v1/chat/messages') && init?.method === 'POST') {
        return jsonResponse(
          {
            request_id: 'req-await',
            status: 'queued',
            original_request_id: null,
            answer: null,
            error: null,
            clarification: null,
          },
          202,
        )
      }
      if (url.endsWith('/api/v1/chat/messages/req-await')) {
        return jsonResponse({
          request_id: 'req-await',
          status: 'awaiting_clarification',
          original_request_id: null,
          answer: null,
          error: null,
          clarification: {
            question: 'Уточните возраст и сопутствующие заболевания?',
            answered: false,
            requested_at: null,
          },
        })
      }
      return jsonResponse({ code: 'not_found', message: 'Not found', details: {}, request_id: null }, 404)
    })

    renderApp('/chat')
    await userEvent.type(await screen.findByLabelText(/клинический вопрос/i), 'Как лечить АГ?')
    await userEvent.click(screen.getByRole('button', { name: /^отправить$/i }))

    expect(await screen.findByText(/требуется уточнение/i)).toBeInTheDocument()
    expect(screen.getByText(/уточните возраст и сопутствующие/i)).toBeInTheDocument()
    expect(screen.getByLabelText(/уточнение врача/i)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /отправить уточнение/i })).toBeInTheDocument()
  })

  test('submit clarification calls endpoint and resumes queued/running polling', async () => {
    setAccessToken('access-token')
    let clarificationPosted = false
    const fetchMock = vi.spyOn(window, 'fetch').mockImplementation(async (input, init) => {
      const url = String(input)
      if (url.endsWith('/api/v1/users/me')) {
        return jsonResponse(profile)
      }
      if (url.endsWith('/api/v1/chat/messages') && init?.method === 'POST') {
        return jsonResponse(
          {
            request_id: 'req-submit-clarify',
            status: 'queued',
            original_request_id: null,
            answer: null,
            error: null,
            clarification: null,
          },
          202,
        )
      }
      if (url.endsWith('/api/v1/chat/messages/req-submit-clarify/clarification') && init?.method === 'POST') {
        clarificationPosted = true
        expect(JSON.parse(init.body as string)).toEqual({ answer: 'Взрослый пациент 58 лет.' })
        return jsonResponse({
          request_id: 'req-submit-clarify',
          status: 'queued',
          original_request_id: null,
          answer: null,
          error: null,
          clarification: null,
        })
      }
      if (url.endsWith('/api/v1/chat/messages/req-submit-clarify')) {
        if (!clarificationPosted) {
          return jsonResponse({
            request_id: 'req-submit-clarify',
            status: 'awaiting_clarification',
            original_request_id: null,
            answer: null,
            error: null,
            clarification: {
              question: 'Уточните контекст пациента?',
              answered: false,
              requested_at: null,
            },
          })
        }
        return jsonResponse({
          request_id: 'req-submit-clarify',
          status: 'running',
          original_request_id: null,
          answer: null,
          error: null,
          clarification: null,
        })
      }
      return jsonResponse({ code: 'not_found', message: 'Not found', details: {}, request_id: null }, 404)
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
          ([u, options]) =>
            String(u).endsWith('/api/v1/chat/messages/req-submit-clarify/clarification')
            && (options as RequestInit).method === 'POST',
        ),
      ).toBe(true)
    })

    expect(await screen.findByText(/выполняется/i)).toBeInTheDocument()
    expect(screen.getByText(/генерируем ответ/i)).toBeInTheDocument()
  })

  test('cancel remains available for awaiting clarification', async () => {
    setAccessToken('access-token')
    vi.spyOn(window, 'fetch').mockImplementation(async (input, init) => {
      const url = String(input)
      if (url.endsWith('/api/v1/users/me')) {
        return jsonResponse(profile)
      }
      if (url.endsWith('/api/v1/chat/messages') && init?.method === 'POST') {
        return jsonResponse(
          {
            request_id: 'req-await-cancel',
            status: 'queued',
            original_request_id: null,
            answer: null,
            error: null,
            clarification: null,
          },
          202,
        )
      }
      if (url.endsWith('/api/v1/chat/messages/req-await-cancel/cancel') && init?.method === 'POST') {
        return jsonResponse({
          request_id: 'req-await-cancel',
          status: 'cancelled',
          original_request_id: null,
          answer: null,
          error: null,
          clarification: null,
        })
      }
      if (url.endsWith('/api/v1/chat/messages/req-await-cancel')) {
        return jsonResponse({
          request_id: 'req-await-cancel',
          status: 'awaiting_clarification',
          original_request_id: null,
          answer: null,
          error: null,
          clarification: {
            question: 'Нужны ли детали коморбидности?',
            answered: false,
            requested_at: null,
          },
        })
      }
      return jsonResponse({ code: 'not_found', message: 'Not found', details: {}, request_id: null }, 404)
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
