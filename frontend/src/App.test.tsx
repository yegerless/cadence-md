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
          { request_id: 'req-1', status: 'queued', original_request_id: null, answer: null, error: null },
          202,
        )
      }
      if (url.endsWith('/api/v1/chat/messages/req-1')) {
        return jsonResponse({
          request_id: 'req-1',
          status: 'succeeded',
          original_request_id: null,
          error: null,
          answer: {
            answer: 'Ответ с цитатой [Doc 1].',
            sources: [{
              rank: 1,
              doc_ref: '[Doc 1]',
              filename: 'guideline.pdf',
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
})
