import { clearAccessToken, getAccessToken } from '../auth/tokenStorage'
import type { ErrorResponse } from './types'

const DEFAULT_API_BASE_URL = 'http://127.0.0.1:8000'

type RequestOptions = {
  method?: 'GET' | 'POST'
  body?: unknown
  auth?: boolean
  headers?: HeadersInit
}

let unauthorizedHandler: (() => void) | undefined

export class ApiError extends Error {
  readonly status: number
  readonly response: ErrorResponse | null

  constructor(status: number, response: ErrorResponse | null) {
    super(response?.message ?? `Request failed with status ${status}`)
    this.name = 'ApiError'
    this.status = status
    this.response = response
  }
}

export function setUnauthorizedHandler(handler: (() => void) | undefined): void {
  unauthorizedHandler = handler
}

export function getApiBaseUrl(): string {
  const configuredUrl = import.meta.env.VITE_API_BASE_URL as string | undefined
  return configuredUrl === undefined ? DEFAULT_API_BASE_URL : configuredUrl.replace(/\/$/, '')
}

async function parseErrorResponse(response: Response): Promise<ErrorResponse | null> {
  try {
    return (await response.json()) as ErrorResponse
  } catch {
    return null
  }
}

export async function apiRequest<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const headers = new Headers(options.headers)
  const hasBody = options.body !== undefined

  if (hasBody && !headers.has('Content-Type')) {
    headers.set('Content-Type', 'application/json')
  }

  if (options.auth !== false) {
    const token = getAccessToken()
    if (token) {
      headers.set('Authorization', `Bearer ${token}`)
    }
  }

  const response = await fetch(`${getApiBaseUrl()}${path}`, {
    method: options.method ?? 'GET',
    headers,
    body: hasBody ? JSON.stringify(options.body) : undefined,
  })

  if (response.status === 401) {
    clearAccessToken()
    unauthorizedHandler?.()
  }

  if (!response.ok) {
    throw new ApiError(response.status, await parseErrorResponse(response))
  }

  return (await response.json()) as T
}
