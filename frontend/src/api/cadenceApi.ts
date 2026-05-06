import { apiRequest } from './client'
import type {
  CreateRAGRequest,
  LoginRequest,
  RAGRequestStatusResponse,
  RegisterRequest,
  TokenResponse,
  UserProfileResponse,
} from './types'

export function login(payload: LoginRequest): Promise<TokenResponse> {
  return apiRequest<TokenResponse>('/api/v1/auth/login', {
    method: 'POST',
    body: payload,
    auth: false,
  })
}

export function register(payload: RegisterRequest): Promise<TokenResponse> {
  return apiRequest<TokenResponse>('/api/v1/auth/register', {
    method: 'POST',
    body: payload,
    auth: false,
  })
}

export function getCurrentUser(): Promise<UserProfileResponse> {
  return apiRequest<UserProfileResponse>('/api/v1/users/me')
}

export function createChatMessage(
  payload: CreateRAGRequest,
  idempotencyKey: string,
): Promise<RAGRequestStatusResponse> {
  return apiRequest<RAGRequestStatusResponse>('/api/v1/chat/messages', {
    method: 'POST',
    headers: { 'Idempotency-Key': idempotencyKey },
    body: payload,
  })
}

export function getChatMessage(requestId: string): Promise<RAGRequestStatusResponse> {
  return apiRequest<RAGRequestStatusResponse>(`/api/v1/chat/messages/${requestId}`)
}

export function cancelChatMessage(requestId: string): Promise<RAGRequestStatusResponse> {
  return apiRequest<RAGRequestStatusResponse>(`/api/v1/chat/messages/${requestId}/cancel`, {
    method: 'POST',
  })
}

export function retryChatMessage(requestId: string): Promise<RAGRequestStatusResponse> {
  return apiRequest<RAGRequestStatusResponse>(`/api/v1/chat/messages/${requestId}/retry`, {
    method: 'POST',
  })
}
