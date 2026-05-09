import { apiBlobRequest, apiRequest } from './client'
import type {
  ChatConversationListResponse,
  ChatConversationResponse,
  ChatMessageHistoryResponse,
  CreateChatConversationRequest,
  CreateRAGRequest,
  LoginRequest,
  RAGRequestStatusResponse,
  RegisterRequest,
  SubmitClarificationRequest,
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

export function createChatConversation(
  payload: CreateChatConversationRequest,
): Promise<ChatConversationResponse> {
  return apiRequest<ChatConversationResponse>('/api/v1/chat/conversations', {
    method: 'POST',
    body: payload,
  })
}

export function listChatConversations(params: { limit?: number, offset?: number } = {}):
Promise<ChatConversationListResponse> {
  const search = new URLSearchParams()
  if (params.limit !== undefined) {
    search.set('limit', String(params.limit))
  }
  if (params.offset !== undefined) {
    search.set('offset', String(params.offset))
  }
  const queryString = search.toString()
  return apiRequest<ChatConversationListResponse>(
    `/api/v1/chat/conversations${queryString ? `?${queryString}` : ''}`,
  )
}

export function getChatConversation(chatId: string): Promise<ChatConversationResponse> {
  return apiRequest<ChatConversationResponse>(`/api/v1/chat/conversations/${chatId}`)
}

export function getChatConversationMessages(chatId: string): Promise<ChatMessageHistoryResponse> {
  return apiRequest<ChatMessageHistoryResponse>(`/api/v1/chat/conversations/${chatId}/messages`)
}

export function deleteChatConversation(chatId: string): Promise<void> {
  return apiRequest<void>(`/api/v1/chat/conversations/${chatId}`, {
    method: 'DELETE',
  })
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

export function submitChatClarification(
  requestId: string,
  payload: SubmitClarificationRequest,
): Promise<RAGRequestStatusResponse> {
  return apiRequest<RAGRequestStatusResponse>(
    `/api/v1/chat/messages/${requestId}/clarification`,
    {
      method: 'POST',
      body: payload,
    },
  )
}

export function downloadChatSource(requestId: string, rank: number): Promise<Blob> {
  return apiBlobRequest(`/api/v1/chat/messages/${requestId}/sources/${rank}/download`)
}
