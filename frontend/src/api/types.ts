export type ErrorResponse = {
  code: string
  message: string
  details: Record<string, unknown>
  request_id: string | null
}

export type RegisterRequest = {
  email: string
  password: string
  first_name?: string | null
  last_name?: string | null
}

export type LoginRequest = {
  email: string
  password: string
}

export type TokenResponse = {
  access_token: string
  token_type: 'bearer' | string
  expires_in: number
}

export type UserProfileResponse = {
  id: string
  email: string
  first_name: string | null
  last_name: string | null
}

export type RAGRequestStatus =
  | 'queued'
  | 'running'
  | 'awaiting_clarification'
  | 'succeeded'
  | 'failed'
  | 'cancelled'

export type CreateChatConversationRequest = {
  title?: string | null
}

export type ChatConversationResponse = {
  id: string
  title: string | null
  created_at: string
  updated_at: string
  last_message_at: string | null
}

export type ChatConversationListResponse = {
  items: ChatConversationResponse[]
  limit: number
  offset: number
}

export type CreateRAGRequest = {
  query: string
  conversation_id?: string | null
  chat_id?: string | null
  idempotency_key?: string | null
  metadata?: Record<string, string>
}

export type RAGSourceResponse = {
  rank: number
  doc_ref: string
  filename: string | null
  source_path: string | null
  document_title: string | null
  section_title: string | null
  section_id: string | null
  chunk_id: string | null
  content: string | null
  score: number | null
}

export type RAGAnswerResponse = {
  answer: string
  sources: RAGSourceResponse[]
  query_hash: string | null
  latency_ms: Record<string, number>
  flags: Record<string, boolean>
  error_type: string | null
  error_message: string | null
  langfuse_trace_id: string | null
}

export type ClarificationResponse = {
  question: string
  answered: boolean
  requested_at: string | null
}

export type SubmitClarificationRequest = {
  answer: string
}

export type RAGRequestStatusResponse = {
  request_id: string
  status: RAGRequestStatus
  chat_id: string | null
  original_request_id: string | null
  answer: RAGAnswerResponse | null
  error: string | null
  clarification: ClarificationResponse | null
}

export type ChatTurnResponse = {
  query: string
  request: RAGRequestStatusResponse
}

export type ChatMessageHistoryResponse = {
  items: ChatTurnResponse[]
}
