import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import type { FormEvent } from 'react'
import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  cancelChatMessage,
  createChatConversation,
  createChatMessage,
  deleteChatConversation,
  getChatConversationMessages,
  getChatMessage,
  listChatConversations,
  retryChatMessage,
  submitChatClarification,
} from '../api/cadenceApi'
import { ApiError } from '../api/client'
import type {
  ChatConversationListResponse,
  ChatConversationResponse,
  ChatMessageHistoryResponse,
  RAGRequestStatus,
  RAGRequestStatusResponse,
} from '../api/types'
import { PageShell } from '../components/PageShell'
import { SourceList } from '../components/SourceList'
import { StatusBadge } from '../components/StatusBadge'
import { Alert } from '../components/ui/Alert'
import { Button } from '../components/ui/Button'
import { Spinner } from '../components/ui/Spinner'

type ChatTurn = RAGRequestStatusResponse & {
  query: string
}

const terminalStatuses = new Set<RAGRequestStatus>(['succeeded', 'failed', 'cancelled'])

const promptSuggestions = [
  'Какие препараты первой линии рекомендованы при артериальной гипертензии?',
  'Когда показана госпитализация при внебольничной пневмонии?',
  'Какие критерии контроля бронхиальной астмы использовать на приеме?',
]

function isTerminal(status: RAGRequestStatus): boolean {
  return terminalStatuses.has(status)
}

function shouldPollStatus(status: RAGRequestStatus): boolean {
  return status === 'queued' || status === 'running'
}

function errorMessage(error: unknown): string {
  if (error instanceof ApiError) {
    if (error.status === 404) {
      return 'Чат не найден или уже удален.'
    }
    if (error.status === 429) {
      return 'Сработал лимит запросов. Попробуйте немного позже.'
    }
    if (error.status === 503) {
      return 'Очередь RAG временно перегружена. Повторите запрос позже.'
    }
    if (error.status === 409) {
      return error.response?.message ?? 'Действие недоступно для текущего статуса запроса.'
    }
    return error.response?.message ?? 'Не удалось выполнить действие.'
  }
  return 'Сервис временно недоступен.'
}

function updateTurn(turns: ChatTurn[], update: RAGRequestStatusResponse): ChatTurn[] {
  return turns.map((turn) => (
    turn.request_id === update.request_id ? { ...turn, ...update } : turn
  ))
}

function makeRequestKey(): string {
  return crypto.randomUUID()
}

function makeChatTitle(clinicalQuery: string): string {
  const normalized = clinicalQuery.replace(/\s+/g, ' ').trim()
  return normalized.length > 72 ? `${normalized.slice(0, 69)}...` : normalized
}

function formatChatDate(value: string | null): string {
  if (!value) {
    return 'Без сообщений'
  }
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) {
    return value
  }
  return new Intl.DateTimeFormat('ru-RU', {
    day: '2-digit',
    month: 'short',
    hour: '2-digit',
    minute: '2-digit',
  }).format(date)
}

function conversationTitle(
  conversation: ChatConversationResponse,
  activeTurns: ChatTurn[],
  isActive: boolean,
): string {
  if (conversation.title?.trim()) {
    return conversation.title
  }
  if (isActive && activeTurns[0]?.query) {
    return makeChatTitle(activeTurns[0].query)
  }
  if (isActive && activeTurns.length === 0) {
    return 'Новый чат'
  }
  return `Чат ${formatChatDate(
    conversation.last_message_at ?? conversation.updated_at ?? conversation.created_at,
  )}`
}

const chatPollQueryKey = ['chat-message-poll'] as const
const chatConversationsQueryKey = ['chat-conversations'] as const

function chatConversationMessagesQueryKey(chatId: string) {
  return ['chat-conversation-messages', chatId] as const
}

export function ChatPage() {
  const queryClient = useQueryClient()
  const [activeChatId, setActiveChatId] = useState<string | null>(null)
  const [query, setQuery] = useState('')
  const [turns, setTurns] = useState<ChatTurn[]>([])
  const [actionError, setActionError] = useState<string | null>(null)
  const [clarificationDrafts, setClarificationDrafts] = useState<Record<string, string>>({})

  const conversationsQuery = useQuery({
    queryKey: chatConversationsQueryKey,
    queryFn: () => listChatConversations({ limit: 50 }),
  })

  const conversationMessagesQuery = useQuery({
    queryKey: activeChatId
      ? chatConversationMessagesQueryKey(activeChatId)
      : ['chat-conversation-messages', 'none'],
    queryFn: () => getChatConversationMessages(activeChatId ?? ''),
    enabled: Boolean(activeChatId),
    staleTime: 10_000,
  })

  const conversations = conversationsQuery.data?.items ?? []
  const activeConversation = conversations.find((conversation) => conversation.id === activeChatId)
    ?? null

  const activeTurn = useMemo(
    () => [...turns].reverse().find((turn) => !isTerminal(turn.status)),
    [turns],
  )

  const pollableTurn = useMemo(() => {
    if (!activeTurn) {
      return undefined
    }
    return shouldPollStatus(activeTurn.status) ? activeTurn : undefined
  }, [activeTurn])

  const refreshChatCaches = useCallback((chatId: string | null | undefined): void => {
    void queryClient.invalidateQueries({ queryKey: chatConversationsQueryKey })
    if (chatId) {
      void queryClient.invalidateQueries({ queryKey: chatConversationMessagesQueryKey(chatId) })
    }
  }, [queryClient])

  const removeChatFromList = useCallback((chatId: string): void => {
    queryClient.setQueryData<ChatConversationListResponse>(
      chatConversationsQueryKey,
      (current) => current
        ? {
          ...current,
          items: current.items.filter((conversation) => conversation.id !== chatId),
        }
        : current,
    )
  }, [queryClient])

  const addConversationToList = useCallback((conversation: ChatConversationResponse): void => {
    queryClient.setQueryData<ChatConversationListResponse>(
      chatConversationsQueryKey,
      (current) => current
        ? {
          ...current,
          items: [
            conversation,
            ...current.items.filter((item) => item.id !== conversation.id),
          ],
        }
        : current,
    )
  }, [queryClient])

  const setEmptyMessagesCache = useCallback((chatId: string): void => {
    queryClient.setQueryData<ChatMessageHistoryResponse>(
      chatConversationMessagesQueryKey(chatId),
      { items: [] },
    )
  }, [queryClient])

  const upsertTurnInMessagesCache = useCallback((chatId: string, turn: ChatTurn): void => {
    queryClient.setQueryData<ChatMessageHistoryResponse>(
      chatConversationMessagesQueryKey(chatId),
      (current) => {
        const items = current?.items ?? []
        const nextTurn = { query: turn.query, request: turn }
        const hasTurn = items.some((item) => item.request.request_id === turn.request_id)
        return {
          items: hasTurn
            ? items.map((item) => (
              item.request.request_id === turn.request_id ? nextTurn : item
            ))
            : [...items, nextTurn],
        }
      },
    )
  }, [queryClient])

  const updateRequestInMessagesCache = useCallback((
    chatId: string | null | undefined,
    update: RAGRequestStatusResponse,
  ): void => {
    if (!chatId) {
      return
    }
    queryClient.setQueryData<ChatMessageHistoryResponse>(
      chatConversationMessagesQueryKey(chatId),
      (current) => current
        ? {
          items: current.items.map((item) => (
            item.request.request_id === update.request_id
              ? { ...item, request: { ...item.request, ...update } }
              : item
          )),
        }
        : current,
    )
  }, [queryClient])

  const resetActiveChat = useCallback((): void => {
    setActiveChatId(null)
    setTurns([])
    setClarificationDrafts({})
    void queryClient.removeQueries({ queryKey: chatPollQueryKey })
  }, [queryClient])

  const selectChat = useCallback((chatId: string): void => {
    if (chatId === activeChatId) {
      return
    }
    setActiveChatId(chatId)
    setTurns([])
    setClarificationDrafts({})
    setActionError(null)
    void queryClient.removeQueries({ queryKey: chatPollQueryKey })
  }, [activeChatId, queryClient])

  const createMutation = useMutation({
    mutationFn: async (clinicalQuery: string) => {
      let chatId = activeChatId
      let createdConversation: ChatConversationResponse | null = null

      if (!chatId) {
        createdConversation = await createChatConversation({ title: makeChatTitle(clinicalQuery) })
        chatId = createdConversation.id
      }

      const idempotencyKey = makeRequestKey()
      const response = await createChatMessage(
        {
          query: clinicalQuery,
          chat_id: chatId,
          idempotency_key: idempotencyKey,
        },
        idempotencyKey,
      )
      return {
        response: {
          ...response,
          chat_id: response.chat_id ?? chatId,
        },
        clinicalQuery,
        createdConversation,
      }
    },
    onSuccess: ({ response, clinicalQuery, createdConversation }) => {
      const chatId = response.chat_id
      if (createdConversation) {
        addConversationToList(createdConversation)
      }
      const turn = { ...response, query: clinicalQuery }
      setActiveChatId(chatId)
      setTurns((current) => [...current, turn])
      setQuery('')
      if (chatId) {
        upsertTurnInMessagesCache(chatId, turn)
      }
      void queryClient.invalidateQueries({ queryKey: chatConversationsQueryKey })
    },
    onError: (error) => setActionError(errorMessage(error)),
  })

  const createConversationMutation = useMutation({
    mutationFn: () => createChatConversation({ title: null }),
    onSuccess: (conversation) => {
      addConversationToList(conversation)
      setEmptyMessagesCache(conversation.id)
      setActiveChatId(conversation.id)
      setTurns([])
      setClarificationDrafts({})
      setActionError(null)
      void queryClient.removeQueries({ queryKey: chatPollQueryKey })
      void queryClient.invalidateQueries({ queryKey: chatConversationsQueryKey })
    },
    onError: (error) => setActionError(errorMessage(error)),
  })

  const cancelMutation = useMutation({
    mutationFn: cancelChatMessage,
    onSuccess: (response) => {
      setActionError(null)
      setTurns((current) => updateTurn(current, response))
      updateRequestInMessagesCache(response.chat_id ?? activeChatId, response)
      refreshChatCaches(response.chat_id ?? activeChatId)
    },
    onError: (error) => setActionError(errorMessage(error)),
  })

  const clarificationMutation = useMutation({
    mutationFn: ({ requestId, answer }: { requestId: string; answer: string }) =>
      submitChatClarification(requestId, { answer }),
    onSuccess: (response) => {
      setActionError(null)
      /** Avoid merging stale poll cache (e.g. awaiting_clarification) after resume. */
      queryClient.removeQueries({ queryKey: chatPollQueryKey })
      setTurns((current) => updateTurn(current, response))
      updateRequestInMessagesCache(response.chat_id ?? activeChatId, response)
      refreshChatCaches(response.chat_id ?? activeChatId)
      setClarificationDrafts((prev) => {
        const next = { ...prev }
        delete next[response.request_id]
        return next
      })
    },
    onError: (error) => setActionError(errorMessage(error)),
  })

  const retryMutation = useMutation({
    mutationFn: async (turn: ChatTurn) => {
      const response = await retryChatMessage(turn.request_id)
      return { response, clinicalQuery: turn.query }
    },
    onSuccess: ({ response, clinicalQuery }) => {
      const turn = { ...response, query: clinicalQuery }
      setTurns((current) => [...current, turn])
      if (response.chat_id ?? activeChatId) {
        upsertTurnInMessagesCache((response.chat_id ?? activeChatId) as string, turn)
      }
      refreshChatCaches(response.chat_id ?? activeChatId)
    },
    onError: (error) => setActionError(errorMessage(error)),
  })

  const deleteConversationMutation = useMutation({
    mutationFn: deleteChatConversation,
    onSuccess: (_result, chatId) => {
      removeChatFromList(chatId)
      queryClient.removeQueries({ queryKey: chatConversationMessagesQueryKey(chatId) })
      if (activeChatId === chatId) {
        resetActiveChat()
      }
      setActionError(null)
      void queryClient.invalidateQueries({ queryKey: chatConversationsQueryKey })
    },
    onError: (error, chatId) => {
      if (error instanceof ApiError && error.status === 404) {
        removeChatFromList(chatId)
        queryClient.removeQueries({ queryKey: chatConversationMessagesQueryKey(chatId) })
        if (activeChatId === chatId) {
          resetActiveChat()
        }
        void queryClient.invalidateQueries({ queryKey: chatConversationsQueryKey })
        setActionError('Чат уже удален или недоступен.')
        return
      }
      setActionError(errorMessage(error))
    },
  })

  const pollQuery = useQuery({
    queryKey: [...chatPollQueryKey, activeChatId, pollableTurn?.request_id],
    queryFn: () => getChatMessage(pollableTurn?.request_id ?? ''),
    enabled: Boolean(pollableTurn),
    refetchInterval: (queryResult) => {
      const data = queryResult.state.data
      return data && isTerminal(data.status) ? false : 2500
    },
  })

  useEffect(() => {
    if (pollQuery.data) {
      setTurns((current) => updateTurn(current, pollQuery.data))
      updateRequestInMessagesCache(pollQuery.data.chat_id ?? activeChatId, pollQuery.data)
      if (isTerminal(pollQuery.data.status)) {
        refreshChatCaches(pollQuery.data.chat_id ?? activeChatId)
      }
    }
  }, [activeChatId, pollQuery.data, refreshChatCaches, updateRequestInMessagesCache])

  useEffect(() => {
    if (!conversationMessagesQuery.data) {
      return
    }

    setTurns(
      conversationMessagesQuery.data.items.map((turn) => ({
        ...turn.request,
        query: turn.query,
      })),
    )
  }, [conversationMessagesQuery.data])

  useEffect(() => {
    if (!conversationMessagesQuery.error || !activeChatId) {
      return
    }

    if (
      conversationMessagesQuery.error instanceof ApiError
      && conversationMessagesQuery.error.status === 404
    ) {
      setActionError('Чат не найден или уже удален. Список чатов обновлен.')
      removeChatFromList(activeChatId)
      resetActiveChat()
      void queryClient.invalidateQueries({ queryKey: chatConversationsQueryKey })
      return
    }

    setActionError(errorMessage(conversationMessagesQuery.error))
  }, [
    activeChatId,
    conversationMessagesQuery.error,
    queryClient,
    removeChatFromList,
    resetActiveChat,
  ])

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    const normalized = query.trim()
    if (!normalized) {
      return
    }
    setActionError(null)
    createMutation.mutate(normalized)
  }

  function handleClarificationSubmit(requestId: string) {
    const normalized = (clarificationDrafts[requestId] ?? '').trim()
    if (!normalized) {
      return
    }
    setActionError(null)
    clarificationMutation.mutate({ requestId, answer: normalized })
  }

  const isHistoryLoading = Boolean(activeChatId && conversationMessagesQuery.isLoading)

  return (
    <PageShell>
      <section className="chat-layout">
        <div className="chat-main">
          <header className="page-header">
            <span className="eyebrow">RAG Chat</span>
            <h1>Задайте клинический вопрос</h1>
            <p>
              Ассистент отвечает по найденным клиническим рекомендациям и показывает источники для
              проверки.
            </p>
          </header>

          {actionError ? <Alert tone="error">{actionError}</Alert> : null}

          <div className="conversation" aria-live="polite">
            {isHistoryLoading ? (
              <div className="empty-chat">
                <Spinner label="Загружаем историю чата" />
              </div>
            ) : null}

            {!isHistoryLoading && turns.length === 0 ? (
              <div className="empty-chat">
                <span className="pulse-dot" />
                <h2>{activeChatId ? 'В этом чате пока нет сообщений' : 'Начните новый чат'}</h2>
                <p>
                  Укажите состояние пациента, контекст и желаемый тип ответа: диагностика,
                  терапия, критерии госпитализации или мониторинг.
                </p>
              </div>
            ) : null}

            {turns.map((turn) => (
              <article className="chat-turn" key={turn.request_id}>
                <div className="message user-message">
                  <span className="message-label">Вопрос врача</span>
                  <p>{turn.query}</p>
                </div>
                <div className="message assistant-message">
                  <div className="assistant-header">
                    <span className="message-label">CADENCE-MD</span>
                    <StatusBadge status={turn.status} />
                  </div>
                  {turn.status === 'queued' || turn.status === 'running' ? (
                    <Spinner label={turn.status === 'queued' ? 'Ожидаем worker' : 'Генерируем ответ'} />
                  ) : null}
                  {turn.status === 'failed' ? (
                    <Alert tone="error">{turn.error ?? 'Запрос завершился ошибкой.'}</Alert>
                  ) : null}
                  {turn.status === 'cancelled' ? (
                    <Alert tone="warning">Запрос отменен. Его можно запустить повторно.</Alert>
                  ) : null}
                  {turn.status === 'awaiting_clarification' && turn.clarification ? (
                    <div className="clarification-panel">
                      <Alert tone="info">
                        <span className="clarification-question">{turn.clarification.question}</span>
                      </Alert>
                      <div className="clarification-form">
                        <label htmlFor={`clarification-${turn.request_id}`}>Уточнение врача</label>
                        <textarea
                          id={`clarification-${turn.request_id}`}
                          className="clarification-textarea"
                          value={clarificationDrafts[turn.request_id] ?? ''}
                          maxLength={4000}
                          onChange={(event) =>
                            setClarificationDrafts((prev) => ({
                              ...prev,
                              [turn.request_id]: event.target.value,
                            }))
                          }
                          placeholder="Уточните контекст или выберите допущения..."
                          rows={3}
                        />
                        <div className="clarification-footer">
                          <span>
                            {(clarificationDrafts[turn.request_id] ?? '').length}/4000
                          </span>
                          <Button
                            type="button"
                            disabled={
                              clarificationMutation.isPending
                              || !(clarificationDrafts[turn.request_id] ?? '').trim()
                            }
                            onClick={() => handleClarificationSubmit(turn.request_id)}
                          >
                            {clarificationMutation.isPending ? 'Отправляем...' : 'Отправить уточнение'}
                          </Button>
                        </div>
                      </div>
                    </div>
                  ) : null}
                  {turn.status === 'succeeded' && turn.answer ? (
                    <>
                      <p className="answer-text">{turn.answer.answer}</p>
                      <SourceList requestId={turn.request_id} sources={turn.answer.sources} />
                    </>
                  ) : null}
                  <div className="turn-actions">
                    {(turn.status === 'queued'
                      || turn.status === 'running'
                      || turn.status === 'awaiting_clarification') ? (
                      <Button
                        variant="secondary"
                        disabled={cancelMutation.isPending}
                        onClick={() => cancelMutation.mutate(turn.request_id)}
                      >
                        Отменить
                      </Button>
                    ) : null}
                    {(turn.status === 'failed' || turn.status === 'cancelled') ? (
                      <Button
                        variant="secondary"
                        disabled={retryMutation.isPending}
                        onClick={() => retryMutation.mutate(turn)}
                      >
                        Повторить
                      </Button>
                    ) : null}
                  </div>
                </div>
              </article>
            ))}
          </div>

          <form className="composer" onSubmit={handleSubmit}>
            <label htmlFor="clinical-query">Клинический вопрос</label>
            <textarea
              id="clinical-query"
              value={query}
              maxLength={4000}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Например: пациент 58 лет, впервые выявленная АГ..."
              rows={4}
            />
            <div className="composer-footer">
              <span>{query.length}/4000</span>
              <Button
                type="submit"
                disabled={createMutation.isPending || isHistoryLoading || !query.trim()}
              >
                {createMutation.isPending ? 'Отправляем...' : 'Отправить'}
              </Button>
            </div>
          </form>
        </div>

        <aside className="assistant-panel chat-sidebar">
          <div className="chat-history-header">
            <div>
              <span className="eyebrow">История</span>
              <h2>Ваши чаты</h2>
            </div>
            <Button
              type="button"
              variant="secondary"
              disabled={createConversationMutation.isPending}
              onClick={() => createConversationMutation.mutate()}
            >
              {createConversationMutation.isPending ? 'Создаем...' : 'Новый чат'}
            </Button>
          </div>

          {conversationsQuery.isLoading ? <Spinner label="Загружаем список чатов" /> : null}
          {conversationsQuery.error ? (
            <Alert tone="error">{errorMessage(conversationsQuery.error)}</Alert>
          ) : null}
          {!conversationsQuery.isLoading && conversations.length === 0 ? (
            <p className="chat-history-empty">Сохраненных чатов пока нет.</p>
          ) : null}

          {conversations.length > 0 ? (
            <ul className="chat-history-list" aria-label="Список чатов">
              {conversations.map((conversation) => {
                const isActive = conversation.id === activeChatId
                const title = conversationTitle(conversation, turns, isActive)
                return (
                  <li
                    className={`chat-history-row ${isActive ? 'active' : ''}`}
                    key={conversation.id}
                  >
                    <button
                      type="button"
                      className="chat-history-item"
                      aria-current={isActive ? 'true' : undefined}
                      onClick={() => selectChat(conversation.id)}
                    >
                      <span className="chat-history-title">{title}</span>
                      <span className="chat-history-meta">
                        {formatChatDate(conversation.last_message_at ?? conversation.updated_at)}
                      </span>
                    </button>
                    <Button
                      type="button"
                      variant="ghost"
                      className="chat-history-delete"
                      aria-label={`Удалить чат ${title}`}
                      disabled={deleteConversationMutation.isPending}
                      onClick={() => deleteConversationMutation.mutate(conversation.id)}
                    >
                      Удалить
                    </Button>
                  </li>
                )
              })}
            </ul>
          ) : null}

          {activeConversation ? (
            <p className="chat-history-active">
              Открыт: {conversationTitle(activeConversation, turns, true)}
            </p>
          ) : null}

          <div className="suggestion-section">
            <h2>Подсказки</h2>
            <div className="suggestion-list">
              {promptSuggestions.map((suggestion) => (
                <button key={suggestion} type="button" onClick={() => setQuery(suggestion)}>
                  {suggestion}
                </button>
              ))}
            </div>
          </div>
          <Alert tone="info">
            Ответы носят информационный характер и требуют клинической валидации врачом.
          </Alert>
        </aside>
      </section>
    </PageShell>
  )
}
