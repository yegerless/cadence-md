import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import type { FormEvent } from 'react'
import { useEffect, useMemo, useRef, useState } from 'react'
import {
  cancelChatMessage,
  createChatMessage,
  getChatMessage,
  retryChatMessage,
  submitChatClarification,
} from '../api/cadenceApi'
import { ApiError } from '../api/client'
import type { RAGRequestStatus, RAGRequestStatusResponse } from '../api/types'
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

const chatPollQueryKey = ['chat-message-poll'] as const

export function ChatPage() {
  const queryClient = useQueryClient()
  const conversationId = useRef(makeRequestKey())
  const [query, setQuery] = useState('')
  const [turns, setTurns] = useState<ChatTurn[]>([])
  const [actionError, setActionError] = useState<string | null>(null)
  const [clarificationDrafts, setClarificationDrafts] = useState<Record<string, string>>({})

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

  const createMutation = useMutation({
    mutationFn: async (clinicalQuery: string) => {
      const idempotencyKey = makeRequestKey()
      const response = await createChatMessage(
        {
          query: clinicalQuery,
          conversation_id: conversationId.current,
          idempotency_key: idempotencyKey,
        },
        idempotencyKey,
      )
      return { response, clinicalQuery }
    },
    onSuccess: ({ response, clinicalQuery }) => {
      setTurns((current) => [...current, { ...response, query: clinicalQuery }])
      setQuery('')
    },
    onError: (error) => setActionError(errorMessage(error)),
  })

  const cancelMutation = useMutation({
    mutationFn: cancelChatMessage,
    onSuccess: (response) => setTurns((current) => updateTurn(current, response)),
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
      setTurns((current) => [...current, { ...response, query: clinicalQuery }])
    },
    onError: (error) => setActionError(errorMessage(error)),
  })

  const pollQuery = useQuery({
    queryKey: [...chatPollQueryKey, pollableTurn?.request_id],
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
    }
  }, [pollQuery.data])

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
            {turns.length === 0 ? (
              <div className="empty-chat">
                <span className="pulse-dot" />
                <h2>Начните с конкретного клинического сценария</h2>
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
              <Button type="submit" disabled={createMutation.isPending || !query.trim()}>
                {createMutation.isPending ? 'Отправляем...' : 'Отправить'}
              </Button>
            </div>
          </form>
        </div>

        <aside className="assistant-panel">
          <h2>Подсказки</h2>
          <div className="suggestion-list">
            {promptSuggestions.map((suggestion) => (
              <button key={suggestion} type="button" onClick={() => setQuery(suggestion)}>
                {suggestion}
              </button>
            ))}
          </div>
          <Alert tone="info">
            Ответы носят информационный характер и требуют клинической валидации врачом.
          </Alert>
        </aside>
      </section>
    </PageShell>
  )
}
