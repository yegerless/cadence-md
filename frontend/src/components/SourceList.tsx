import { useEffect, useRef, useState } from 'react'
import type { MouseEvent } from 'react'
import { downloadChatSource } from '../api/cadenceApi'
import type { RAGSourceResponse } from '../api/types'

const SOURCE_PREVIEW_CHARS = 180

function sourceTitle(source: RAGSourceResponse): string {
  return source.document_title || source.filename || `Источник ${source.rank}`
}

function downloadFilename(source: RAGSourceResponse): string {
  return source.filename || `source-${source.rank}.pdf`
}

function sourcePreviewText(content: string): string {
  const normalized = content.replace(/\s+/g, ' ').trim()
  if (normalized.length <= SOURCE_PREVIEW_CHARS) {
    return normalized
  }

  const breakpoint = normalized.lastIndexOf(' ', SOURCE_PREVIEW_CHARS)
  const previewEnd = breakpoint > SOURCE_PREVIEW_CHARS * 0.65 ? breakpoint : SOURCE_PREVIEW_CHARS
  return `${normalized.slice(0, previewEnd).trimEnd()}...`
}

function SourceChunkDialog({
  source,
  onClose,
}: {
  source: RAGSourceResponse
  onClose: () => void
}) {
  const closeButtonRef = useRef<HTMLButtonElement | null>(null)
  const titleId = `source-dialog-title-${source.rank}`

  useEffect(() => {
    closeButtonRef.current?.focus()

    function handleKeyDown(event: KeyboardEvent): void {
      if (event.key === 'Escape') {
        onClose()
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [onClose])

  function handleBackdropMouseDown(event: MouseEvent<HTMLDivElement>): void {
    if (event.target === event.currentTarget) {
      onClose()
    }
  }

  return (
    <div className="source-dialog-backdrop" onMouseDown={handleBackdropMouseDown}>
      <section
        aria-labelledby={titleId}
        aria-modal="true"
        className="source-dialog"
        role="dialog"
      >
        <div className="source-dialog-header">
          <div>
            <span className="source-ref">{source.doc_ref}</span>
            <h3 id={titleId}>{sourceTitle(source)}</h3>
            {source.section_title ? <p className="source-section">{source.section_title}</p> : null}
          </div>
          <button
            className="source-dialog-close"
            ref={closeButtonRef}
            type="button"
            onClick={onClose}
          >
            Закрыть
          </button>
        </div>
        <div className="source-dialog-content">{source.content}</div>
      </section>
    </div>
  )
}

export function SourceList({
  requestId,
  sources,
}: {
  requestId: string
  sources: RAGSourceResponse[]
}) {
  const [downloadingRank, setDownloadingRank] = useState<number | null>(null)
  const [downloadError, setDownloadError] = useState<string | null>(null)
  const [selectedSource, setSelectedSource] = useState<RAGSourceResponse | null>(null)
  const lastTriggerRef = useRef<HTMLButtonElement | null>(null)

  if (sources.length === 0) {
    return <p className="muted">Источники не вернулись в ответе.</p>
  }

  async function handleDownload(source: RAGSourceResponse): Promise<void> {
    setDownloadError(null)
    setDownloadingRank(source.rank)
    try {
      const blob = await downloadChatSource(requestId, source.rank)
      const objectUrl = URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = objectUrl
      link.download = downloadFilename(source)
      document.body.appendChild(link)
      link.click()
      link.remove()
      URL.revokeObjectURL(objectUrl)
    } catch {
      setDownloadError('Не удалось скачать PDF источника.')
    } finally {
      setDownloadingRank(null)
    }
  }

  function handleOpenSource(
    source: RAGSourceResponse,
    event: MouseEvent<HTMLButtonElement>,
  ): void {
    lastTriggerRef.current = event.currentTarget
    setSelectedSource(source)
  }

  function handleCloseSource(): void {
    setSelectedSource(null)
    lastTriggerRef.current?.focus()
  }

  return (
    <>
      {downloadError ? <p className="source-download-error">{downloadError}</p> : null}
      <div className="source-grid" aria-label="Источники ответа">
        {sources.map((source) => (
          <article className="source-card" key={`${source.rank}-${source.chunk_id ?? source.doc_ref}`}>
            <div className="source-card-header">
              <span className="source-ref">{source.doc_ref}</span>
              {source.score !== null ? <span className="source-score">{source.score.toFixed(3)}</span> : null}
            </div>
            <h4>{sourceTitle(source)}</h4>
            {source.section_title ? <p className="source-section">{source.section_title}</p> : null}
            {source.content ? (
              <button
                aria-label={`Показать полный текст чанка ${source.doc_ref}`}
                className="source-content-preview"
                type="button"
                onClick={(event) => handleOpenSource(source, event)}
              >
                {sourcePreviewText(source.content)}
              </button>
            ) : null}
            <button
              className="source-download"
              type="button"
              disabled={!source.source_path || downloadingRank === source.rank}
              onClick={() => void handleDownload(source)}
            >
              {downloadingRank === source.rank ? 'Скачиваем...' : 'Скачать PDF'}
            </button>
          </article>
        ))}
      </div>
      {selectedSource ? (
        <SourceChunkDialog source={selectedSource} onClose={handleCloseSource} />
      ) : null}
    </>
  )
}
