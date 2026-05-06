import { useState } from 'react'
import { downloadChatSource } from '../api/cadenceApi'
import type { RAGSourceResponse } from '../api/types'

function sourceTitle(source: RAGSourceResponse): string {
  return source.document_title || source.filename || `Источник ${source.rank}`
}

function downloadFilename(source: RAGSourceResponse): string {
  return source.filename || `source-${source.rank}.pdf`
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
            {source.content ? <p className="source-content">{source.content}</p> : null}
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
    </>
  )
}
