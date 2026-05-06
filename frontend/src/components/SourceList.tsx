import type { RAGSourceResponse } from '../api/types'

function sourceTitle(source: RAGSourceResponse): string {
  return source.document_title || source.filename || `Источник ${source.rank}`
}

export function SourceList({ sources }: { sources: RAGSourceResponse[] }) {
  if (sources.length === 0) {
    return <p className="muted">Источники не вернулись в ответе.</p>
  }

  return (
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
        </article>
      ))}
    </div>
  )
}
