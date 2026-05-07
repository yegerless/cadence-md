import type { RAGRequestStatus } from '../api/types'

const labels: Record<RAGRequestStatus, string> = {
  queued: 'В очереди',
  running: 'Выполняется',
  awaiting_clarification: 'Требуется уточнение',
  succeeded: 'Готово',
  failed: 'Ошибка',
  cancelled: 'Отменено',
}

export function StatusBadge({ status }: { status: RAGRequestStatus }) {
  return <span className={`status-badge status-${status}`}>{labels[status]}</span>
}
