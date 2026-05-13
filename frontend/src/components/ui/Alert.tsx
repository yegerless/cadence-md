import type { ReactNode } from 'react'

type AlertTone = 'info' | 'error' | 'success' | 'warning'

export function Alert({ tone = 'info', children }: { tone?: AlertTone; children: ReactNode }) {
  return (
    <div className={`alert alert-${tone}`} role={tone === 'error' ? 'alert' : 'status'}>
      {children}
    </div>
  )
}
