import type { ReactNode } from 'react'
import { NavLink } from 'react-router-dom'
import { useAuth } from '../auth/useAuth'
import { Button } from './ui/Button'

export function PageShell({ children }: { children: ReactNode }) {
  const { user, logout } = useAuth()

  return (
    <div className="app-shell">
      <aside className="sidebar" aria-label="Основная навигация">
        <NavLink className="brand" to="/chat">
          <span className="brand-mark">CM</span>
          <span>
            <strong>CADENCE-MD</strong>
            <small>Clinical RAG assistant</small>
          </span>
        </NavLink>
        <nav className="nav-links">
          <NavLink to="/chat">Чат</NavLink>
          <NavLink to="/profile">Профиль</NavLink>
        </nav>
        <div className="sidebar-footer">
          <span>{user?.email}</span>
          <Button variant="ghost" onClick={logout}>Выйти</Button>
        </div>
      </aside>
      <main className="main-panel">{children}</main>
    </div>
  )
}
