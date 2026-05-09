import { authStoragePolicy } from '../auth/tokenStorage'
import { useAuth } from '../auth/useAuth'
import { PageShell } from '../components/PageShell'
import { Alert } from '../components/ui/Alert'
import { Button } from '../components/ui/Button'

function displayName(firstName: string | null | undefined, lastName: string | null | undefined): string {
  return [firstName, lastName].filter(Boolean).join(' ') || 'Врач CADENCE-MD'
}

export function ProfilePage() {
  const { user, logout } = useAuth()

  return (
    <PageShell>
      <section className="page-header">
        <span className="eyebrow">Личный кабинет</span>
        <h1>{displayName(user?.first_name, user?.last_name)}</h1>
        <p>Профиль используется для авторизации запросов и изоляции истории RAG-запросов.</p>
      </section>

      <div className="profile-grid">
        <article className="panel-card">
          <h2>Данные пользователя</h2>
          <dl className="profile-list">
            <div>
              <dt>Email</dt>
              <dd>{user?.email}</dd>
            </div>
            <div>
              <dt>User ID</dt>
              <dd>{user?.id}</dd>
            </div>
            <div>
              <dt>Имя</dt>
              <dd>{user?.first_name || 'Не указано'}</dd>
            </div>
            <div>
              <dt>Фамилия</dt>
              <dd>{user?.last_name || 'Не указана'}</dd>
            </div>
          </dl>
          <Button variant="danger" onClick={logout}>Выйти из аккаунта</Button>
        </article>

        <article className="panel-card">
          <h2>Политика сессии</h2>
          <Alert tone="info">
            Access token хранится в {authStoragePolicy.storage}; refresh token в MVP не используется.
            При `401` приложение очищает состояние и переводит пользователя на экран входа.
          </Alert>
          <p className="muted">{authStoragePolicy.rationale}</p>
        </article>
      </div>
    </PageShell>
  )
}
