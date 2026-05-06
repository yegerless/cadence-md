import type { FormEvent } from 'react'
import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { ApiError } from '../api/client'
import { useAuth } from '../auth/useAuth'
import { Alert } from '../components/ui/Alert'
import { Button } from '../components/ui/Button'
import { Input } from '../components/ui/Input'

function authErrorMessage(error: unknown): string {
  if (error instanceof ApiError) {
    if (error.status === 401) {
      return 'Проверьте email и пароль.'
    }
    if (error.status === 429) {
      return 'Слишком много попыток входа. Попробуйте позже.'
    }
    return error.response?.message ?? 'Не удалось войти.'
  }
  return 'Сервис временно недоступен. Проверьте соединение.'
}

export function LoginPage() {
  const navigate = useNavigate()
  const { loginUser } = useAuth()
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [isSubmitting, setIsSubmitting] = useState(false)

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setError(null)
    setIsSubmitting(true)

    try {
      await loginUser({ email, password })
      navigate('/chat', { replace: true })
    } catch (authError) {
      setError(authErrorMessage(authError))
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <section className="auth-layout">
      <div className="auth-hero">
        <span className="eyebrow">Evidence-first AI</span>
        <h1>Клинический RAG-ассистент для врачей</h1>
        <p>
          Авторизуйтесь, чтобы задавать вопросы по клиническим рекомендациям и получать ответы с
          проверяемыми источниками.
        </p>
      </div>
      <form className="auth-card" onSubmit={handleSubmit}>
        <div>
          <span className="eyebrow">Вход</span>
          <h2>Добро пожаловать</h2>
        </div>
        {error ? <Alert tone="error">{error}</Alert> : null}
        <Input
          label="Email"
          name="email"
          type="email"
          autoComplete="email"
          required
          value={email}
          onChange={(event) => setEmail(event.target.value)}
        />
        <Input
          label="Пароль"
          name="password"
          type="password"
          autoComplete="current-password"
          required
          value={password}
          onChange={(event) => setPassword(event.target.value)}
        />
        <Button type="submit" disabled={isSubmitting}>
          {isSubmitting ? 'Входим...' : 'Войти'}
        </Button>
        <p className="auth-switch">
          Нет аккаунта? <Link to="/register">Зарегистрироваться</Link>
        </p>
      </form>
    </section>
  )
}
