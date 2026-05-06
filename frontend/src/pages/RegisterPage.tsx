import type { FormEvent } from 'react'
import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { ApiError } from '../api/client'
import { useAuth } from '../auth/useAuth'
import { Alert } from '../components/ui/Alert'
import { Button } from '../components/ui/Button'
import { Input } from '../components/ui/Input'

function registerErrorMessage(error: unknown): string {
  if (error instanceof ApiError) {
    if (error.status === 409) {
      return 'Пользователь с таким email уже зарегистрирован.'
    }
    if (error.status === 429) {
      return 'Слишком много регистраций с этого адреса. Попробуйте позже.'
    }
    return error.response?.message ?? 'Не удалось создать аккаунт.'
  }
  return 'Сервис временно недоступен. Проверьте соединение.'
}

export function RegisterPage() {
  const navigate = useNavigate()
  const { registerUser } = useAuth()
  const [firstName, setFirstName] = useState('')
  const [lastName, setLastName] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [isSubmitting, setIsSubmitting] = useState(false)

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setError(null)
    setIsSubmitting(true)

    try {
      await registerUser({
        email,
        password,
        first_name: firstName || null,
        last_name: lastName || null,
      })
      navigate('/chat', { replace: true })
    } catch (authError) {
      setError(registerErrorMessage(authError))
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <section className="auth-layout">
      <div className="auth-hero">
        <span className="eyebrow">CADENCE-MD</span>
        <h1>Единое рабочее место для доказательных ответов</h1>
        <p>
          Зарегистрируйтесь, чтобы запускать асинхронные RAG-запросы и видеть источники ответа.
        </p>
      </div>
      <form className="auth-card" onSubmit={handleSubmit}>
        <div>
          <span className="eyebrow">Регистрация</span>
          <h2>Создать аккаунт</h2>
        </div>
        {error ? <Alert tone="error">{error}</Alert> : null}
        <div className="two-column">
          <Input
            label="Имя"
            name="first_name"
            value={firstName}
            onChange={(event) => setFirstName(event.target.value)}
          />
          <Input
            label="Фамилия"
            name="last_name"
            value={lastName}
            onChange={(event) => setLastName(event.target.value)}
          />
        </div>
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
          autoComplete="new-password"
          minLength={8}
          maxLength={128}
          required
          hint="Минимум 8 символов."
          value={password}
          onChange={(event) => setPassword(event.target.value)}
        />
        <Button type="submit" disabled={isSubmitting}>
          {isSubmitting ? 'Создаем...' : 'Создать аккаунт'}
        </Button>
        <p className="auth-switch">
          Уже есть аккаунт? <Link to="/login">Войти</Link>
        </p>
      </form>
    </section>
  )
}
