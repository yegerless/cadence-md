import { useQueryClient } from '@tanstack/react-query'
import {
  type ReactNode,
  useCallback,
  useEffect,
  useMemo,
  useState,
} from 'react'
import { useNavigate } from 'react-router-dom'
import { getCurrentUser, login, register } from '../api/cadenceApi'
import { setUnauthorizedHandler } from '../api/client'
import type { LoginRequest, RegisterRequest, UserProfileResponse } from '../api/types'
import { clearAccessToken, getAccessToken, setAccessToken } from './tokenStorage'
import { AuthContext, type AuthContextValue, type AuthStatus } from './AuthContext'

export function AuthProvider({ children }: { children: ReactNode }) {
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const [status, setStatus] = useState<AuthStatus>('bootstrapping')
  const [user, setUser] = useState<UserProfileResponse | null>(null)

  const clearSession = useCallback(() => {
    clearAccessToken()
    setUser(null)
    setStatus('anonymous')
    queryClient.clear()
  }, [queryClient])

  const logout = useCallback(() => {
    clearSession()
    navigate('/login', { replace: true })
  }, [clearSession, navigate])

  useEffect(() => {
    setUnauthorizedHandler(() => {
      clearSession()
      navigate('/login', { replace: true })
    })

    return () => setUnauthorizedHandler(undefined)
  }, [clearSession, navigate])

  useEffect(() => {
    let active = true

    async function bootstrap() {
      if (!getAccessToken()) {
        setStatus('anonymous')
        return
      }

      try {
        const profile = await getCurrentUser()
        if (active) {
          setUser(profile)
          setStatus('authenticated')
        }
      } catch {
        if (active) {
          clearSession()
        }
      }
    }

    void bootstrap()

    return () => {
      active = false
    }
  }, [clearSession])

  const authenticateWithToken = useCallback(async (token: string) => {
    setAccessToken(token)
    const profile = await getCurrentUser()
    setUser(profile)
    setStatus('authenticated')
    queryClient.invalidateQueries()
  }, [queryClient])

  const loginUser = useCallback(async (payload: LoginRequest) => {
    const token = await login(payload)
    await authenticateWithToken(token.access_token)
  }, [authenticateWithToken])

  const registerUser = useCallback(async (payload: RegisterRequest) => {
    const token = await register(payload)
    await authenticateWithToken(token.access_token)
  }, [authenticateWithToken])

  const value = useMemo<AuthContextValue>(
    () => ({ status, user, loginUser, registerUser, logout }),
    [loginUser, logout, registerUser, status, user],
  )

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}
