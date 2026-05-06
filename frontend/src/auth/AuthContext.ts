import { createContext } from 'react'
import type { LoginRequest, RegisterRequest, UserProfileResponse } from '../api/types'

export type AuthStatus = 'bootstrapping' | 'authenticated' | 'anonymous'

export type AuthContextValue = {
  status: AuthStatus
  user: UserProfileResponse | null
  loginUser: (payload: LoginRequest) => Promise<void>
  registerUser: (payload: RegisterRequest) => Promise<void>
  logout: () => void
}

export const AuthContext = createContext<AuthContextValue | undefined>(undefined)
