const ACCESS_TOKEN_KEY = 'cadence_md_access_token'

export function getAccessToken(): string | null {
  return window.sessionStorage.getItem(ACCESS_TOKEN_KEY)
}

export function setAccessToken(token: string): void {
  window.sessionStorage.setItem(ACCESS_TOKEN_KEY, token)
}

export function clearAccessToken(): void {
  window.sessionStorage.removeItem(ACCESS_TOKEN_KEY)
}

export const authStoragePolicy = {
  storage: 'sessionStorage',
  refreshToken: false,
  rationale:
    'The MVP stores only the short-lived access token in sessionStorage. This preserves refresh UX within a browser session, avoids automatic cookie submission and CSRF exposure, and clears the token when the browser session ends. XSS remains the primary risk, so the app never logs tokens or renders untrusted HTML.',
} as const
