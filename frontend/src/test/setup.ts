import '@testing-library/jest-dom/vitest'
import { cleanup } from '@testing-library/react'
import { afterEach, vi } from 'vitest'

afterEach(() => {
  cleanup()
  window.sessionStorage.clear()
  vi.restoreAllMocks()
})

if (!crypto.randomUUID) {
  Object.defineProperty(crypto, 'randomUUID', {
    value: () => '00000000-0000-4000-8000-000000000000',
  })
}
