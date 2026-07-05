import '@testing-library/jest-dom'
import { vi } from 'vitest'

// Mock scrollIntoView which is not implemented in jsdom
window.HTMLElement.prototype.scrollIntoView = vi.fn()

// Mock next/navigation
vi.mock('next/navigation', () => {
  return {
    usePathname: () => '/chat',
    useRouter: () => ({
      push: vi.fn(),
      replace: vi.fn(),
      prefetch: vi.fn(),
      back: vi.fn(),
    }),
  }
})
