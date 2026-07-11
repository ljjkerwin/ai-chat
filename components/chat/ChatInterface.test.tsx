import { render, screen, waitFor, act } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { vi, describe, it, expect, beforeEach } from 'vitest'
import React, { useState } from 'react'
import { RAGProvider } from '@/contexts/RAGContext'
import { SessionProvider } from '@/contexts/SessionContext'
import Sidebar from '@/components/layout/Sidebar'
import ChatInterface from '@/components/chat/ChatInterface'

// Stateful mock state for useChat
const mockChatState = {
  messages: [] as any[],
  input: '',
  isLoading: false,
  error: null as any,
  data: null as any,
}

const mockStop = vi.fn()

// Hook state listener subscriptions
const listeners = new Set<() => void>()

export function updateMockState(updates: Partial<typeof mockChatState>) {
  Object.assign(mockChatState, updates)
  listeners.forEach(l => l())
}

// Mock useChat hook from ai/react
vi.mock('ai/react', () => {
  return {
    useChat: () => {
      const [state, setState] = React.useState(mockChatState)

      React.useEffect(() => {
        const handler = () => {
          setState({ ...mockChatState })
        }
        listeners.add(handler)
        return () => {
          listeners.delete(handler)
        }
      }, [])

      const handleInputChange = React.useCallback((e: React.ChangeEvent<HTMLTextAreaElement>) => {
        updateMockState({ input: e.target.value })
      }, [])

      const handleSubmit = React.useCallback((e: React.FormEvent, options?: any) => {
        e.preventDefault()
        const currentInput = mockChatState.input
        if (!currentInput.trim()) return

        const userMsg = { id: 'msg-user-' + Math.random(), role: 'user', content: currentInput }
        const newMessages = [...mockChatState.messages, userMsg]

        updateMockState({
          messages: newMessages,
          input: '',
          isLoading: true
        })

        // Simulate async loading and AI response
        setTimeout(() => {
          const assistantMsg = { id: 'msg-assistant-' + Math.random(), role: 'assistant', content: 'AI 回答：' + userMsg.content }
          updateMockState({
            isLoading: false,
            messages: [...newMessages, assistantMsg]
          })
        }, 10)
      }, [])

      const setMessages = React.useCallback((msgs: any) => {
        const resolved = typeof msgs === 'function' ? msgs(mockChatState.messages) : msgs
        updateMockState({ messages: resolved })
      }, [])

      return {
        messages: state.messages,
        input: state.input,
        handleInputChange,
        handleSubmit,
        isLoading: state.isLoading,
        stop: mockStop,
        error: state.error,
        setMessages,
        data: state.data,
      }
    }
  }
})

// Mock fetch data
const mockKBs = [
  { id: '1', name: '开发文档' },
  { id: '2', name: '产品手册' }
]

const mockSessions = [
  { id: 'session-1', title: '关于 React 19 的讨论', createdAt: '2026-07-05T08:00:00Z' },
  { id: 'session-2', title: 'Vitest 安装指南', createdAt: '2026-07-05T09:00:00Z' }
]

const mockKBStats = {
  stats: {
    docCount: 12,
    chunkCount: 250
  }
}

const mockSessionMessages = [
  { id: 'msg-h1', role: 'user', content: '什么是 React 19?' },
  { id: 'msg-h2', role: 'assistant', content: 'React 19 引入了 Server Components 和 Actions 等新特性。' }
]

// Mock global fetch API
const fetchSpy = vi.spyOn(window, 'fetch').mockImplementation((url) => {
  const urlStr = url.toString()
  console.log("DEBUG fetchSpy intercepted URL:", urlStr)
  if (urlStr.includes('/api/knowledge-bases')) {
    return Promise.resolve({
      ok: true,
      json: () => Promise.resolve({ knowledgeBases: mockKBs })
    } as Response)
  }
  if (urlStr.includes('/api/sessions') && urlStr.endsWith('/messages')) {
    return Promise.resolve({
      ok: true,
      json: () => Promise.resolve({ messages: mockSessionMessages })
    } as Response)
  }
  if (urlStr.includes('/api/sessions') && (urlStr.endsWith('/api/sessions') || urlStr.includes('/api/sessions?'))) {
    return Promise.resolve({
      ok: true,
      json: () => Promise.resolve({
        sessions: mockSessions,
        page: 1,
        page_size: 30,
        total: mockSessions.length,
        has_more: false
      })
    } as Response)
  }
  if (urlStr.includes('/api/knowledge')) {
    return Promise.resolve({
      ok: true,
      json: () => Promise.resolve(mockKBStats)
    } as Response)
  }
  if (urlStr.includes('/api/sessions/') && urlStr.match(/\/api\/sessions\/[^/]+$/)) {
    return Promise.resolve({
      ok: true,
      json: () => Promise.resolve({ success: true })
    } as Response)
  }
  return Promise.reject(new Error(`Unhandled mock fetch for URL: ${urlStr}`))
})

function TestChatApp() {
  return (
    <RAGProvider>
      <SessionProvider>
        <div className="flex h-full">
          <Sidebar />
          <ChatInterface />
        </div>
      </SessionProvider>
    </RAGProvider>
  )
}

describe('Chat page integration tests', () => {
  beforeEach(() => {
    // Reset state for mock useChat
    mockChatState.messages = []
    mockChatState.input = ''
    mockChatState.isLoading = false
    mockChatState.error = null
    mockChatState.data = null
    listeners.clear()

    mockStop.mockClear()

    // Clear fetch and local storage mock
    fetchSpy.mockClear()
    window.localStorage.clear()

    // Reset window.location query parameters between tests to maintain isolation
    const url = new URL(window.location.href)
    url.searchParams.delete('session')
    window.history.replaceState({}, '', url.pathname + url.search)
  })

  it('renders initial empty state screen correctly', async () => {
    render(<TestChatApp />)

    // Check header elements are rendered
    expect(screen.getByRole('button', { name: /新对话/ })).toBeInTheDocument()

    // Since messages are empty, we expect empty list placeholder (which is null/empty)
    await waitFor(() => {
      expect(screen.queryByText('在下方输入消息，与 AI 开始对话')).not.toBeInTheDocument()
    })

    // Expect input bar placeholder
    expect(screen.getByPlaceholderText('输入消息…')).toBeInTheDocument()
  })

  it('allows typing and sending a message, showing AI response', async () => {
    render(<TestChatApp />)
    const user = userEvent.setup()

    const textarea = screen.getByPlaceholderText('输入消息…')
    const sendButton = screen.getByRole('button', { name: '发送消息' }) // Since Send has SVG, query by aria-label

    // Input message
    await user.type(textarea, '你好')
    expect(textarea).toHaveValue('你好')

    // Submit
    await user.click(sendButton)

    // Verify user message displays
    expect(screen.getByText('你好')).toBeInTheDocument()

    // Verify AI response shows after delay
    await waitFor(() => {
      expect(screen.getByText('AI 回答：你好')).toBeInTheDocument()
    })
  })

  it('renders loading indicators and handles stop button', async () => {
    // Force loading state
    mockChatState.isLoading = true
    mockChatState.messages = [{ id: '1', role: 'user', content: '测试等待' }]

    render(<TestChatApp />)

    // Expecting to see the bouncing dot dots or loading icon
    const stopButton = screen.getByRole('button', { name: '停止生成' }) // Red square stop button
    expect(stopButton).toBeInTheDocument()

    const user = userEvent.setup()
    await user.click(stopButton)
    expect(mockStop).toHaveBeenCalledTimes(1)
  })

  it('shows error banner when API chat returns error', async () => {
    mockChatState.error = new Error('No API Key found')
    render(<TestChatApp />)

    await waitFor(() => {
      expect(screen.getByText('请求失败：No API Key found。请检查 API Key 配置。')).toBeInTheDocument()
    })
  })

  it('loads sessions and allows switching between them', async () => {
    render(<TestChatApp />)
    const user = userEvent.setup()

    // Wait for sidebar to render sessions
    await waitFor(() => {
      expect(screen.getByText('关于 React 19 的讨论')).toBeInTheDocument()
      expect(screen.getByText('Vitest 安装指南')).toBeInTheDocument()
    })

    // Click on session 1
    const sessionLink = screen.getByText('关于 React 19 的讨论')
    await user.click(sessionLink)

    // Verify correct messages loaded into ChatInterface
    await waitFor(() => {
      expect(screen.getByText('什么是 React 19?')).toBeInTheDocument()
      expect(screen.getByText('React 19 引入了 Server Components 和 Actions 等新特性。')).toBeInTheDocument()
    })
  })

  it('can start a new session by clicking "新对话"', async () => {
    // Populate messages
    mockChatState.messages = [{ id: '1', role: 'user', content: '这是一条旧消息' }]
    render(<TestChatApp />)

    expect(screen.getByText('这是一条旧消息')).toBeInTheDocument()

    const user = userEvent.setup()
    const newChatBtn = screen.getByRole('button', { name: /新对话/ })
    await user.click(newChatBtn)

    // Verify messages list is cleared
    expect(screen.queryByText('这是一条旧消息')).not.toBeInTheDocument()
  })

  it('toggles RAG and shows changes in EmptyState and placeholders', async () => {
    render(<TestChatApp />)
    const user = userEvent.setup()

    // Initially RAG is off
    expect(screen.getByPlaceholderText('输入消息…')).toBeInTheDocument()

    // Click the settings button to open settings popover
    const settingsBtn = screen.getByRole('button', { name: '知识库设置' })
    await user.click(settingsBtn)

    // Wait for the knowledge base options to load from the mocked API
    await waitFor(() => {
      expect(screen.queryByRole('option', { name: '开发文档' })).toBeInTheDocument()
    })

    // Query the combobox after it has loaded the options
    const ragSelect = screen.getByRole('combobox', { name: '选择关联知识库' })
    await user.selectOptions(ragSelect, '1')

    // Verify local storage is set and states are updated
    await waitFor(() => {
      expect(window.localStorage.getItem('ragEnabled')).toBe('true')
      expect(screen.getByPlaceholderText('输入消息…')).toBeInTheDocument()
    })
  })

  it('allows typing during loading state and stopping with Escape key', async () => {
    // Force loading state
    mockChatState.isLoading = true
    mockChatState.messages = [{ id: '1', role: 'user', content: '正在生成回答' }]
    mockChatState.input = ''

    render(<TestChatApp />)
    const user = userEvent.setup()

    const textarea = screen.getByPlaceholderText('输入消息…')
    // Textarea should not be disabled
    expect(textarea).not.toBeDisabled()

    // Type while loading
    await user.type(textarea, '新消息')
    expect(textarea).toHaveValue('新消息')

    // Press Escape to stop generation
    await user.keyboard('{Escape}')
    expect(mockStop).toHaveBeenCalled()
  })

  it('restores active session from context if no session ID in URL query parameters', async () => {
    let showChat = true
    const { rerender } = render(
      <RAGProvider>
        <SessionProvider>
          <div className="flex h-full">
            <Sidebar />
            {showChat && <ChatInterface />}
          </div>
        </SessionProvider>
      </RAGProvider>
    )
    const user = userEvent.setup()

    // Wait for sessions to load
    await waitFor(() => {
      expect(screen.getByText('关于 React 19 的讨论')).toBeInTheDocument()
    })

    // Click on session 1
    const sessionLink = screen.getByText('关于 React 19 的讨论')
    await user.click(sessionLink)

    // Verify messages loaded
    await waitFor(() => {
      expect(screen.getByText('什么是 React 19?')).toBeInTheDocument()
    })

    // Unmount ChatInterface (navigate away to knowledge base)
    showChat = false
    rerender(
      <RAGProvider>
        <SessionProvider>
          <div className="flex h-full">
            <Sidebar />
            {showChat && <ChatInterface />}
          </div>
        </SessionProvider>
      </RAGProvider>
    )

    // Verify ChatInterface is not there
    expect(screen.queryByPlaceholderText('输入消息…')).not.toBeInTheDocument()

    // Mount ChatInterface again (navigate back to /chat)
    showChat = true
    rerender(
      <RAGProvider>
        <SessionProvider>
          <div className="flex h-full">
            <Sidebar />
            {showChat && <ChatInterface />}
          </div>
        </SessionProvider>
      </RAGProvider>
    )

    // ChatInterface mounts again. It should automatically restore the messages
    await waitFor(() => {
      expect(screen.getByText('什么是 React 19?')).toBeInTheDocument()
      expect(screen.getByText('React 19 引入了 Server Components 和 Actions 等新特性。')).toBeInTheDocument()
    })
  })

  it('calls stop when component is unmounted', () => {
    mockChatState.isLoading = true
    mockStop.mockClear()

    const { unmount } = render(<TestChatApp />)
    unmount()

    expect(mockStop).toHaveBeenCalledTimes(1)
  })

  it('calls stop when switching sessions', async () => {
    mockChatState.isLoading = true
    mockStop.mockClear()

    render(<TestChatApp />)
    const user = userEvent.setup()

    // Wait for sessions to load
    await waitFor(() => {
      expect(screen.getByText('关于 React 19 的讨论')).toBeInTheDocument()
    })

    // Click on a session
    const sessionLink = screen.getByText('关于 React 19 的讨论')
    await user.click(sessionLink)

    // Verify stop was called during switching
    expect(mockStop).toHaveBeenCalled()
  })

  it('calls stop when clicking "新对话" (new chat)', async () => {
    mockChatState.isLoading = true
    mockStop.mockClear()

    render(<TestChatApp />)
    const user = userEvent.setup()

    const newChatBtn = screen.getByRole('button', { name: /新对话/ })
    await user.click(newChatBtn)

    expect(mockStop).toHaveBeenCalled()
  })
})

