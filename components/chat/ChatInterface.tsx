'use client'

import { useChat } from 'ai/react'
import { useEffect, useRef, type FormEvent } from 'react'
import { Send, Square, Plus, Zap, AlertCircle } from 'lucide-react'
import { useRAG } from '@/contexts/RAGContext'
import { useSession } from '@/contexts/SessionContext'
import MessageItem from './MessageItem'
import { cn } from '@/lib/utils'

export default function ChatInterface() {
  const { ragEnabled, kbId } = useRAG()
  const {
    currentSessionId,
    setCurrentSessionId,
    pendingMessages,
    clearPendingMessages,
    reloadSessions,
    switchSession,
  } = useSession()

  const messagesEndRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const prevDataLen = useRef(0)
  const urlSyncReady = useRef(false)

  const {
    messages,
    input,
    handleInputChange,
    handleSubmit: originalHandleSubmit,
    isLoading,
    stop,
    error,
    setMessages,
    data,
  } = useChat({ api: `${process.env.NEXT_PUBLIC_BACKEND_URL ?? 'http://localhost:8000'}/api/chat` })
  // Restore session from URL on mount
  useEffect(() => {
    const params = new URLSearchParams(window.location.search)
    const sessionId = params.get('session')
    if (sessionId) {
      switchSession(sessionId)
    } else if (currentSessionId) {
      switchSession(currentSessionId)
      const url = new URL(window.location.href)
      url.searchParams.set('session', currentSessionId)
      window.history.replaceState({}, '', url.toString())
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Sync currentSessionId → URL query param
  useEffect(() => {
    if (!urlSyncReady.current) {
      urlSyncReady.current = true
      return
    }
    const url = new URL(window.location.href)
    if (currentSessionId) {
      url.searchParams.set('session', currentSessionId)
    } else {
      url.searchParams.delete('session')
    }
    window.history.replaceState({}, '', url.toString())
  }, [currentSessionId])

  // Load messages when switching sessions
  useEffect(() => {
    if (pendingMessages === null) return
    setMessages(
      pendingMessages.map(m => ({ id: m.id, role: m.role, content: m.content }))
    )
    prevDataLen.current = 0
    clearPendingMessages()
  }, [pendingMessages, setMessages, clearPendingMessages])

  // Extract sessionId from data stream
  useEffect(() => {
    if (!data || data.length <= prevDataLen.current) return
    const newItems = data.slice(prevDataLen.current)
    prevDataLen.current = data.length
    for (const item of newItems) {
      const d = item as { type?: string; sessionId?: string }
      if (d?.type === 'session' && d.sessionId) {
        setCurrentSessionId(d.sessionId)
        reloadSessions()
      }
    }
  }, [data, setCurrentSessionId, reloadSessions])

  // Auto-scroll
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isLoading])

  // Listen for Escape key globally to stop generation
  useEffect(() => {
    const handleGlobalKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isLoading) {
        stop()
      }
    }
    window.addEventListener('keydown', handleGlobalKeyDown)
    return () => {
      window.removeEventListener('keydown', handleGlobalKeyDown)
    }
  }, [isLoading, stop])

  // Auto-resize textarea
  useEffect(() => {
    const el = textareaRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`
  }, [input])

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault()
    if (!input.trim() || isLoading) return
    originalHandleSubmit(e, { body: { ragEnabled, sessionId: currentSessionId, kbId } })
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault()
      handleSubmit(e as unknown as FormEvent)
    }
  }

  const clearHistory = () => {
    setMessages([])
    setCurrentSessionId(null)
    prevDataLen.current = 0
  }

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <header className="flex items-center justify-between px-6 py-3.5 bg-white border-b border-gray-200 shrink-0">
        <div className="flex items-center gap-2">
          <h1 className="font-semibold text-gray-900">AI 聊天</h1>
          {/* {ragEnabled && (
            <span className="flex items-center gap-1 text-xs bg-amber-100 text-amber-700 px-2 py-0.5 rounded-full font-medium">
              <Zap className="w-3 h-3" />
              RAG 已开启
            </span>
          )} */}
        </div>

        <button
          onClick={clearHistory}
          className="flex items-center gap-1.5 text-xs text-gray-500 hover:text-blue-500 transition-colors"
        >
          <Plus className="w-3.5 h-3.5" />
          新对话
        </button>
      </header>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto py-4">
        {messages.length === 0 ? (
          <EmptyState ragEnabled={ragEnabled} />
        ) : (
          messages.map((msg, idx) => {
            const isLastAssistant =
              msg.role === 'assistant' &&
              idx === messages.length - 1

            return (
              <MessageItem
                key={msg.id}
                role={msg.role as 'user' | 'assistant'}
                content={msg.content}
                isStreaming={isLastAssistant && isLoading}
              />
            )
          })
        )}

        {isLoading && messages[messages.length - 1]?.role !== 'assistant' && (
          <div className="flex gap-3 px-4 py-3">
            <div className="w-8 h-8 rounded-full bg-gray-100 border border-gray-200 flex items-center justify-center shrink-0">
              <div className="w-3 h-3 border-2 border-gray-400 border-t-transparent rounded-full animate-spin" />
            </div>
            <div className="bg-white border border-gray-200 rounded-2xl rounded-tl-sm shadow-sm px-4 py-3">
              <div className="flex gap-1">
                {[0, 1, 2].map(i => (
                  <span
                    key={i}
                    className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce"
                    style={{ animationDelay: `${i * 0.15}s` }}
                  />
                ))}
              </div>
            </div>
          </div>
        )}

        {error && (
          <div className="mx-4 my-2 flex items-start gap-2 bg-red-50 border border-red-200 rounded-lg px-4 py-3 text-sm text-red-700">
            <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
            <span>请求失败：{error.message}。请检查 API Key 配置。</span>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="shrink-0 border-t border-gray-200 bg-white px-4 py-3">
        <form onSubmit={handleSubmit} className="flex gap-2 items-center">
          <div className="flex-1 relative flex">
            <textarea
              ref={textareaRef}
              value={input}
              onChange={handleInputChange}
              onKeyDown={handleKeyDown}
              placeholder={ragEnabled ? '输入问题（将参考知识库）…' : '输入消息…'}
              rows={1}
              className={cn(
                'w-full resize-none rounded-xl border border-gray-200 bg-gray-50 px-4 py-3 pr-12 no-scrollbar',
                'text-sm placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent',
                'transition-colors'
              )}
            />
          </div>
          {isLoading ? (
            <button
              type="button"
              onClick={stop}
              aria-label="停止生成"
              className="shrink-0 w-10 h-10 rounded-xl flex items-center justify-center bg-red-500 text-white hover:bg-red-600 transition-colors"
            >
              <Square className="w-4 h-4 fill-current" />
            </button>
          ) : (
            <button
              type="submit"
              disabled={!input.trim()}
              aria-label="发送消息"
              className={cn(
                'shrink-0 w-10 h-10 rounded-xl flex items-center justify-center transition-colors',
                input.trim()
                  ? 'bg-blue-600 text-white hover:bg-blue-700'
                  : 'bg-gray-200 text-gray-400 cursor-not-allowed'
              )}
            >
              <Send className="w-4 h-4" />
            </button>
          )}
        </form>
        <p className="text-xs text-gray-400 mt-1.5 ml-1">
          Enter 发送 · Shift+Enter 换行
        </p>
      </div>
    </div>
  )
}

function EmptyState({ ragEnabled }: { ragEnabled: boolean }) {
  const tips = ragEnabled
    ? ['知识库中有什么内容？', '请总结相关文档', '查找关于…的信息']
    : ['你好，介绍一下自己', '帮我写一段代码', '解释一个概念']

  return (
    <div className="flex flex-col items-center justify-center h-full gap-6 px-4 text-center">
      <div>
        <div className="w-16 h-16 bg-blue-50 rounded-2xl flex items-center justify-center mx-auto mb-3">
          <Zap className={cn('w-8 h-8', ragEnabled ? 'text-amber-500' : 'text-blue-400')} />
        </div>
        <h2 className="text-lg font-semibold text-gray-800 mb-1">
          {ragEnabled ? 'RAG 增强模式' : '开始对话'}
        </h2>
        <p className="text-sm text-gray-500 max-w-xs">
          {ragEnabled
            ? '已连接知识库，AI 将结合知识库内容回答您的问题'
            : '在下方输入消息，与 AiBot 开始对话'}
        </p>
      </div>
      <div className="flex flex-wrap gap-2 justify-center">
        {tips.map(tip => (
          <span
            key={tip}
            className="text-xs bg-white border border-gray-200 text-gray-600 px-3 py-1.5 rounded-full"
          >
            {tip}
          </span>
        ))}
      </div>
    </div>
  )
}
