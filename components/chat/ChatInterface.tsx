'use client'

import { useChat } from 'ai/react'
import { useEffect, useRef, useState, type FormEvent } from 'react'
import { Send, Square, Plus, Zap, AlertCircle, Menu, Settings, Database } from 'lucide-react'
import Link from 'next/link'



import { useRAG } from '@/contexts/RAGContext'
import { useSession } from '@/contexts/SessionContext'
import agents from '@/data/agents.json'
import MessageItem from './MessageItem'
import { cn } from '@/lib/utils'

export default function ChatInterface() {
  const { ragEnabled, setRagEnabled, kbId, setKbId, agentType, setAgentType, knowledgeBases } = useRAG()
  const {
    currentSessionId,
    setCurrentSessionId,
    pendingMessages,
    clearPendingMessages,
    reloadSessions,
    switchSession,
    setSidebarOpen,
  } = useSession()

  const messagesEndRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const prevDataLen = useRef(0)
  const urlSyncReady = useRef(false)
  const prevAgentType = useRef<string | null>(null)

  const [isSettingsOpen, setIsSettingsOpen] = useState(false)
  const settingsRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (settingsRef.current && !settingsRef.current.contains(event.target as Node)) {
        setIsSettingsOpen(false)
      }
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => {
      document.removeEventListener('mousedown', handleClickOutside)
    }
  }, [])

  const apiEndpoint = `${process.env.NEXT_PUBLIC_BACKEND_URL ?? 'http://localhost:8000'}/api/chat${agentType === '2' ? '/langgraph' : ''}`

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
  } = useChat({ api: apiEndpoint })
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

  // Sync currentSessionId and agentType → URL query param
  useEffect(() => {
    if (!urlSyncReady.current) {
      urlSyncReady.current = true
      return
    }
    const url = new URL(window.location.href)
    const currentAgent = url.searchParams.get('agent')

    if (agentType === '1') {
      if (currentAgent !== '1' && currentAgent !== 'default' && currentAgent !== 'react') {
        url.searchParams.delete('agent')
      }
    } else {
      url.searchParams.set('agent', agentType)
    }
    if (currentSessionId) {
      url.searchParams.set('session', currentSessionId)
    } else {
      url.searchParams.delete('session')
    }
    window.history.replaceState({}, '', url.toString())
  }, [currentSessionId, agentType])

  // Load messages when switching sessions
  useEffect(() => {
    if (pendingMessages === null) return
    stop() // Abort active streaming if we switch session
    setMessages(
      pendingMessages.map(m => ({ id: m.id, role: m.role, content: m.content }))
    )
    prevDataLen.current = 0
    clearPendingMessages()
  }, [pendingMessages, setMessages, clearPendingMessages, stop])

  // Clear chat history when switching agents
  useEffect(() => {
    if (prevAgentType.current !== null && prevAgentType.current !== agentType) {
      stop()
      setMessages([])
      prevDataLen.current = 0
    }
    prevAgentType.current = agentType
  }, [agentType, setMessages, stop])

  // Sync URL search params → agentType state
  useEffect(() => {
    const params = new URLSearchParams(window.location.search)
    const agentParam = params.get('agent')
    if (agentParam === '2' || agentParam === 'langgraph') {
      if (agentType !== '2') {
        setAgentType('2')
      }
    } else if (agentParam === '1' || agentParam === 'default' || agentParam === 'react') {
      if (agentType !== '1') {
        setAgentType('1')
      }
    } else {
      if (agentType !== '1') {
        setAgentType('1')
      }
    }
  }, [agentType, setAgentType])

  // Abort active generation on component unmount (e.g. navigating away)
  useEffect(() => {
    return () => {
      stop()
    }
  }, [stop])

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
    originalHandleSubmit(e, { body: { ragEnabled, sessionId: currentSessionId, kbId, agentType } })
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault()
      handleSubmit(e as unknown as FormEvent)
    }
  }

  const clearHistory = () => {
    stop() // Abort active generation when starting a new chat
    setMessages([])
    setCurrentSessionId(null)
    prevDataLen.current = 0
  }

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <header className="flex items-center justify-between px-4 md:px-6 py-3.5 bg-white border-b border-gray-200 shrink-0">
        <div className="flex items-center gap-3">
          <button
            onClick={() => setSidebarOpen(true)}
            className="md:hidden p-1 mr-1 text-gray-500 hover:text-gray-900 hover:bg-gray-100 rounded-lg transition-colors"
            aria-label="打开侧边栏"
          >
            <Menu className="w-5 h-5" />
          </button>
          <h1 className="text-sm font-semibold text-gray-800">
            {agents.find(a => a.id === agentType)?.name ?? '智能助理'}
          </h1>
          <div ref={settingsRef} className="relative flex items-center">
            <button
              onClick={() => setIsSettingsOpen(!isSettingsOpen)}
              className={cn(
                "p-1.5 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-lg transition-all flex items-center justify-center",
                isSettingsOpen && "text-blue-600 bg-blue-50"
              )}
              title="知识库设置"
              aria-label="知识库设置"
            >
              <Settings className="w-4 h-4" />
            </button>
            {isSettingsOpen && (
              <div className="absolute left-1/2 -translate-x-1/2 md:left-0 md:translate-x-0 mt-2 w-72 bg-white border border-gray-200 rounded-xl shadow-xl p-4 z-50 animate-in fade-in slide-in-from-top-2 duration-200 top-full">
                <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3">关联知识库</h3>
                <div className="space-y-3.5">
                  <div className="flex items-center gap-2">
                    <div className="flex-1 flex items-center gap-1.5 bg-gray-50 hover:bg-gray-100 border border-gray-200 rounded-lg px-2.5 py-1.5 transition-all min-w-0">
                      <Database className={cn("w-3.5 h-3.5 shrink-0 transition-colors", ragEnabled ? "text-amber-500" : "text-gray-400")} />
                      <select
                        value={ragEnabled ? (kbId || '1') : 'none'}
                        onChange={e => {
                          const val = e.target.value
                          if (val === 'none') {
                            setRagEnabled(false)
                            setKbId(null)
                          } else {
                            setRagEnabled(true)
                            setKbId(val)
                          }
                        }}
                        className="bg-transparent text-xs text-gray-700 font-semibold focus:outline-none cursor-pointer w-full truncate"
                        aria-label="选择关联知识库"
                      >
                        <option value="none">无 (禁用 RAG)</option>
                        {knowledgeBases.map(kb => (
                          <option key={kb.id} value={kb.id}>
                            {kb.name}
                          </option>
                        ))}
                      </select>
                    </div>
                  </div>
                  <div className="pt-2 border-t border-gray-100 flex items-center justify-between text-xs text-gray-400">
                    <span>
                      {ragEnabled ? "已启用 RAG 检索" : "RAG 已禁用"}
                    </span>
                    <Link
                      href="/knowledge"
                      onClick={() => setIsSettingsOpen(false)}
                      className="text-blue-600 hover:text-blue-700 transition-colors font-medium flex items-center gap-1"
                    >
                      管理知识库
                    </Link>
                  </div>
                </div>
              </div>
            )}
          </div>
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
          null
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
              placeholder={'输入消息…'}
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
        {/* <p className="text-xs text-gray-400 mt-1.5 ml-1">
          Enter 发送 · Shift+Enter 换行
        </p> */}
      </div>
    </div>
  )
}

function EmptyState({ ragEnabled }: { ragEnabled: boolean }) {
  const tips = ragEnabled
    ? ['知识库中有什么内容？', '请总结相关文档', '查找关于…的信息']
    : ['今天的天气怎样', '人民币兑韩元的汇率是多少']

  return (
    <div className="flex flex-col items-center justify-center h-full gap-6 px-4 text-center">
      <div>
        {/* <h2 className="text-lg font-semibold text-gray-800 mb-1">
          {ragEnabled ? 'RAG 增强模式' : '开始对话'}
        </h2> */}
        <p className="text-sm text-gray-500 max-w-xs">
          {ragEnabled
            ? '已连接知识库，AI 将结合知识库内容回答您的问题'
            : '在下方输入消息，与 AI 开始对话'}
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
