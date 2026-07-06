'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { MessageSquare, BookOpen, Zap, Database, Bot, Trash2 } from 'lucide-react'
import { useRAG } from '@/contexts/RAGContext'
import { useSession } from '@/contexts/SessionContext'
import { useEffect, useState } from 'react'
import { cn } from '@/lib/utils'
import type { KBStats } from '@/types'

export default function Sidebar() {
  const pathname = usePathname()
  const { ragEnabled, setRagEnabled, kbId, setKbId, knowledgeBases } = useRAG()
  const { sessions, reloadSessions, currentSessionId, switchSession } = useSession()
  const [stats, setStats] = useState<KBStats>({ docCount: 0, chunkCount: 0 })

  const isChat = pathname === '/chat'

  useEffect(() => {
    const activeKbId = kbId || '1'
    fetch(`/api/knowledge?kb_id=${activeKbId}`)
      .then(r => r.json())
      .then(d => setStats(d.stats))
      .catch(() => { })
  }, [pathname, kbId])

  useEffect(() => {
    if (isChat) reloadSessions()
  }, [isChat, reloadSessions])

  const handleDelete = async (e: React.MouseEvent, id: string) => {
    e.stopPropagation()
    await fetch(`/api/sessions/${id}`, { method: 'DELETE' })
    if (currentSessionId === id) switchSession(null)
    reloadSessions()
  }

  const navItems = [
    {
      href: currentSessionId ? `/chat?session=${currentSessionId}` : '/chat',
      icon: MessageSquare,
      label: 'AI 聊天',
      active: pathname === '/chat'
    },
    {
      href: '/knowledge',
      icon: BookOpen,
      label: '知识库',
      active: pathname === '/knowledge' || pathname.startsWith('/knowledge/')
    },
  ]

  return (
    <aside className="w-60 shrink-0 flex flex-col h-full bg-white border-r border-gray-200">
      {/* Logo */}
      <div className="flex items-center gap-2.5 px-4 py-5 border-b border-gray-100">
        <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center shrink-0">
          <Bot className="w-5 h-5 text-white" />
        </div>
        <div>
          <p className="font-semibold text-gray-900 text-sm leading-tight">AI Chat</p>
          <p className="text-xs text-gray-400">RAG 增强对话</p>
        </div>
      </div>

      {/* Navigation */}
      <nav className="p-3 space-y-1 border-b border-gray-100">
        {navItems.map(({ href, icon: Icon, label, active }) => (
          <Link
            key={href}
            href={href}
            className={cn(
              'flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors',
              active
                ? 'bg-blue-50 text-blue-700'
                : 'text-gray-600 hover:bg-gray-100 hover:text-gray-900'
            )}
          >
            <Icon className="w-4 h-4 shrink-0" />
            {label}
          </Link>
        ))}
      </nav>

      {/* Session history — only on /chat */}
      {isChat && (
        <div className="flex-1 flex flex-col min-h-0 p-3">

          <div className="flex-1 overflow-y-auto space-y-0.5">
            {sessions.length === 0 ? (
              <p className="text-xs text-gray-400 text-center pt-4">暂无历史对话</p>
            ) : (
              sessions.map(session => (
                <div
                  key={session.id}
                  onClick={() => switchSession(session.id)}
                  className={cn(
                    'group flex items-center gap-2 px-3 py-2 rounded-lg cursor-pointer transition-colors',
                    currentSessionId === session.id
                      ? 'bg-blue-50 text-blue-700'
                      : 'text-gray-600 hover:bg-gray-100'
                  )}
                >
                  <span className="flex-1 text-xs truncate">{session.title}</span>
                  <button
                    onClick={e => handleDelete(e, session.id)}
                    className="shrink-0 opacity-0 group-hover:opacity-100 text-gray-400 hover:text-red-500 transition-all"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                </div>
              ))
            )}
          </div>
        </div>
      )}

      {/* Spacer when not on chat */}
      {!isChat && <div className="flex-1" />}

      {/* RAG Toggle */}
      <div className="p-3 border-t border-gray-100">
        <div className="bg-gray-50 rounded-xl p-3 space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Zap className={cn('w-4 h-4', ragEnabled ? 'text-amber-500' : 'text-gray-400')} />
              <span className="text-sm font-medium text-gray-700">RAG 增强</span>
            </div>
            <button
              onClick={() => setRagEnabled(!ragEnabled)}
              className={cn(
                'relative inline-flex h-5 w-9 items-center rounded-full transition-colors focus:outline-none',
                ragEnabled ? 'bg-blue-600' : 'bg-gray-300'
              )}
              role="switch"
              aria-checked={ragEnabled}
              aria-label="RAG 增强"
            >
              <span
                className={cn(
                  'inline-block h-3.5 w-3.5 transform rounded-full bg-white shadow transition-transform',
                  ragEnabled ? 'translate-x-4.5' : 'translate-x-0.5'
                )}
              />
            </button>
          </div>

          {/* <p className="text-xs text-gray-500 leading-snug">
            {ragEnabled
              ? '已开启：回复将参考知识库内容'
              : '已关闭：纯模型回复，不使用知识库'}
          </p> */}

          {/* KB Dropdown Select */}
          {ragEnabled && knowledgeBases.length > 0 && (
            <div className="space-y-1 pt-1 border-t border-gray-200">
              <label className="text-[10px] font-semibold text-gray-400 block uppercase tracking-wider">切换当前知识库</label>
              <select
                value={kbId || '1'}
                onChange={e => setKbId(e.target.value)}
                className="w-full bg-white border border-gray-200 rounded-lg px-2 py-1.5 text-xs text-gray-700 focus:outline-none focus:ring-1 focus:ring-blue-500 cursor-pointer"
              >
                {knowledgeBases.map(kb => (
                  <option key={kb.id} value={kb.id}>
                    {kb.name}
                  </option>
                ))}
              </select>
            </div>
          )}

          {/* KB Stats */}
          <div className="flex items-center gap-1.5 pt-1 border-t border-gray-200">
            <Database className="w-3.5 h-3.5 text-gray-400 shrink-0" />
            <span className="text-xs text-gray-500">
              {stats.docCount} 篇文档 · {stats.chunkCount} 个片段
            </span>
          </div>
        </div>
      </div>
    </aside>
  )
}
