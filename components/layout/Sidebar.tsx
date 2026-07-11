'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { Bot, Trash2 } from 'lucide-react'
import { useSession } from '@/contexts/SessionContext'
import { useRAG } from '@/contexts/RAGContext'
import { useEffect } from 'react'
import { cn } from '@/lib/utils'
import agents from '@/data/agents.json'

export default function Sidebar() {
  const pathname = usePathname()
  const { agentType, setAgentType } = useRAG()
  const {
    sessions,
    reloadSessions,
    currentSessionId,
    switchSession,
    sidebarOpen,
    setSidebarOpen,
    loadingMore,
    loadMoreSessions,
    initialLoading
  } = useSession()

  const handleScroll = (e: React.UIEvent<HTMLDivElement>) => {
    const target = e.currentTarget
    if (target.scrollHeight - target.scrollTop - target.clientHeight < 20) {
      loadMoreSessions()
    }
  }

  const isChat = pathname.startsWith('/chat')

  useEffect(() => {
    if (isChat) reloadSessions()
  }, [isChat, reloadSessions])

  const handleDelete = async (e: React.MouseEvent, id: string) => {
    e.stopPropagation()
    await fetch(`/api/sessions/${id}`, { method: 'DELETE' })
    if (currentSessionId === id) switchSession(null)
    reloadSessions()
  }

  return (
    <>
      {/* Backdrop */}
      {sidebarOpen && (
        <div
          onClick={() => setSidebarOpen(false)}
          className="fixed inset-0 bg-black/40 z-30 md:hidden transition-opacity duration-300"
        />
      )}
      <aside
        className={cn(
          "w-60 shrink-0 flex flex-col h-full bg-white border-r border-gray-200 fixed inset-y-0 left-0 z-40 transition-transform duration-300 ease-in-out md:static md:translate-x-0",
          sidebarOpen ? "translate-x-0" : "-translate-x-full"
        )}
      >
        {/* Navigation */}
        <nav className="p-3 pt-4 space-y-1 shrink-0">
          <h3 className="px-3 mb-2 text-xs font-semibold text-gray-400 uppercase tracking-wider">智能体</h3>
          {agents.map(agent => {
            const isActive = pathname.startsWith('/chat') && agentType === agent.id

            let linkHref = '/chat'
            const params = new URLSearchParams()
            if (agent.id !== '1') {
              params.set('agent', agent.id)
            }
            const queryStr = params.toString()
            if (queryStr) {
              linkHref += `?${queryStr}`
            }

            return (
              <Link
                key={agent.id}
                href={linkHref}
                onClick={() => {
                  setAgentType(agent.id as '1' | '2')
                  switchSession(null)
                }}
                className={cn(
                  'group flex items-start gap-2.5 px-3 py-2.5 rounded-xl transition-all border border-transparent',
                  isActive
                    ? 'bg-blue-50/70 border-blue-100/50 text-blue-700 font-semibold'
                    : 'text-gray-600 hover:bg-gray-100/70 hover:text-gray-900'
                )}
              >
                <Bot className={cn(
                  "w-4 h-4 shrink-0 transition-colors mt-0.5",
                  isActive ? "text-blue-600" : "text-gray-400 group-hover:text-gray-500"
                )} />
                <div className="flex flex-col min-w-0">
                  <span className={cn(
                    "text-sm font-semibold truncate leading-5",
                    isActive ? "text-blue-700" : "text-gray-800"
                  )}>
                    {agent.name}
                  </span>
                  <span className={cn(
                    "text-[11px] truncate leading-4 font-normal mt-0.5",
                    isActive ? "text-blue-500" : "text-gray-400 group-hover:text-gray-500"
                  )}>
                    {agent.description}
                  </span>
                </div>
              </Link>
            )
          })}
        </nav>



        {/* Session history — only on /chat */}
        {isChat && (
          <div className="flex-1 flex flex-col min-h-0 p-3">
            <h3 className="px-3 mb-2 text-xs font-semibold text-gray-400 uppercase tracking-wider">最近对话</h3>
            <div
              className="flex-1 overflow-y-auto space-y-0.5"
              onScroll={handleScroll}
            >
              {initialLoading ? (
                <div className="flex justify-center items-center gap-1.5 py-4 text-gray-400 text-xs">
                  <span className="inline-block w-3.5 h-3.5 border-2 border-blue-600 border-t-transparent rounded-full animate-spin"></span>
                  <span>加载中...</span>
                </div>
              ) : sessions.length === 0 ? (
                <p className="text-xs text-gray-400 text-center pt-4">暂无历史对话</p>
              ) : (
                <>
                  {sessions.map(session => (
                    <div
                      key={session.id}
                      onClick={() => switchSession(session.id)}
                      className={cn(
                        'group flex items-center gap-2 px-3 py-2 rounded-lg cursor-pointer transition-colors text-sm font-medium',
                        currentSessionId === session.id
                          ? 'text-blue-700 font-semibold'
                          : 'text-gray-600 hover:text-gray-900'
                      )}
                    >
                      <span className="flex-1 truncate">{session.title}</span>
                      <button
                        onClick={e => handleDelete(e, session.id)}
                        className="shrink-0 opacity-0 group-hover:opacity-100 text-gray-400 hover:text-red-500 transition-all ml-1"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  ))}
                  {loadingMore && (
                    <div className="flex justify-center items-center gap-1.5 py-2 text-gray-400 text-xs">
                      <span className="inline-block w-3.5 h-3.5 border-2 border-blue-600 border-t-transparent rounded-full animate-spin"></span>
                      <span>加载中...</span>
                    </div>
                  )}
                </>
              )}
            </div>
          </div>
        )}

        {/* Spacer when not on chat */}
        {!isChat && <div className="flex-1" />}


      </aside>
    </>
  )
}
