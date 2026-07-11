'use client'

import { createContext, useContext, useState, useCallback, useEffect, useRef, type ReactNode } from 'react'
import type { Session, SessionMessage } from '@/types'
import { useRAG } from './RAGContext'

interface SessionContextValue {
  currentSessionId: string | null
  setCurrentSessionId: (id: string | null) => void
  sessions: Session[]
  reloadSessions: () => void
  pendingMessages: SessionMessage[] | null
  switchSession: (id: string | null) => Promise<void>
  clearPendingMessages: () => void
  sidebarOpen: boolean
  setSidebarOpen: (open: boolean) => void
  hasMore: boolean
  loadingMore: boolean
  loadMoreSessions: () => Promise<void>
  initialLoading: boolean
}

const SessionContext = createContext<SessionContextValue>({
  currentSessionId: null,
  setCurrentSessionId: () => { },
  sessions: [],
  reloadSessions: () => { },
  pendingMessages: null,
  switchSession: async () => { },
  clearPendingMessages: () => { },
  sidebarOpen: false,
  setSidebarOpen: () => { },
  hasMore: false,
  loadingMore: false,
  loadMoreSessions: async () => { },
  initialLoading: true,
})

export function SessionProvider({ children }: { children: ReactNode }) {
  const { agentType } = useRAG()
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null)
  const [sessions, setSessions] = useState<Session[]>([])
  const [pendingMessages, setPendingMessages] = useState<SessionMessage[] | null>(null)
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [page, setPage] = useState(1)
  const [hasMore, setHasMore] = useState(true)
  const [loadingMore, setLoadingMore] = useState(false)
  const [initialLoading, setInitialLoading] = useState(true)

  const lastFetchId = useRef(0)

  const reloadSessions = useCallback(() => {
    const fetchId = ++lastFetchId.current
    setPage(1)
    setHasMore(true)
    setInitialLoading(true)
    fetch(`/api/sessions?page=1&page_size=30&agent_type=${agentType}`)
      .then(r => r.json())
      .then(d => {
        if (fetchId === lastFetchId.current) {
          setSessions(d.sessions ?? [])
          setHasMore(d.has_more ?? false)
        }
      })
      .catch(() => { })
      .finally(() => {
        if (fetchId === lastFetchId.current) {
          setInitialLoading(false)
        }
      })
  }, [agentType])

  const loadMoreSessions = useCallback(async () => {
    if (loadingMore || !hasMore) return
    setLoadingMore(true)
    try {
      const nextPage = page + 1
      const res = await fetch(`/api/sessions?page=${nextPage}&page_size=30&agent_type=${agentType}`)
      const d = await res.json()
      setSessions(prev => {
        const existingIds = new Set(prev.map(s => s.id))
        const newSessions = (d.sessions ?? []).filter((s: Session) => !existingIds.has(s.id))
        return [...prev, ...newSessions]
      })
      setPage(nextPage)
      setHasMore(d.has_more ?? false)
    } catch (err) {
      console.error('Failed to load more sessions:', err)
    } finally {
      setLoadingMore(false)
    }
  }, [page, hasMore, loadingMore, agentType])

  useEffect(() => {
    reloadSessions()
    setCurrentSessionId(null)
  }, [agentType, reloadSessions])

  const switchSession = useCallback(async (id: string | null) => {
    setCurrentSessionId(id)
    if (id === null) {
      setPendingMessages([])
      return
    }
    const res = await fetch(`/api/sessions/${id}/messages`)
    const data = await res.json()
    setPendingMessages(data.messages ?? [])
    setSidebarOpen(false) // Close sidebar drawer when switching session on mobile
  }, [])

  const clearPendingMessages = useCallback(() => setPendingMessages(null), [])

  return (
    <SessionContext.Provider value={{
      currentSessionId,
      setCurrentSessionId,
      sessions,
      reloadSessions,
      pendingMessages,
      switchSession,
      clearPendingMessages,
      sidebarOpen,
      setSidebarOpen,
      hasMore,
      loadingMore,
      loadMoreSessions,
      initialLoading,
    }}>
      {children}
    </SessionContext.Provider>
  )
}

export const useSession = () => useContext(SessionContext)
