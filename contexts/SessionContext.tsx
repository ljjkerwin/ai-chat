'use client'

import { createContext, useContext, useState, useCallback, type ReactNode } from 'react'
import type { Session, SessionMessage } from '@/types'

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
})

export function SessionProvider({ children }: { children: ReactNode }) {
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null)
  const [sessions, setSessions] = useState<Session[]>([])
  const [pendingMessages, setPendingMessages] = useState<SessionMessage[] | null>(null)
  const [sidebarOpen, setSidebarOpen] = useState(false)

  const reloadSessions = useCallback(() => {
    fetch('/api/sessions')
      .then(r => r.json())
      .then(d => setSessions(d.sessions ?? []))
      .catch(() => { })
  }, [])

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
    }}>
      {children}
    </SessionContext.Provider>
  )
}

export const useSession = () => useContext(SessionContext)
