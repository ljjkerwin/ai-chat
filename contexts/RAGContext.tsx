'use client'

import { createContext, useContext, useState, useEffect, useCallback, type ReactNode } from 'react'
import type { KnowledgeBase } from '@/types'

interface RAGContextValue {
  ragEnabled: boolean
  setRagEnabled: (v: boolean) => void
  kbId: string | null
  setKbId: (v: string | null) => void
  knowledgeBases: KnowledgeBase[]
  refreshKBs: () => Promise<void>
}

const RAGContext = createContext<RAGContextValue>({
  ragEnabled: false,
  setRagEnabled: () => {},
  kbId: null,
  setKbId: () => {},
  knowledgeBases: [],
  refreshKBs: async () => {},
})

export function RAGProvider({ children }: { children: ReactNode }) {
  const [ragEnabled, setRagEnabledState] = useState(false)
  const [kbId, setKbIdState] = useState<string | null>(null)
  const [knowledgeBases, setKnowledgeBases] = useState<KnowledgeBase[]>([])

  useEffect(() => {
    setRagEnabledState(localStorage.getItem('ragEnabled') === 'true')
    setKbIdState(localStorage.getItem('kbId') || '1')
  }, [])

  const setRagEnabled = (v: boolean) => {
    setRagEnabledState(v)
    localStorage.setItem('ragEnabled', String(v))
  }

  const setKbId = (v: string | null) => {
    setKbIdState(v)
    if (v) {
      localStorage.setItem('kbId', v)
    } else {
      localStorage.removeItem('kbId')
    }
  }

  const refreshKBs = useCallback(async () => {
    try {
      const res = await fetch('/api/knowledge-bases')
      const data = await res.json()
      setKnowledgeBases(data.knowledgeBases ?? [])
    } catch (e) {
      console.error('[RAGContext] Failed to load knowledge bases:', e)
    }
  }, [])

  useEffect(() => {
    refreshKBs()
  }, [refreshKBs])

  return (
    <RAGContext.Provider value={{
      ragEnabled,
      setRagEnabled,
      kbId,
      setKbId,
      knowledgeBases,
      refreshKBs
    }}>
      {children}
    </RAGContext.Provider>
  )
}

export const useRAG = () => useContext(RAGContext)
