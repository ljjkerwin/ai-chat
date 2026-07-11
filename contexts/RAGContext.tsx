'use client'

import { createContext, useContext, useState, useEffect, useCallback, type ReactNode } from 'react'
import type { KnowledgeBase } from '@/types'

interface RAGContextValue {
  ragEnabled: boolean
  setRagEnabled: (v: boolean) => void
  kbId: string | null
  setKbId: (v: string | null) => void
  agentType: '1' | '2'
  setAgentType: (v: '1' | '2') => void
  knowledgeBases: KnowledgeBase[]
  refreshKBs: () => Promise<void>
}

const RAGContext = createContext<RAGContextValue>({
  ragEnabled: false,
  setRagEnabled: () => {},
  kbId: null,
  setKbId: () => {},
  agentType: '1',
  setAgentType: () => {},
  knowledgeBases: [],
  refreshKBs: async () => {},
})

export function RAGProvider({ children }: { children: ReactNode }) {
  const [ragEnabled, setRagEnabledState] = useState(false)
  const [kbId, setKbIdState] = useState<string | null>(null)
  const [agentType, setAgentTypeState] = useState<'1' | '2'>('1')
  const [knowledgeBases, setKnowledgeBases] = useState<KnowledgeBase[]>([])

  useEffect(() => {
    setRagEnabledState(localStorage.getItem('ragEnabled') === 'true')
    setKbIdState(localStorage.getItem('kbId') || '1')
    const params = new URLSearchParams(window.location.search)
    const agentParam = params.get('agent')
    if (agentParam && (agentParam === 'react' || agentParam === 'langgraph' || agentParam === 'default' || agentParam === '1' || agentParam === '2')) {
      const mappedAgent = (agentParam === 'react' || agentParam === 'default') ? '1' : (agentParam === 'langgraph' ? '2' : agentParam)
      setAgentTypeState(mappedAgent as '1' | '2')
      localStorage.setItem('agentType', mappedAgent)
    } else {
      const cached = localStorage.getItem('agentType')
      const defaultAgent = (cached === 'react' || cached === 'default') ? '1' : (cached === 'langgraph' ? '2' : ((cached as '1' | '2') || '1'))
      setAgentTypeState(defaultAgent)
      localStorage.setItem('agentType', defaultAgent)
    }
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

  const setAgentType = (v: '1' | '2') => {
    setAgentTypeState(v)
    localStorage.setItem('agentType', v)
  }

  const refreshKBs = useCallback(async () => {
    console.log("DEBUG: refreshKBs starting...")
    try {
      const res = await fetch('/api/knowledge-bases')
      const data = await res.json()
      console.log("DEBUG: refreshKBs succeeded, data:", data)
      setKnowledgeBases(data.knowledgeBases ?? [])
    } catch (e) {
      console.error('[RAGContext] Failed to load knowledge bases:', e)
    }
  }, [])

  useEffect(() => {
    console.log("DEBUG: RAGProvider useEffect for refreshKBs running")
    refreshKBs()
  }, [refreshKBs])

  return (
    <RAGContext.Provider value={{
      ragEnabled,
      setRagEnabled,
      kbId,
      setKbId,
      agentType,
      setAgentType,
      knowledgeBases,
      refreshKBs
    }}>
      {children}
    </RAGContext.Provider>
  )
}

export const useRAG = () => useContext(RAGContext)
