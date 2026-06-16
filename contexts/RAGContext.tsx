'use client'

import { createContext, useContext, useState, useEffect, type ReactNode } from 'react'

interface RAGContextValue {
  ragEnabled: boolean
  setRagEnabled: (v: boolean) => void
}

const RAGContext = createContext<RAGContextValue>({
  ragEnabled: false,
  setRagEnabled: () => {},
})

export function RAGProvider({ children }: { children: ReactNode }) {
  const [ragEnabled, setRagEnabledState] = useState(false)

  useEffect(() => {
    setRagEnabledState(localStorage.getItem('ragEnabled') === 'true')
  }, [])

  const setRagEnabled = (v: boolean) => {
    setRagEnabledState(v)
    localStorage.setItem('ragEnabled', String(v))
  }

  return (
    <RAGContext.Provider value={{ ragEnabled, setRagEnabled }}>
      {children}
    </RAGContext.Provider>
  )
}

export const useRAG = () => useContext(RAGContext)
