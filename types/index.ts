export interface KnowledgeDocument {
  id: string
  title: string
  content: string
  fileType: string
  createdAt: number
  chunkCount?: number
}

export interface Source {
  documentId: string
  documentTitle: string
  chunkContent: string
  score: number
}

export interface RAGContext {
  context: string
  sources: Source[]
}

export interface KBStats {
  docCount: number
  chunkCount: number
}

export interface Session {
  id: string
  title: string
  created_at: number
  updated_at: number
  message_count: number
}

export interface SessionMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  created_at: number
}
