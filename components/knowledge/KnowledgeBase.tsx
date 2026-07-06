'use client'

import { useState, useEffect, useCallback } from 'react'
import { BookOpen, Plus, Trash2, FileText, Hash, Calendar, ChevronDown, ChevronUp, Loader2, FolderKanban, Menu } from 'lucide-react'
import { cn, formatDate, truncate } from '@/lib/utils'
import type { KnowledgeDocument, KBStats } from '@/types'
import { useRAG } from '@/contexts/RAGContext'
import { useSession } from '@/contexts/SessionContext'

export default function KnowledgeBase() {
  const { kbId, setKbId, knowledgeBases, refreshKBs } = useRAG()
  const { setSidebarOpen } = useSession()
  const [kbListOpen, setKbListOpen] = useState(false)
  const [docs, setDocs] = useState<KnowledgeDocument[]>([])
  const [stats, setStats] = useState<KBStats>({ docCount: 0, chunkCount: 0 })
  const [showForm, setShowForm] = useState(false)
  const [loading, setLoading] = useState(true)
  const [showKbForm, setShowKbForm] = useState(false)
  const [newKbName, setNewKbName] = useState('')
  const [newKbDesc, setNewKbDesc] = useState('')
  const [creatingKb, setCreatingKb] = useState(false)
  const [kbError, setKbError] = useState('')

  const activeKb = knowledgeBases.find(k => k.id === kbId) || knowledgeBases[0]

  const fetchDocs = useCallback(async () => {
    if (!kbId) return
    setLoading(true)
    try {
      const res = await fetch(`/api/knowledge?kb_id=${kbId}`)
      const data = await res.json()
      setDocs(data.documents ?? [])
      setStats(data.stats ?? { docCount: 0, chunkCount: 0 })
    } catch (e) {
      console.error('[KB] Failed to fetch documents:', e)
    } finally {
      setLoading(false)
    }
  }, [kbId])

  useEffect(() => {
    if (kbId) {
      fetchDocs()
    }
  }, [kbId, fetchDocs])

  const handleCreateKb = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!newKbName.trim()) {
      setKbError('名称不能为空')
      return
    }
    setCreatingKb(true)
    setKbError('')
    try {
      const res = await fetch('/api/knowledge-bases', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: newKbName.trim(), description: newKbDesc.trim() }),
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || '创建失败')
      setNewKbName('')
      setNewKbDesc('')
      setShowKbForm(false)
      await refreshKBs()
      if (data.id) {
        setKbId(data.id)
      }
    } catch (err) {
      setKbError((err as Error).message)
    } finally {
      setCreatingKb(false)
    }
  }

  const handleDeleteKb = async (e: React.MouseEvent, id: string, name: string) => {
    e.stopPropagation()
    if (id === '1') {
      alert('系统默认知识库无法删除')
      return
    }
    if (!confirm(`确认删除整个知识库「${name}」？此操作将级联删除该知识库下的所有文档及其向量数据，不可恢复！`)) return

    try {
      const res = await fetch(`/api/knowledge-bases/${id}`, { method: 'DELETE' })
      if (!res.ok) throw new Error('删除失败')
      await refreshKBs()
      if (kbId === id) {
        setKbId('1')
      }
    } catch (err) {
      alert((err as Error).message)
    }
  }

  return (
    <div className="flex h-full bg-gray-50 overflow-hidden relative">
      {/* KB List Sidebar Drawer Backdrop */}
      {kbListOpen && (
        <div
          onClick={() => setKbListOpen(false)}
          className="fixed inset-0 bg-black/40 z-30 lg:hidden transition-opacity duration-300"
        />
      )}

      {/* Left Column: KB list */}
      <div
        className={cn(
          "w-64 shrink-0 bg-white border-r border-gray-200 flex flex-col h-full fixed inset-y-0 left-0 z-40 transition-transform duration-300 ease-in-out lg:static lg:translate-x-0",
          kbListOpen ? "translate-x-0" : "-translate-x-full"
        )}
      >
        <div className="p-4 border-b border-gray-100 flex items-center justify-between shrink-0">
          <h2 className="font-semibold text-gray-800 text-sm flex items-center gap-1.5">
            <FolderKanban className="w-4 h-4 text-blue-600" />
            知识库列表
          </h2>
          <button
            onClick={() => setShowKbForm(v => !v)}
            className="p-1 hover:bg-gray-100 rounded text-blue-600 hover:text-blue-700 transition-colors"
            title="新建知识库"
          >
            <Plus className="w-4 h-4" />
          </button>
        </div>

        {/* KB Creation Form */}
        {showKbForm && (
          <form onSubmit={handleCreateKb} className="p-3 bg-blue-50 border-b border-blue-100 shrink-0 space-y-2.5">
            <div className="text-xs font-semibold text-blue-800">新建知识库</div>
            <input
              type="text"
              placeholder="知识库名称..."
              value={newKbName}
              onChange={e => setNewKbName(e.target.value)}
              className="w-full text-xs border border-gray-200 rounded px-2 py-1.5 focus:outline-none focus:ring-1 focus:ring-blue-500 bg-white"
            />
            <input
              type="text"
              placeholder="描述信息（可选）..."
              value={newKbDesc}
              onChange={e => setNewKbDesc(e.target.value)}
              className="w-full text-xs border border-gray-200 rounded px-2 py-1.5 focus:outline-none focus:ring-1 focus:ring-blue-500 bg-white"
            />
            {kbError && <p className="text-[10px] text-red-600">{kbError}</p>}
            <div className="flex gap-1.5">
              <button
                type="submit"
                disabled={creatingKb}
                className="text-[10px] bg-blue-600 text-white rounded px-2.5 py-1 hover:bg-blue-700 disabled:bg-blue-400 font-medium"
              >
                {creatingKb ? '创建中...' : '确定'}
              </button>
              <button
                type="button"
                onClick={() => { setShowKbForm(false); setKbError('') }}
                className="text-[10px] bg-white border border-gray-200 text-gray-600 rounded px-2.5 py-1 hover:bg-gray-50"
              >
                取消
              </button>
            </div>
          </form>
        )}

        {/* List mapping */}
        <div className="flex-1 overflow-y-auto p-2 space-y-1">
          {knowledgeBases.map(kb => {
            const isActive = kb.id === kbId
            return (
              <div
                key={kb.id}
                onClick={() => { setKbId(kb.id); setShowForm(false); setKbListOpen(false) }}
                className={cn(
                  'group flex items-center justify-between p-2.5 rounded-lg cursor-pointer transition-colors relative',
                  isActive ? 'bg-blue-50 text-blue-700' : 'hover:bg-gray-50 text-gray-700'
                )}
              >
                <div className="min-w-0 pr-6">
                  <p className="text-xs font-semibold truncate">{kb.name}</p>
                  <p className="text-[10px] text-gray-400 truncate mt-0.5">
                    {kb.description || '暂无描述'}
                  </p>
                  <p className="text-[9px] text-gray-400 mt-1">
                    {kb.docCount ?? 0} 文档 · {kb.chunkCount ?? 0} 片段
                  </p>
                </div>
                {kb.id !== '1' && (
                  <button
                    onClick={e => handleDeleteKb(e, kb.id, kb.name)}
                    className="absolute right-2 opacity-0 group-hover:opacity-100 hover:text-red-500 text-gray-400 transition-all p-1"
                    title="删除知识库"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                )}
              </div>
            )
          })}
        </div>
      </div>

      {/* Right Column: Doc list for active KB */}
      <div className="flex-1 flex flex-col h-full bg-white overflow-hidden">
        {/* Header */}
        <header className="flex items-center justify-between px-4 lg:px-6 py-3.5 bg-white border-b border-gray-200 shrink-0">
          <div className="flex items-center gap-2 min-w-0">
            <button
              onClick={() => setSidebarOpen(true)}
              className="md:hidden p-1 mr-1 text-gray-500 hover:text-gray-900 hover:bg-gray-100 rounded-lg transition-colors shrink-0"
              aria-label="打开菜单"
            >
              <Menu className="w-5 h-5" />
            </button>
            <button
              onClick={() => setKbListOpen(true)}
              className="lg:hidden p-1 mr-1 text-gray-500 hover:text-gray-900 hover:bg-gray-100 rounded-lg transition-colors shrink-0"
              aria-label="打开知识库列表"
            >
              <FolderKanban className="w-5 h-5 text-blue-600" />
            </button>
            <h1 className="font-semibold text-gray-900 truncate max-w-[120px] md:max-w-xs">{activeKb?.name || '加载中...'}</h1>
            <span className="text-xs bg-gray-100 text-gray-600 px-2 py-0.5 rounded-full shrink-0">
              {stats.docCount} 篇 · {stats.chunkCount} 片段
            </span>
          </div>
          <button
            onClick={() => setShowForm(v => !v)}
            disabled={!kbId}
            className="flex items-center gap-1.5 text-xs md:text-sm bg-blue-600 text-white px-2.5 py-1.5 md:px-3 rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50 shrink-0"
          >
            <Plus className="w-4 h-4" />
            <span className="hidden sm:inline">添加文档</span>
            <span className="sm:hidden">添加</span>
          </button>
        </header>

        <div className="flex-1 overflow-y-auto p-4 md:p-6 space-y-4 bg-gray-50">
          {/* Upload form */}
          {showForm && kbId && (
            <UploadForm
              kbId={kbId}
              onSuccess={() => { setShowForm(false); fetchDocs(); refreshKBs() }}
              onCancel={() => setShowForm(false)}
            />
          )}

          {/* Document list */}
          {loading ? (
            <div className="flex items-center justify-center py-20">
              <Loader2 className="w-6 h-6 animate-spin text-gray-400" />
            </div>
          ) : docs.length === 0 && !showForm ? (
            <EmptyKB onAdd={() => setShowForm(true)} />
          ) : (
            docs.map(doc => (
              <DocCard
                key={doc.id}
                doc={doc}
                onDelete={() => { fetchDocs(); refreshKBs() }}
              />
            ))
          )}
        </div>
      </div>
    </div>
  )
}

// ── Upload Form ─────────────────────────────────────────────────────────────

function UploadForm({ kbId, onSuccess, onCancel }: { kbId: string; onSuccess: () => void; onCancel: () => void }) {
  const [title, setTitle] = useState('')
  const [content, setContent] = useState('')
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!title.trim() || !content.trim()) {
      setError('标题和内容不能为空')
      return
    }

    setSaving(true)
    setError('')
    try {
      const res = await fetch('/api/knowledge', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title: title.trim(), content: content.trim(), kbId }),
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || '保存失败')
      onSuccess()
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="bg-white border border-blue-200 rounded-xl p-5 shadow-sm">
      <h2 className="font-semibold text-gray-900 mb-4 flex items-center gap-2">
        <Plus className="w-4 h-4 text-blue-600" />
        新建文档
      </h2>

      <form onSubmit={handleSubmit} className="space-y-3">
        <div>
          <label className="block text-xs font-medium text-gray-600 mb-1">文档标题</label>
          <input
            type="text"
            value={title}
            onChange={e => setTitle(e.target.value)}
            placeholder="例如：产品使用手册"
            className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>

        <div>
          <label className="block text-xs font-medium text-gray-600 mb-1">
            文档内容
            <span className="text-gray-400 font-normal ml-1">（支持 Markdown、纯文本）</span>
          </label>
          <textarea
            value={content}
            onChange={e => setContent(e.target.value)}
            placeholder="粘贴或输入文档内容…"
            rows={10}
            className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm font-mono focus:outline-none focus:ring-2 focus:ring-blue-500 resize-y"
          />
          <p className="text-xs text-gray-400 mt-1">
            {content.length} 字符 · 约 {Math.ceil(content.length / 600)} 个分块
          </p>
        </div>

        {error && (
          <p className="text-sm text-red-600 bg-red-50 border border-red-200 rounded-lg px-3 py-2">{error}</p>
        )}

        <div className="flex gap-2 pt-1">
          <button
            type="submit"
            disabled={saving}
            className={cn(
              'flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium text-white transition-colors',
              saving ? 'bg-blue-400 cursor-not-allowed' : 'bg-blue-600 hover:bg-blue-700'
            )}
          >
            {saving && <Loader2 className="w-3.5 h-3.5 animate-spin" />}
            {saving ? '处理中…' : '保存并向量化'}
          </button>
          <button
            type="button"
            onClick={onCancel}
            className="px-4 py-2 rounded-lg text-sm font-medium text-gray-600 hover:bg-gray-100 transition-colors"
          >
            取消
          </button>
        </div>
      </form>
    </div>
  )
}

// ── Document Card ───────────────────────────────────────────────────────────

function DocCard({ doc, onDelete }: { doc: KnowledgeDocument; onDelete: () => void }) {
  const [expanded, setExpanded] = useState(false)
  const [deleting, setDeleting] = useState(false)

  const handleDelete = async () => {
    if (!confirm(`确认删除「${doc.title}」？此操作不可恢复。`)) return
    setDeleting(true)
    try {
      const res = await fetch(`/api/knowledge/${doc.id}`, { method: 'DELETE' })
      if (!res.ok) throw new Error('删除失败')
      onDelete()
    } catch (e) {
      alert((e as Error).message)
    } finally {
      setDeleting(false)
    }
  }

  return (
    <div className="bg-white border border-gray-200 rounded-xl overflow-hidden shadow-sm hover:shadow-md transition-shadow">
      <div className="flex items-start justify-between p-4">
        <div className="flex items-start gap-3 min-w-0">
          <div className="w-9 h-9 bg-blue-50 rounded-lg flex items-center justify-center shrink-0 mt-0.5">
            <FileText className="w-4.5 h-4.5 text-blue-600" />
          </div>
          <div className="min-w-0">
            <h3 className="font-medium text-gray-900 text-sm truncate">{doc.title}</h3>
            <div className="flex flex-wrap items-center gap-2 md:gap-3 mt-1 text-xs text-gray-400">
              <span className="flex items-center gap-1">
                <Hash className="w-3 h-3" />
                {doc.chunkCount} 个片段
              </span>
              <span className="flex items-center gap-1">
                <Calendar className="w-3 h-3" />
                {formatDate(doc.createdAt)}
              </span>
              <span>{doc.content.length} 字符</span>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-1 shrink-0 ml-3">
          <button
            onClick={() => setExpanded(v => !v)}
            className="p-1.5 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-lg transition-colors"
            title={expanded ? '收起' : '预览内容'}
          >
            {expanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
          </button>
          <button
            onClick={handleDelete}
            disabled={deleting}
            className="p-1.5 text-gray-400 hover:text-red-500 hover:bg-red-50 rounded-lg transition-colors"
            title="删除"
          >
            {deleting ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Trash2 className="w-4 h-4" />
            )}
          </button>
        </div>
      </div>

      {expanded && (
        <div className="border-t border-gray-100 bg-gray-50 px-4 py-3">
          <pre className="text-xs text-gray-600 whitespace-pre-wrap font-mono leading-relaxed max-h-48 overflow-y-auto">
            {truncate(doc.content, 2000)}
          </pre>
        </div>
      )}
    </div>
  )
}

// ── Empty State ─────────────────────────────────────────────────────────────

function EmptyKB({ onAdd }: { onAdd: () => void }) {
  return (
    <div className="flex flex-col items-center justify-center py-24 gap-4 text-center">
      <div className="w-16 h-16 bg-gray-100 rounded-2xl flex items-center justify-center">
        <BookOpen className="w-8 h-8 text-gray-400" />
      </div>
      <div>
        <h3 className="font-semibold text-gray-800 mb-1">知识库为空</h3>
        <p className="text-sm text-gray-500 max-w-xs">
          添加文档后，AI 将在 RAG 模式下参考这些内容回答问题
        </p>
      </div>
      <button
        onClick={onAdd}
        className="flex items-center gap-2 text-sm bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors"
      >
        <Plus className="w-4 h-4" />
        添加第一篇文档
      </button>
    </div>
  )
}
