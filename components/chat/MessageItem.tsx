'use client'

import { useState, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { Bot, User, Brain, ChevronDown, ChevronUp, Loader2 } from 'lucide-react'
import { cn } from '@/lib/utils'

interface MessageItemProps {
  role: 'user' | 'assistant'
  content: string
  isStreaming?: boolean
}

// 解析消息内容，分离出思考过程和最终回答
function parseMessageContent(content: string) {
  const thinkStart = content.indexOf('<think>')
  if (thinkStart === -1) {
    return { thinking: '', response: content, isThinking: false }
  }

  const thinkEnd = content.indexOf('</think>')
  const prefix = content.slice(0, thinkStart)

  if (thinkEnd === -1) {
    // 还在思考中，或者 think 块尚未闭合
    const thinking = content.slice(thinkStart + 7)
    return { thinking, response: prefix, isThinking: true }
  } else {
    // 思考已结束
    const thinking = content.slice(thinkStart + 7, thinkEnd)
    const suffix = content.slice(thinkEnd + 8)
    return { thinking, response: prefix + suffix, isThinking: false }
  }
}

export default function MessageItem({ role, content, isStreaming }: MessageItemProps) {
  const isUser = role === 'user'

  // 提取思考过程与最终回答
  const { thinking, response, isThinking } = isUser
    ? { thinking: '', response: content, isThinking: false }
    : parseMessageContent(content)

  // 控制思考折叠面板的展开状态，默认收起
  const [isExpanded, setIsExpanded] = useState(false)

  return (
    <div className={cn('flex gap-3 px-4 py-3', isUser ? 'flex-row-reverse' : 'flex-row')}>
      {/* Avatar */}
      <div
        className={cn(
          'w-8 h-8 rounded-full flex items-center justify-center shrink-0 mt-0.5',
          isUser ? 'bg-blue-600' : 'bg-gray-100 border border-gray-200'
        )}
      >
        {isUser ? (
          <User className="w-4 h-4 text-white" />
        ) : (
          <Bot className="w-4 h-4 text-gray-600" />
        )}
      </div>

      {/* Bubble */}
      <div className={cn('flex flex-col gap-1.5 max-w-[75%]', isUser ? 'items-end' : 'items-start')}>
        <div
          className={cn(
            'rounded-2xl px-4 py-2.5 text-sm leading-relaxed w-full',
            isUser
              ? 'bg-blue-600 text-white rounded-tr-sm'
              : 'bg-white text-gray-800 border border-gray-200 rounded-tl-sm shadow-sm'
          )}
        >
          {isUser ? (
            <p className="whitespace-pre-wrap">{response}</p>
          ) : (
            <div className="space-y-3">
              {/* 思考过程展示区 */}
              {thinking && (
                <div className="border border-gray-100 rounded-xl bg-gray-50/70 overflow-hidden text-xs">
                  {/* 折叠栏头部 */}
                  <button
                    onClick={() => setIsExpanded(!isExpanded)}
                    className="flex items-center justify-between w-full px-3 py-2 text-gray-500 hover:text-gray-700 transition-colors font-medium cursor-pointer"
                  >
                    <div className="flex items-center gap-1.5">
                      {isThinking ? (
                        <Loader2 className="w-3.5 h-3.5 animate-spin text-blue-500" />
                      ) : (
                        <Brain className="w-3.5 h-3.5 text-purple-500" />
                      )}
                      <span>
                        {isThinking ? "正在思考..." : "已思考完成"}
                      </span>
                    </div>
                    <div>
                      {isExpanded ? (
                        <ChevronUp className="w-3.5 h-3.5" />
                      ) : (
                        <ChevronDown className="w-3.5 h-3.5" />
                      )}
                    </div>
                  </button>

                  {/* 折叠栏内容 */}
                  {isExpanded && (
                    <div className="px-3 pb-3 pt-0 text-gray-500 border-t border-gray-100/50 whitespace-pre-wrap leading-relaxed">
                      {thinking}
                      {isThinking && isStreaming && (
                        <span className="inline-block w-1 h-3.5 bg-gray-400 ml-0.5 animate-pulse rounded-sm" />
                      )}
                    </div>
                  )}
                </div>
              )}

              {/* 最终回答展示区 */}
              {response && (
                <div className="prose-chat">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>{response}</ReactMarkdown>
                  {isStreaming && !isThinking && (
                    <span className="inline-block w-1.5 h-4 bg-gray-400 ml-0.5 animate-pulse rounded-sm" />
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
