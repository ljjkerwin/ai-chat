'use client'

import { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { Bot, User, Brain, ChevronDown, ChevronUp, Loader2 } from 'lucide-react'
import { cn } from '@/lib/utils'

interface MessageItemProps {
  role: 'user' | 'assistant'
  content: string
  isStreaming?: boolean
}

// 解析消息内容，分离并合并所有的思考过程和最终回答
function parseMessageContent(content: string) {
  const thinkingParts: string[] = []
  const responseParts: string[] = []
  let isThinking = false

  let currentIndex = 0
  while (currentIndex < content.length) {
    const nextThinkStart = content.indexOf('<think>', currentIndex)

    if (nextThinkStart === -1) {
      // 没有更多的思考块，剩余内容全部为回答
      responseParts.push(content.slice(currentIndex))
      break
    }

    // <think> 之前的内容是回答段落
    if (nextThinkStart > currentIndex) {
      responseParts.push(content.slice(currentIndex, nextThinkStart))
    }

    const thinkContentStart = nextThinkStart + 7 // '<think>'.length === 7
    const nextThinkEnd = content.indexOf('</think>', thinkContentStart)

    if (nextThinkEnd === -1) {
      // 思考块尚未闭合，剩余内容均属于当前的思考过程
      thinkingParts.push(content.slice(thinkContentStart))
      isThinking = true
      break
    }

    // 思考块已闭合，提取思考内容
    thinkingParts.push(content.slice(thinkContentStart, nextThinkEnd))
    currentIndex = nextThinkEnd + 8 // '</think>'.length === 8
  }

  const thinking = thinkingParts.filter(Boolean).join('\n\n').trim()
  const response = responseParts.join('')

  return { thinking, response, isThinking }
}

export default function MessageItem({ role, content, isStreaming }: MessageItemProps) {
  const isUser = role === 'user'

  // 提取思考过程与最终回答
  const { thinking, response, isThinking: parsedIsThinking } = isUser
    ? { thinking: '', response: content, isThinking: false }
    : parseMessageContent(content)

  // 当有思考内容，且消息气泡内容为空，并且消息还在接收中时，依然展示思考中状态（因为有可能下一轮消息内容也需要思考）
  const isThinking = parsedIsThinking || !!(isStreaming && !response && thinking)

  // 控制思考折叠面板的展开状态，默认不展开
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

      {/* Bubble Container */}
      <div className={cn('flex flex-col gap-1.5 max-w-[75%]', isUser ? 'items-end' : 'items-start')}>
        {/* 思考过程展示区 (独立于气泡之外) */}
        {!isUser && (thinking || isThinking) && (
          <div className="border border-gray-100 rounded-xl bg-gray-50/70 overflow-hidden text-xs shadow-sm">
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
                  {isThinking ? '思考中...' : '思考'}
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
              <div className="px-3 pb-3 pt-2 text-gray-500 border-t border-gray-100/50 leading-relaxed">
                <div className="prose-thinking">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>{thinking}</ReactMarkdown>
                </div>
              </div>
            )}
          </div>
        )}

        {/* 最终回答气泡 */}
        {(isUser || response || (isStreaming && !response && !isThinking)) && (
          <div
            className={cn(
              'rounded-2xl px-4 py-2.5 text-sm leading-relaxed',
              isUser
                ? 'bg-blue-600 text-white rounded-tr-sm'
                : 'bg-white text-gray-800 border border-gray-200 rounded-tl-sm shadow-sm'
            )}
          >
            {isUser ? (
              <p className="whitespace-pre-wrap">{response}</p>
            ) : response ? (
              <div className="prose-chat">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>{response}</ReactMarkdown>
              </div>
            ) : (
              // 当消息气泡还没有内容且消息还在接收中时，显示loading
              isStreaming && (
                <div className="flex items-center gap-2 text-gray-400 py-1 select-none">
                  <Loader2 className="w-4 h-4 animate-spin text-blue-500 shrink-0" />
                  <span className="font-medium">正在生成回答...</span>
                </div>
              )
            )}
          </div>
        )}
      </div>
    </div>
  )
}
