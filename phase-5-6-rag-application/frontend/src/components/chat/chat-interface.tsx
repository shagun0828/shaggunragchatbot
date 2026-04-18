'use client'

import { useState, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Send, Mic, Paperclip, Smile, MoreVertical, ThumbsUp, ThumbsDown } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { cn } from '@/lib/utils'

interface Message {
  id: string
  type: 'user' | 'assistant' | 'system'
  content: string
  timestamp: number
  sources?: Array<{
    id: string
    title: string
    snippet: string
    score: number
  }>
  isTyping?: boolean
}

export function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      type: 'system',
      content: 'Welcome to the Phase 5-6 RAG Chat Assistant! I can help you with mutual fund information, investment advice, and financial queries.',
      timestamp: Date.now() / 1000
    }
  ])
  const [inputValue, setInputValue] = useState('')
  const [isTyping, setIsTyping] = useState(false)
  const [isConnected, setIsConnected] = useState(true)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleSendMessage = async () => {
    if (!inputValue.trim() || isTyping) return

    const userMessage: Message = {
      id: Date.now().toString(),
      type: 'user',
      content: inputValue,
      timestamp: Date.now() / 1000
    }

    setMessages(prev => [...prev, userMessage])
    setInputValue('')
    setIsTyping(true)

    // Simulate AI response
    setTimeout(() => {
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'assistant',
        content: `I understand your question about "${userMessage.content}". Based on my analysis, I can provide you with comprehensive information about this topic. Let me help you with detailed insights and recommendations.`,
        timestamp: Date.now() / 1000,
        sources: [
          {
            id: 'doc1',
            title: 'Mutual Fund Basics',
            snippet: 'Mutual funds are investment vehicles that pool money from multiple investors...',
            score: 0.95
          },
          {
            id: 'doc2',
            title: 'Investment Strategies',
            snippet: 'Diversification is key to managing investment risk effectively...',
            score: 0.87
          }
        ]
      }
      
      setMessages(prev => [...prev, assistantMessage])
      setIsTyping(false)
    }, 2000)
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSendMessage()
    }
  }

  return (
    <div className="flex flex-col h-full bg-background">
      {/* Chat messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        <AnimatePresence>
          {messages.map((message) => (
            <motion.div
              key={message.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.3 }}
              className={cn(
                "flex",
                message.type === 'user' ? "justify-end" : "justify-start"
              )}
            >
              <div
                className={cn(
                  "max-w-[80%] lg:max-w-[70%]",
                  message.type === 'user' ? "order-2" : "order-1"
                )}
              >
                <Card
                  className={cn(
                    "p-4 shadow-lg",
                    message.type === 'user'
                      ? "bg-primary text-primary-foreground ml-4"
                      : "bg-card text-card-foreground mr-4",
                    message.type === 'system' && "border-l-4 border-l-blue-500 bg-blue-50 dark:bg-blue-950/20"
                  )}
                >
                  <div className="text-sm whitespace-pre-wrap">{message.content}</div>
                  
                  {message.sources && (
                    <div className="mt-3 pt-3 border-t border-border">
                      <div className="text-xs text-muted-foreground mb-2">Sources:</div>
                      <div className="space-y-2">
                        {message.sources.map((source, index) => (
                          <div key={index} className="text-xs p-2 bg-accent/50 rounded">
                            <div className="font-medium">{source.title}</div>
                            <div className="text-muted-foreground mt-1">
                              {source.snippet}
                            </div>
                            <div className="text-xs text-muted-foreground mt-1">
                              Relevance: {(source.score * 100).toFixed(1)}%
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                  
                  <div className="flex items-center justify-between mt-3 pt-2 border-t border-border">
                    <div className="text-xs text-muted-foreground">
                      {new Date(message.timestamp * 1000).toLocaleTimeString()}
                    </div>
                    
                    {message.type === 'assistant' && (
                      <div className="flex items-center space-x-2">
                        <Button variant="ghost" size="icon" className="w-6 h-6">
                          <ThumbsUp className="w-3 h-3" />
                        </Button>
                        <Button variant="ghost" size="icon" className="w-6 h-6">
                          <ThumbsDown className="w-3 h-3" />
                        </Button>
                      </div>
                    )}
                  </div>
                </Card>
              </div>
            </motion.div>
          ))}
        </AnimatePresence>

        {/* Typing indicator */}
        <AnimatePresence>
          {isTyping && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="flex justify-start"
            >
              <Card className="p-4 bg-card text-card-foreground mr-4 max-w-[80%] lg:max-w-[70%]">
                <div className="flex items-center space-x-2">
                  <div className="flex space-x-1">
                    <div className="w-2 h-2 bg-primary rounded-full animate-bounce" />
                    <div className="w-2 h-2 bg-primary rounded-full animate-bounce" style={{ animationDelay: '0.1s' }} />
                    <div className="w-2 h-2 bg-primary rounded-full animate-bounce" style={{ animationDelay: '0.2s' }} />
                  </div>
                  <span className="text-sm text-muted-foreground">Assistant is thinking...</span>
                </div>
              </Card>
            </motion.div>
          )}
        </AnimatePresence>

        <div ref={messagesEndRef} />
      </div>

      {/* Input area */}
      <div className="border-t border-border p-4">
        <div className="flex items-end space-x-2">
          {/* Attachment button */}
          <Button variant="ghost" size="icon" className="shrink-0">
            <Paperclip className="w-5 h-5" />
          </Button>

          {/* Message input */}
          <div className="flex-1 relative">
            <textarea
              ref={inputRef}
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Type your message..."
              className="w-full px-4 py-3 bg-accent/50 border border-border rounded-lg resize-none focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent transition-all duration-200"
              rows={1}
              style={{ minHeight: '48px', maxHeight: '120px' }}
            />
          </div>

          {/* Emoji button */}
          <Button variant="ghost" size="icon" className="shrink-0">
            <Smile className="w-5 h-5" />
          </Button>

          {/* Voice input button */}
          <Button variant="ghost" size="icon" className="shrink-0">
            <Mic className="w-5 h-5" />
          </Button>

          {/* Send button */}
          <Button
            onClick={handleSendMessage}
            disabled={!inputValue.trim() || isTyping}
            size="icon"
            className="shrink-0"
          >
            <Send className="w-5 h-5" />
          </Button>
        </div>

        {/* Connection status */}
        <div className="flex items-center justify-between mt-2 text-xs text-muted-foreground">
          <div className="flex items-center space-x-1">
            <div className={cn(
              "w-2 h-2 rounded-full",
              isConnected ? "bg-green-500" : "bg-red-500"
            )} />
            <span>{isConnected ? "Connected" : "Disconnected"}</span>
          </div>
          <div>
            Powered by Phase 5-6 RAG System
          </div>
        </div>
      </div>
    </div>
  )
}
