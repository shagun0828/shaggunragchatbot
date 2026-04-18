'use client'

import { useState } from 'react'
import { ChatInterface } from '@/components/chat/chat-interface'
import { SearchInterface } from '@/components/search/search-interface'
import { Dashboard } from '@/components/dashboard/dashboard'
import { Sidebar } from '@/components/layout/sidebar'
import { Header } from '@/components/layout/header'

export default function Home() {
  const [activeView, setActiveView] = useState<'chat' | 'search' | 'dashboard'>('chat')
  const [isSidebarOpen, setIsSidebarOpen] = useState(true)

  return (
    <div className="flex h-screen bg-background">
      <Sidebar 
        isOpen={isSidebarOpen}
        onToggle={() => setIsSidebarOpen(!isSidebarOpen)}
        activeView={activeView}
        onViewChange={setActiveView}
      />
      
      <div className="flex-1 flex flex-col">
        <Header 
          onSidebarToggle={() => setIsSidebarOpen(!isSidebarOpen)}
          activeView={activeView}
        />
        
        <main className="flex-1 overflow-hidden">
          {activeView === 'chat' && <ChatInterface />}
          {activeView === 'search' && <SearchInterface />}
          {activeView === 'dashboard' && <Dashboard />}
        </main>
      </div>
    </div>
  )
}
