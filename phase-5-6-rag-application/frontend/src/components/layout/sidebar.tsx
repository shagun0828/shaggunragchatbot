'use client'

import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  MessageSquare, 
  Search, 
  BarChart3, 
  Settings, 
  Menu, 
  X,
  Brain,
  Database,
  Activity,
  Shield
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'

interface SidebarProps {
  isOpen: boolean
  onToggle: () => void
  activeView: 'chat' | 'search' | 'dashboard'
  onViewChange: (view: 'chat' | 'search' | 'dashboard') => void
}

const navigationItems = [
  {
    id: 'chat',
    label: 'Chat Interface',
    icon: MessageSquare,
    description: 'AI-powered conversational assistant'
  },
  {
    id: 'search',
    label: 'Search',
    icon: Search,
    description: 'Advanced semantic search'
  },
  {
    id: 'dashboard',
    label: 'Dashboard',
    icon: BarChart3,
    description: 'Analytics and monitoring'
  }
]

const secondaryItems = [
  {
    id: 'settings',
    label: 'Settings',
    icon: Settings,
    description: 'Application preferences'
  },
  {
    id: 'status',
    label: 'System Status',
    icon: Activity,
    description: 'Health and performance'
  }
]

export function Sidebar({ isOpen, onToggle, activeView, onViewChange }: SidebarProps) {
  const [isHovered, setIsHovered] = useState<string | null>(null)

  return (
    <>
      {/* Mobile backdrop */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/50 backdrop-blur-sm z-40 lg:hidden"
            onClick={onToggle}
          />
        )}
      </AnimatePresence>

      {/* Sidebar */}
      <motion.div
        initial={false}
        animate={{
          x: isOpen ? 0 : -320,
          transition: { type: "spring", stiffness: 300, damping: 30 }
        }}
        className={cn(
          "fixed left-0 top-0 h-full w-80 bg-background/95 backdrop-blur-xl border-r border-border z-50 lg:relative lg:translate-x-0",
          "overflow-hidden flex flex-col"
        )}
      >
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-border">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
              <Brain className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-lg font-semibold text-foreground">RAG System</h1>
              <p className="text-xs text-muted-foreground">Phase 5-6</p>
            </div>
          </div>
          <Button
            variant="ghost"
            size="icon"
            onClick={onToggle}
            className="lg:hidden"
          >
            <X className="w-5 h-5" />
          </Button>
        </div>

        {/* Navigation */}
        <nav className="flex-1 p-4 space-y-2">
          <div className="space-y-1">
            {navigationItems.map((item) => {
              const Icon = item.icon
              const isActive = activeView === item.id
              
              return (
                <motion.button
                  key={item.id}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  onClick={() => onViewChange(item.id as 'chat' | 'search' | 'dashboard')}
                  onMouseEnter={() => setIsHovered(item.id)}
                  onMouseLeave={() => setIsHovered(null)}
                  className={cn(
                    "w-full flex items-center space-x-3 px-4 py-3 rounded-lg transition-all duration-200",
                    "relative overflow-hidden",
                    isActive 
                      ? "bg-primary text-primary-foreground shadow-lg shadow-primary/25" 
                      : "text-muted-foreground hover:bg-accent hover:text-accent-foreground"
                  )}
                >
                  {/* Background gradient effect */}
                  {isActive && (
                    <motion.div
                      layoutId="activeTab"
                      className="absolute inset-0 bg-gradient-to-r from-blue-500/20 to-purple-600/20"
                      transition={{ type: "spring", bounce: 0.2, duration: 0.6 }}
                    />
                  )}
                  
                  <Icon className="w-5 h-5 relative z-10" />
                  <div className="flex-1 text-left relative z-10">
                    <div className="font-medium">{item.label}</div>
                    <AnimatePresence>
                      {isHovered === item.id && (
                        <motion.div
                          initial={{ opacity: 0, height: 0 }}
                          animate={{ opacity: 1, height: "auto" }}
                          exit={{ opacity: 0, height: 0 }}
                          className="text-xs opacity-70"
                        >
                          {item.description}
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </div>
                  
                  {/* Active indicator */}
                  {isActive && (
                    <motion.div
                      layoutId="activeIndicator"
                      className="w-2 h-2 bg-primary-foreground rounded-full"
                      transition={{ type: "spring", bounce: 0.2, duration: 0.6 }}
                    />
                  )}
                </motion.button>
              )
            })}
          </div>

          <div className="pt-4 mt-4 border-t border-border space-y-1">
            {secondaryItems.map((item) => {
              const Icon = item.icon
              return (
                <motion.button
                  key={item.id}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  className="w-full flex items-center space-x-3 px-4 py-2 rounded-lg text-muted-foreground hover:bg-accent hover:text-accent-foreground transition-all duration-200"
                >
                  <Icon className="w-4 h-4" />
                  <span className="text-sm">{item.label}</span>
                </motion.button>
              )
            })}
          </div>
        </nav>

        {/* Footer */}
        <div className="p-4 border-t border-border">
          <div className="space-y-2">
            <div className="flex items-center justify-between text-xs text-muted-foreground">
              <span>System Status</span>
              <div className="flex items-center space-x-1">
                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                <span>Online</span>
              </div>
            </div>
            <div className="flex items-center justify-between text-xs text-muted-foreground">
              <span>Version</span>
              <span>2.0.0</span>
            </div>
          </div>
        </div>
      </motion.div>
    </>
  )
}
