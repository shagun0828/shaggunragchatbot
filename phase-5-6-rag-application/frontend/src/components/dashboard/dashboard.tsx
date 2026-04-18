'use client'

import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  TrendingUp, 
  Users, 
  Activity, 
  Clock, 
  AlertTriangle, 
  CheckCircle,
  BarChart3,
  PieChart,
  Zap,
  Database,
  Brain,
  RefreshCw
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { cn } from '@/lib/utils'

interface MetricCard {
  title: string
  value: string
  change: number
  changeType: 'increase' | 'decrease'
  icon: React.ElementType
  color: string
}

interface AlertItem {
  id: string
  severity: 'low' | 'medium' | 'high' | 'critical'
  message: string
  component: string
  timestamp: number
}

export function Dashboard() {
  const [metrics, setMetrics] = useState<MetricCard[]>([])
  const [alerts, setAlerts] = useState<AlertItem[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [selectedTimeRange, setSelectedTimeRange] = useState('24h')
  const [isRefreshing, setIsRefreshing] = useState(false)

  useEffect(() => {
    loadDashboardData()
    const interval = setInterval(loadDashboardData, 30000) // Refresh every 30 seconds
    return () => clearInterval(interval)
  }, [])

  const loadDashboardData = async () => {
    setIsLoading(true)
    
    // Simulate API call
    setTimeout(() => {
      setMetrics([
        {
          title: 'Total Queries',
          value: '1,247',
          change: 12.5,
          changeType: 'increase',
          icon: TrendingUp,
          color: 'text-blue-500'
        },
        {
          title: 'Active Users',
          value: '89',
          change: 8.3,
          changeType: 'increase',
          icon: Users,
          color: 'text-green-500'
        },
        {
          title: 'Avg Response Time',
          value: '0.85s',
          change: -5.2,
          changeType: 'decrease',
          icon: Clock,
          color: 'text-orange-500'
        },
        {
          title: 'Success Rate',
          value: '98.7%',
          change: 2.1,
          changeType: 'increase',
          icon: CheckCircle,
          color: 'text-green-500'
        }
      ])

      setAlerts([
        {
          id: '1',
          severity: 'medium',
          message: 'High CPU usage detected on Chroma client',
          component: 'chroma_client',
          timestamp: Date.now() / 1000 - 300
        },
        {
          id: '2',
          severity: 'low',
          message: 'Response time increased for search endpoint',
          component: 'search_api',
          timestamp: Date.now() / 1000 - 600
        },
        {
          id: '3',
          severity: 'high',
          message: 'Memory usage approaching threshold',
          component: 'system',
          timestamp: Date.now() / 1000 - 900
        }
      ])

      setIsLoading(false)
    }, 1000)
  }

  const handleRefresh = async () => {
    setIsRefreshing(true)
    await loadDashboardData()
    setIsRefreshing(false)
  }

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return 'bg-red-500'
      case 'high': return 'bg-orange-500'
      case 'medium': return 'bg-yellow-500'
      case 'low': return 'bg-green-500'
      default: return 'bg-gray-500'
    }
  }

  const timeRanges = ['1h', '6h', '24h', '7d', '30d']

  return (
    <div className="flex flex-col h-full bg-background">
      {/* Header */}
      <div className="p-6 border-b border-border">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-foreground">Analytics Dashboard</h1>
            <p className="text-muted-foreground">Real-time system metrics and performance analytics</p>
          </div>
          
          <div className="flex items-center space-x-4">
            {/* Time range selector */}
            <div className="flex items-center space-x-2 bg-accent/50 rounded-lg p-1">
              {timeRanges.map((range) => (
                <button
                  key={range}
                  onClick={() => setSelectedTimeRange(range)}
                  className={cn(
                    "px-3 py-1 rounded text-sm font-medium transition-colors",
                    selectedTimeRange === range
                      ? "bg-primary text-primary-foreground"
                      : "text-muted-foreground hover:text-foreground"
                  )}
                >
                  {range}
                </button>
              ))}
            </div>

            {/* Refresh button */}
            <Button
              variant="outline"
              onClick={handleRefresh}
              disabled={isRefreshing}
              className="flex items-center space-x-2"
            >
              <RefreshCw className={cn("w-4 h-4", isRefreshing && "animate-spin")} />
              <span>Refresh</span>
            </Button>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 overflow-y-auto p-6">
        <div className="max-w-7xl mx-auto space-y-6">
          {/* Metrics Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <AnimatePresence>
              {metrics.map((metric, index) => (
                <motion.div
                  key={metric.title}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.1 }}
                >
                  <Card className="p-6 hover:shadow-lg transition-shadow duration-200">
                    <div className="flex items-center justify-between mb-4">
                      <div className="p-2 bg-accent/50 rounded-lg">
                        <metric.icon className={cn("w-6 h-6", metric.color)} />
                      </div>
                      <div className={cn(
                        "flex items-center space-x-1 text-sm font-medium",
                        metric.changeType === 'increase' ? "text-green-500" : "text-red-500"
                      )}>
                        {metric.changeType === 'increase' ? (
                          <TrendingUp className="w-4 h-4" />
                        ) : (
                          <TrendingUp className="w-4 h-4 rotate-180" />
                        )}
                        <span>{Math.abs(metric.change)}%</span>
                      </div>
                    </div>
                    
                    <div>
                      <div className="text-2xl font-bold text-foreground mb-1">
                        {metric.value}
                      </div>
                      <div className="text-sm text-muted-foreground">
                        {metric.title}
                      </div>
                    </div>
                  </Card>
                </motion.div>
              ))}
            </AnimatePresence>
          </div>

          {/* Charts Row */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Query Volume Chart */}
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.4 }}
            >
              <Card className="p-6">
                <div className="flex items-center justify-between mb-6">
                  <h3 className="text-lg font-semibold text-foreground">Query Volume</h3>
                  <BarChart3 className="w-5 h-5 text-muted-foreground" />
                </div>
                
                <div className="h-64 bg-accent/50 rounded-lg flex items-center justify-center">
                  <div className="text-center">
                    <BarChart3 className="w-12 h-12 text-muted-foreground mx-auto mb-2" />
                    <p className="text-muted-foreground">Query volume chart</p>
                    <p className="text-sm text-muted-foreground">Last {selectedTimeRange}</p>
                  </div>
                </div>
              </Card>
            </motion.div>

            {/* Response Time Distribution */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.5 }}
            >
              <Card className="p-6">
                <div className="flex items-center justify-between mb-6">
                  <h3 className="text-lg font-semibold text-foreground">Response Time Distribution</h3>
                  <PieChart className="w-5 h-5 text-muted-foreground" />
                </div>
                
                <div className="h-64 bg-accent/50 rounded-lg flex items-center justify-center">
                  <div className="text-center">
                    <PieChart className="w-12 h-12 text-muted-foreground mx-auto mb-2" />
                    <p className="text-muted-foreground">Response time distribution</p>
                    <p className="text-sm text-muted-foreground">P50: 0.6s, P95: 1.2s, P99: 2.1s</p>
                  </div>
                </div>
              </Card>
            </motion.div>
          </div>

          {/* System Status */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Component Health */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.6 }}
            >
              <Card className="p-6">
                <div className="flex items-center justify-between mb-6">
                  <h3 className="text-lg font-semibold text-foreground">Component Health</h3>
                  <Activity className="w-5 h-5 text-muted-foreground" />
                </div>
                
                <div className="space-y-4">
                  {[
                    { name: 'Chroma Client', status: 'healthy', icon: Database },
                    { name: 'LLM Service', status: 'healthy', icon: Brain },
                    { name: 'WebSocket', status: 'healthy', icon: Zap },
                    { name: 'API Gateway', status: 'degraded', icon: Activity }
                  ].map((component, index) => (
                    <div key={component.name} className="flex items-center justify-between">
                      <div className="flex items-center space-x-3">
                        <component.icon className="w-4 h-4 text-muted-foreground" />
                        <span className="text-sm text-foreground">{component.name}</span>
                      </div>
                      <div className={cn(
                        "w-2 h-2 rounded-full",
                        component.status === 'healthy' ? "bg-green-500" : 
                        component.status === 'degraded' ? "bg-yellow-500" : "bg-red-500"
                      )} />
                    </div>
                  ))}
                </div>
              </Card>
            </motion.div>

            {/* Alerts */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.7 }}
              className="lg:col-span-2"
            >
              <Card className="p-6">
                <div className="flex items-center justify-between mb-6">
                  <h3 className="text-lg font-semibold text-foreground">System Alerts</h3>
                  <AlertTriangle className="w-5 h-5 text-muted-foreground" />
                </div>
                
                <div className="space-y-3">
                  <AnimatePresence>
                    {alerts.map((alert) => (
                      <motion.div
                        key={alert.id}
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        exit={{ opacity: 0, x: 20 }}
                        className="flex items-center justify-between p-3 bg-accent/50 rounded-lg"
                      >
                        <div className="flex items-center space-x-3">
                          <div className={cn(
                            "w-3 h-3 rounded-full",
                            getSeverityColor(alert.severity)
                          )} />
                          <div>
                            <p className="text-sm text-foreground">{alert.message}</p>
                            <p className="text-xs text-muted-foreground">
                              {alert.component} · {new Date(alert.timestamp * 1000).toLocaleTimeString()}
                            </p>
                          </div>
                        </div>
                        <Button variant="ghost" size="icon" className="w-6 h-6">
                          ×
                        </Button>
                      </motion.div>
                    ))}
                  </AnimatePresence>
                </div>
              </Card>
            </motion.div>
          </div>
        </div>
      </div>
    </div>
  )
}
