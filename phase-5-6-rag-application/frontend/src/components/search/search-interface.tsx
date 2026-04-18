'use client'

import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Search, Filter, Sparkles, TrendingUp, Clock, Star } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { cn } from '@/lib/utils'

interface SearchResult {
  id: string
  title: string
  content: string
  score: number
  highlights: string[]
  metadata: {
    source: string
    date: string
    type: string
    fundName?: string
    category?: string
  }
}

export function SearchInterface() {
  const [query, setQuery] = useState('')
  const [isSearching, setIsSearching] = useState(false)
  const [results, setResults] = useState<SearchResult[]>([])
  const [filters, setFilters] = useState({
    searchType: 'semantic',
    fundType: '',
    riskLevel: '',
    timeRange: ''
  })
  const [showFilters, setShowFilters] = useState(false)
  const [suggestions, setSuggestions] = useState<string[]>([])

  const handleSearch = async () => {
    if (!query.trim()) return

    setIsSearching(true)
    
    // Simulate search API call
    setTimeout(() => {
      const mockResults: SearchResult[] = [
        {
          id: '1',
          title: 'HDFC Mid-Cap Fund Analysis',
          content: 'The HDFC Mid-Cap Fund Direct Growth is a popular mid-cap equity fund that invests in companies with market capitalization between the 101st and 250th largest companies...',
          score: 0.95,
          highlights: ['HDFC Mid-Cap Fund', 'mid-cap equity', 'Direct Growth plan'],
          metadata: {
            source: 'Mutual Fund Database',
            date: '2024-04-18',
            type: 'fund_analysis',
            fundName: 'HDFC Mid-Cap Fund',
            category: 'mid-cap'
          }
        },
        {
          id: '2',
          title: 'Understanding Mid-Cap Investments',
          content: 'Mid-cap funds offer a balance between the growth potential of small-cap funds and the stability of large-cap funds...',
          score: 0.87,
          highlights: ['mid-cap investments', 'growth potential', 'risk balance'],
          metadata: {
            source: 'Investment Guide',
            date: '2024-04-15',
            type: 'educational_content'
          }
        },
        {
          id: '3',
          title: 'Top Performing Mid-Cap Funds 2024',
          content: 'Analysis of the best performing mid-cap mutual funds in 2024, including performance metrics and risk assessment...',
          score: 0.82,
          highlights: ['top performing funds', '2024 analysis', 'performance metrics'],
          metadata: {
            source: 'Market Analysis',
            date: '2024-04-10',
            type: 'market_analysis'
          }
        }
      ]
      
      setResults(mockResults)
      setIsSearching(false)
    }, 1500)
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSearch()
    }
  }

  useEffect(() => {
    if (query.length > 2) {
      // Generate suggestions based on query
      const mockSuggestions = [
        `${query} mutual funds`,
        `${query} investment strategy`,
        `${query} performance analysis`,
        `${query} risk assessment`
      ]
      setSuggestions(mockSuggestions.slice(0, 3))
    } else {
      setSuggestions([])
    }
  }, [query])

  return (
    <div className="flex flex-col h-full bg-background">
      {/* Search Header */}
      <div className="p-6 border-b border-border">
        <div className="max-w-4xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-center mb-6"
          >
            <h1 className="text-3xl font-bold text-foreground mb-2">
              Semantic Search
            </h1>
            <p className="text-muted-foreground">
              AI-enhanced search with intelligent ranking and contextual understanding
            </p>
          </motion.div>

          {/* Search Input */}
          <div className="relative">
            <div className="flex items-center space-x-2">
              <div className="relative flex-1">
                <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 w-5 h-5 text-muted-foreground" />
                <input
                  type="text"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="Search for mutual funds, investment advice, financial information..."
                  className="w-full pl-12 pr-4 py-4 bg-accent/50 border border-border rounded-lg text-lg focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent transition-all duration-200"
                />
                {query && (
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={() => setQuery('')}
                    className="absolute right-2 top-1/2 transform -translate-y-1/2"
                  >
                    ×
                  </Button>
                )}
              </div>
              
              <Button
                onClick={handleSearch}
                disabled={!query.trim() || isSearching}
                className="px-6"
              >
                {isSearching ? (
                  <motion.div
                    animate={{ rotate: 360 }}
                    transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                  >
                    <Sparkles className="w-5 h-5" />
                  </motion.div>
                ) : (
                  <Search className="w-5 h-5" />
                )}
              </Button>
              
              <Button
                variant="outline"
                onClick={() => setShowFilters(!showFilters)}
                className="px-4"
              >
                <Filter className="w-4 h-4 mr-2" />
                Filters
              </Button>
            </div>

            {/* Suggestions */}
            <AnimatePresence>
              {suggestions.length > 0 && (
                <motion.div
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  className="absolute top-full left-0 right-0 mt-2 bg-card border border-border rounded-lg shadow-lg z-10"
                >
                  {suggestions.map((suggestion, index) => (
                    <button
                      key={index}
                      onClick={() => setQuery(suggestion)}
                      className="w-full text-left px-4 py-3 hover:bg-accent transition-colors first:rounded-t-lg last:rounded-b-lg"
                    >
                      <div className="flex items-center space-x-2">
                        <Search className="w-4 h-4 text-muted-foreground" />
                        <span>{suggestion}</span>
                      </div>
                    </button>
                  ))}
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* Advanced Filters */}
          <AnimatePresence>
            {showFilters && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                className="mt-4 p-4 bg-accent/30 rounded-lg border border-border"
              >
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div>
                    <label className="block text-sm font-medium mb-2">Search Type</label>
                    <select
                      value={filters.searchType}
                      onChange={(e) => setFilters({...filters, searchType: e.target.value})}
                      className="w-full px-3 py-2 bg-background border border-border rounded-md"
                    >
                      <option value="semantic">Semantic Search</option>
                      <option value="keyword">Keyword Search</option>
                      <option value="hybrid">Hybrid Search</option>
                    </select>
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium mb-2">Fund Type</label>
                    <select
                      value={filters.fundType}
                      onChange={(e) => setFilters({...filters, fundType: e.target.value})}
                      className="w-full px-3 py-2 bg-background border border-border rounded-md"
                    >
                      <option value="">All Types</option>
                      <option value="equity">Equity</option>
                      <option value="debt">Debt</option>
                      <option value="hybrid">Hybrid</option>
                    </select>
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium mb-2">Time Range</label>
                    <select
                      value={filters.timeRange}
                      onChange={(e) => setFilters({...filters, timeRange: e.target.value})}
                      className="w-full px-3 py-2 bg-background border border-border rounded-md"
                    >
                      <option value="">All Time</option>
                      <option value="recent">Recent</option>
                      <option value="1year">Last Year</option>
                      <option value="3years">Last 3 Years</option>
                    </select>
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>

      {/* Search Results */}
      <div className="flex-1 overflow-y-auto p-6">
        <div className="max-w-4xl mx-auto">
          {isSearching ? (
            <div className="flex flex-col items-center justify-center py-20">
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                className="w-12 h-12 bg-primary rounded-full flex items-center justify-center mb-4"
              >
                <Sparkles className="w-6 h-6 text-primary-foreground" />
              </motion.div>
              <p className="text-muted-foreground">Searching for relevant information...</p>
            </div>
          ) : results.length > 0 ? (
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <h2 className="text-xl font-semibold text-foreground">
                  Found {results.length} results
                </h2>
                <div className="flex items-center space-x-2 text-sm text-muted-foreground">
                  <Clock className="w-4 h-4" />
                  <span>~1.5s</span>
                </div>
              </div>

              <AnimatePresence>
                {results.map((result, index) => (
                  <motion.div
                    key={result.id}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: index * 0.1 }}
                  >
                    <Card className="p-6 hover:shadow-lg transition-shadow duration-200">
                      <div className="flex items-start justify-between mb-3">
                        <div className="flex-1">
                          <h3 className="text-lg font-semibold text-foreground mb-2">
                            {result.title}
                          </h3>
                          <p className="text-muted-foreground line-clamp-3">
                            {result.content}
                          </p>
                        </div>
                        <div className="ml-4 flex flex-col items-center">
                          <div className="w-16 h-16 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full flex items-center justify-center text-white font-bold">
                            {(result.score * 100).toFixed(0)}%
                          </div>
                          <div className="text-xs text-muted-foreground mt-1">Match</div>
                        </div>
                      </div>

                      {/* Highlights */}
                      {result.highlights.length > 0 && (
                        <div className="mb-3 p-3 bg-accent/50 rounded-lg">
                          <div className="text-sm font-medium text-foreground mb-2">Key Highlights:</div>
                          <div className="flex flex-wrap gap-2">
                            {result.highlights.map((highlight, index) => (
                              <span
                                key={index}
                                className="px-2 py-1 bg-primary/20 text-primary text-xs rounded-full"
                              >
                                {highlight}
                              </span>
                            ))}
                          </div>
                        </div>
                      )}

                      {/* Metadata */}
                      <div className="flex items-center justify-between text-sm text-muted-foreground">
                        <div className="flex items-center space-x-4">
                          <span>Source: {result.metadata.source}</span>
                          <span>Date: {result.metadata.date}</span>
                          <span>Type: {result.metadata.type}</span>
                        </div>
                        <div className="flex items-center space-x-2">
                          <Star className="w-4 h-4" />
                          <span>Save</span>
                        </div>
                      </div>
                    </Card>
                  </motion.div>
                ))}
              </AnimatePresence>
            </div>
          ) : query ? (
            <div className="flex flex-col items-center justify-center py-20">
              <div className="text-center">
                <Search className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
                <h3 className="text-lg font-semibold text-foreground mb-2">
                  No results found
                </h3>
                <p className="text-muted-foreground mb-4">
                  Try adjusting your search terms or filters
                </p>
                <div className="space-y-2">
                  <p className="text-sm text-muted-foreground">Try searching for:</p>
                  <div className="flex flex-wrap gap-2 justify-center">
                    {['mutual funds', 'investment strategy', 'risk assessment', 'portfolio diversification'].map((term) => (
                      <button
                        key={term}
                        onClick={() => setQuery(term)}
                        className="px-3 py-1 bg-accent hover:bg-accent/80 rounded-full text-sm transition-colors"
                      >
                        {term}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center py-20">
              <div className="text-center">
                <Search className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
                <h3 className="text-lg font-semibold text-foreground mb-2">
                  Start Searching
                </h3>
                <p className="text-muted-foreground">
                  Enter a query above to search through our financial knowledge base
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
