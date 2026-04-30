'use client'

import { useState } from 'react'
import { ChatInterface } from '@/components/chat/chat-interface'
import { SearchInterface } from '@/components/search/search-interface'
import { Dashboard } from '@/components/dashboard/dashboard'

export default function Home() {
  const [showApp, setShowApp] = useState(false)
  const [activeView, setActiveView] = useState<'chat' | 'search' | 'dashboard'>('chat')

  if (showApp) {
    return (
      <div className="flex h-screen bg-background">
        {/* Minimal top nav */}
        <div className="fixed top-0 left-0 right-0 z-50 h-14 bg-card/80 backdrop-blur border-b border-border flex items-center px-6 gap-6">
          <button
            onClick={() => setShowApp(false)}
            className="font-semibold text-primary text-lg"
          >
            RAG Assistant
          </button>
          <div className="flex gap-1 ml-4">
            {(['chat', 'search', 'dashboard'] as const).map((view) => (
              <button
                key={view}
                onClick={() => setActiveView(view)}
                className={`px-4 py-1.5 rounded-full text-sm capitalize transition-all ${activeView === view
                    ? 'bg-primary text-primary-foreground'
                    : 'text-muted-foreground hover:text-foreground hover:bg-muted'
                  }`}
              >
                {view}
              </button>
            ))}
          </div>
        </div>

        {/* Main content */}
        <main className="flex-1 pt-14 overflow-hidden">
          {activeView === 'chat' && <ChatInterface />}
          {activeView === 'search' && <SearchInterface />}
          {activeView === 'dashboard' && <Dashboard />}
        </main>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-background">
      {/* Floating Navbar */}
      <header className="fixed top-4 left-1/2 -translate-x-1/2 z-50 bg-card/90 backdrop-blur border border-border rounded-full px-6 py-2.5 flex items-center gap-6 shadow-sm">
        <span className="font-bold text-foreground text-sm">RAG Assistant</span>
        <nav className="flex items-center gap-5 text-sm text-muted-foreground">
          <a href="#features" className="hover:text-foreground transition-colors">Features</a>
          <a href="#how" className="hover:text-foreground transition-colors">How it works</a>
          <a href="#trust" className="hover:text-foreground transition-colors">Trust</a>
        </nav>
        <button
          onClick={() => setShowApp(true)}
          className="bg-primary text-primary-foreground text-sm px-4 py-1.5 rounded-full hover:opacity-90 transition-opacity font-medium"
        >
          Ask a question
        </button>
      </header>

      {/* Hero */}
      <section className="pt-40 pb-20 px-6 text-center max-w-4xl mx-auto">
        <div className="inline-flex items-center gap-2 bg-primary/10 text-primary text-xs font-medium px-3 py-1 rounded-full mb-8">
          <span className="w-1.5 h-1.5 rounded-full bg-primary inline-block"></span>
          AI-Powered Knowledge Assistant
        </div>
        <h1 className="text-5xl md:text-6xl font-bold text-foreground leading-tight mb-6">
          Get Instant Answers to<br />
          <span className="text-primary">Your Questions</span>
        </h1>
        <p className="text-lg text-muted-foreground max-w-2xl mx-auto mb-10">
          Ask anything about your documents — powered by RAG AI with verified citations from official sources.
        </p>
        <div className="flex items-center justify-center gap-4">
          <button
            onClick={() => setShowApp(true)}
            className="bg-primary text-primary-foreground px-8 py-3 rounded-full font-medium hover:opacity-90 transition-opacity flex items-center gap-2"
          >
            Start asking →
          </button>
          <a href="#how" className="border border-border text-foreground px-8 py-3 rounded-full font-medium hover:bg-muted transition-colors">
            Learn more
          </a>
        </div>

        {/* Stats ticker */}
        <div className="flex items-center justify-center gap-8 mt-12 text-sm">
          {[
            { value: '10ms', label: 'Avg response time' },
            { value: '100%', label: 'Cited answers' },
            { value: '24/7', label: 'Always available' },
          ].map((stat) => (
            <div key={stat.label} className="flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-primary inline-block"></span>
              <span className="font-semibold text-foreground">{stat.value}</span>
              <span className="text-muted-foreground">{stat.label}</span>
            </div>
          ))}
        </div>
      </section>

      {/* Stats cards */}
      <section className="max-w-3xl mx-auto px-6 mb-24">
        <div className="bg-card border border-border rounded-2xl p-8 grid grid-cols-3 divide-x divide-border">
          {[
            { value: '< 1s', label: 'Average response time' },
            { value: '100%', label: 'Cited answers' },
            { value: '∞', label: 'Documents supported' },
          ].map((s) => (
            <div key={s.label} className="text-center px-6">
              <div className="text-3xl font-bold text-foreground mb-1">{s.value}</div>
              <div className="text-sm text-muted-foreground">{s.label}</div>
            </div>
          ))}
        </div>
      </section>

      {/* Features */}
      <section id="features" className="max-w-5xl mx-auto px-6 mb-24">
        <div className="text-center mb-12">
          <div className="text-xs font-semibold text-primary uppercase tracking-widest mb-3">— Features</div>
          <h2 className="text-4xl font-bold text-foreground">Everything you need,<br /><span className="text-primary">nothing you don't</span></h2>
          <p className="text-muted-foreground mt-4">Zero fluff. Only verified facts delivered in seconds.</p>
        </div>
        <div className="grid grid-cols-3 gap-4">
          {[
            { icon: '⚡', title: 'Instant Answers', desc: 'Get precise answers in under a second, sourced directly from your documents.' },
            { icon: '📎', title: 'Cited Sources', desc: 'Every answer comes with a verifiable citation so you can check the source yourself.' },
            { icon: '🔍', title: 'Semantic Search', desc: 'Advanced search that understands meaning, not just keywords.' },
            { icon: '📊', title: 'Analytics Dashboard', desc: 'Track queries, response times, and system performance in real time.' },
            { icon: '🔒', title: 'Safe & Transparent', desc: 'No hallucinations — if we don\'t know, we say so.' },
            { icon: '🌐', title: 'Always Online', desc: '24/7 availability with real-time document processing.' },
          ].map((f) => (
            <div key={f.title} className="bg-card border border-border rounded-2xl p-6 hover:border-primary/30 transition-colors">
              <div className="w-10 h-10 bg-muted rounded-xl flex items-center justify-center text-xl mb-4">{f.icon}</div>
              <h3 className="font-semibold text-foreground mb-2">{f.title}</h3>
              <p className="text-sm text-muted-foreground leading-relaxed">{f.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* How it works */}
      <section id="how" className="bg-muted/40 py-24 mb-24">
        <div className="max-w-5xl mx-auto px-6">
          <div className="text-center mb-12">
            <div className="text-xs font-semibold text-primary uppercase tracking-widest mb-3">— How it works</div>
            <h2 className="text-4xl font-bold text-foreground">Three steps to<br /><span className="text-primary">verified answers</span></h2>
          </div>
          <div className="grid grid-cols-3 gap-6">
            {[
              { num: '01', title: 'Ask any question', desc: 'Type any question about your documents — our AI understands natural language.' },
              { num: '02', title: 'Get a cited answer', desc: 'Receive a precise answer backed by the exact source document with full citation.' },
              { num: '03', title: 'Verify the source', desc: 'Click the citation to view the original document yourself — full transparency.' },
            ].map((step) => (
              <div key={step.num} className="bg-card border border-border rounded-2xl p-6">
                <div className="w-10 h-10 bg-primary rounded-full flex items-center justify-center text-primary-foreground text-sm font-bold mb-4">{step.num}</div>
                <h3 className="font-semibold text-foreground mb-2">{step.title}</h3>
                <p className="text-sm text-muted-foreground leading-relaxed">{step.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Trust */}
      <section id="trust" className="max-w-5xl mx-auto px-6 mb-24">
        <div className="grid grid-cols-2 gap-12 items-center">
          <div>
            <div className="text-xs font-semibold text-primary uppercase tracking-widest mb-3">— Trust & Safety</div>
            <h2 className="text-4xl font-bold text-foreground mb-4">Built for <span className="text-primary">accuracy</span></h2>
            <p className="text-muted-foreground leading-relaxed">We don't hallucinate. If we don't know, we say so. Every answer comes with a verifiable citation from the source document.</p>
          </div>
          <div className="space-y-3">
            {[
              'Every answer cites the original source document',
              'If information isn\'t in the database, we tell you',
              'Real-time document processing and indexing',
              'Zero guesswork — only verified facts',
            ].map((item) => (
              <div key={item} className="flex items-start gap-3 bg-card border border-border rounded-xl p-4">
                <div className="w-5 h-5 rounded-full bg-primary/15 flex items-center justify-center flex-shrink-0 mt-0.5">
                  <span className="text-primary text-xs">✓</span>
                </div>
                <p className="text-sm text-foreground">{item}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="max-w-3xl mx-auto px-6 mb-24">
        <div className="bg-primary/10 border border-primary/20 rounded-3xl p-12 text-center">
          <h2 className="text-3xl font-bold text-foreground mb-4">Ready to get instant answers?</h2>
          <p className="text-muted-foreground mb-8">Ask anything — all backed by official sources.</p>
          <button
            onClick={() => setShowApp(true)}
            className="bg-primary text-primary-foreground px-10 py-3.5 rounded-full font-medium hover:opacity-90 transition-opacity"
          >
            Open RAG Assistant →
          </button>
        </div>
      </section>

      {/* Footer ticker */}
      <div className="border-t border-border py-4">
        <div className="flex items-center justify-center gap-8 text-xs text-muted-foreground">
          {['Official Sources', 'Verified Answers', 'Real-time Processing', 'No Data Stored', 'Always Available'].map((item) => (
            <span key={item} className="flex items-center gap-1.5">
              <span className="w-1 h-1 rounded-full bg-primary inline-block"></span>
              {item}
            </span>
          ))}
        </div>
      </div>
    </div>
  )
}