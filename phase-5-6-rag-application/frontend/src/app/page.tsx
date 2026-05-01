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
        <div className="fixed top-0 left-0 right-0 z-50 h-14 bg-white/90 backdrop-blur border-b border-border flex items-center px-6 gap-6">
          <button
            onClick={() => setShowApp(false)}
            className="font-bold text-foreground text-base tracking-tight"
          >
            Nova<span className="text-primary">Docs</span>
          </button>
          <div className="flex gap-1 ml-4">
            {(['chat', 'search', 'dashboard'] as const).map((view) => (
              <button
                key={view}
                onClick={() => setActiveView(view)}
                className={`px-4 py-1.5 rounded-full text-sm capitalize transition-all ${activeView === view
                    ? 'bg-primary text-white'
                    : 'text-muted-foreground hover:text-foreground hover:bg-accent'
                  }`}
              >
                {view}
              </button>
            ))}
          </div>
        </div>
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

      {/* Navbar */}
      <header className="sticky top-0 z-50 bg-white border-b border-border">
        <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
          <span className="text-base font-bold tracking-tight">
            Nova<span className="text-primary">Docs</span>
          </span>
          <nav className="flex items-center gap-6 text-sm text-muted-foreground font-medium">
            <a href="#features" className="hover:text-foreground transition-colors">Features</a>
            <a href="#how" className="hover:text-foreground transition-colors">How it works</a>
            <a href="#trust" className="hover:text-foreground transition-colors">Trust</a>
          </nav>
          <button
            onClick={() => setShowApp(true)}
            className="bg-primary text-white text-sm px-5 py-2 rounded-full font-semibold hover:opacity-90 transition-opacity"
          >
            Ask a question
          </button>
        </div>
      </header>

      {/* Hero */}
      <section className="max-w-6xl mx-auto px-6 pt-20 pb-16">
        <div className="inline-flex items-center gap-2 bg-accent text-accent-foreground text-xs font-semibold px-4 py-2 rounded-full mb-8">
          <span className="w-1.5 h-1.5 rounded-full bg-primary inline-block"></span>
          AI Mutual Fund Intelligence
        </div>
        <h1 className="text-5xl md:text-6xl font-extrabold tracking-tight leading-[1.08] text-foreground mb-6">
          Mutual fund answers,<br />
          <span className="text-primary">instantly sourced.</span>
        </h1>
        <p className="text-lg text-muted-foreground leading-relaxed max-w-xl mb-10">
          NovaDocs answers your toughest mutual fund questions — SIP, NAV, ELSS, expense ratios — with precision and full citations. No fluff, no guesswork.
        </p>
        <div className="flex items-center gap-4 mb-14">
          <button
            onClick={() => setShowApp(true)}
            className="bg-primary text-white px-8 py-3.5 rounded-xl font-bold text-sm hover:opacity-90 transition-opacity"
          >
            Ask your first question →
          </button>
          <a href="#how" className="text-muted-foreground text-sm font-medium underline underline-offset-4 hover:text-foreground transition-colors">
            See how it works
          </a>
        </div>

        {/* Stats */}
        <div className="flex items-center gap-0 border-t border-border pt-8">
          {[
            { value: '2s', label: 'Avg response time' },
            { value: '100%', label: 'Cited answers' },
            { value: '24/7', label: 'Always available' },
          ].map((stat, i) => (
            <div key={stat.label} className={`pr-10 mr-10 ${i < 2 ? 'border-r border-border' : ''}`}>
              <div className="text-3xl font-extrabold tracking-tight text-foreground">{stat.value}</div>
              <div className="text-xs text-muted-foreground mt-1 font-medium">{stat.label}</div>
            </div>
          ))}
        </div>
      </section>

      {/* Ticker */}
      <div className="bg-primary overflow-hidden">
        <div className="flex items-center">
          <div className="bg-violet-700 text-white text-xs font-bold px-5 py-2.5 tracking-widest uppercase whitespace-nowrap border-r border-violet-500">
            Live
          </div>
          {['SIP Minimums', 'Expense Ratios', 'ELSS Lock-in', 'NAV Queries', 'Exit Loads', 'Fund Comparisons', 'Capital Gains'].map((item) => (
            <div key={item} className="text-white/90 text-xs font-medium px-5 py-2.5 border-r border-violet-400 whitespace-nowrap">
              {item}
            </div>
          ))}
        </div>
      </div>

      {/* Features */}
      <section id="features" className="max-w-6xl mx-auto px-6 py-20">
        <div className="mb-12">
          <div className="text-xs font-bold text-primary uppercase tracking-widest mb-3">— Features</div>
          <h2 className="text-4xl font-extrabold tracking-tight text-foreground">
            Everything you need,<br />
            <span className="text-primary">nothing you don't.</span>
          </h2>
          <p className="text-muted-foreground mt-4 text-base max-w-lg">Zero fluff. Only verified facts from official sources, delivered in seconds.</p>
        </div>
        <div className="grid grid-cols-3 gap-4">
          {[
            { icon: '⚡', title: 'Instant Answers', desc: 'Precise fund data delivered in under 2 seconds. No loading screens, no delays.' },
            { icon: '📎', title: 'Always Cited', desc: 'Every answer links back to the original AMC or SEBI document. Verify yourself.' },
            { icon: '🔍', title: 'Smart Search', desc: 'Understands what you mean, not just what you type. Semantic AI search.' },
            { icon: '📊', title: 'Analytics Dashboard', desc: 'Track query history, response times, and system health in real time.' },
            { icon: '🛡️', title: 'No Hallucinations', desc: "If we don't have the answer, we tell you — no guessing, no faking." },
            { icon: '🌐', title: 'Always Online', desc: '24/7 uptime with real-time document processing and indexing.' },
          ].map((f) => (
            <div key={f.title} className="bg-white border border-border rounded-2xl p-6 hover:border-primary/40 hover:shadow-sm transition-all">
              <div className="w-9 h-9 bg-accent rounded-xl flex items-center justify-center text-lg mb-4">{f.icon}</div>
              <h3 className="font-bold text-foreground text-sm mb-2">{f.title}</h3>
              <p className="text-xs text-muted-foreground leading-relaxed">{f.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* How it works */}
      <section id="how" className="bg-accent/30 border-y border-border py-20">
        <div className="max-w-6xl mx-auto px-6">
          <div className="text-center mb-14">
            <div className="text-xs font-bold text-primary uppercase tracking-widest mb-3">— How it works</div>
            <h2 className="text-4xl font-extrabold tracking-tight text-foreground">
              Three steps to <span className="text-primary">verified answers</span>
            </h2>
          </div>
          <div className="grid grid-cols-3 gap-6">
            {[
              { num: '01', title: 'Ask any question', desc: 'Type any mutual fund question in plain English. Our AI understands context and intent.' },
              { num: '02', title: 'Get a cited answer', desc: 'Receive a precise answer backed by an official AMC, SEBI, or AMFI document with a citation.' },
              { num: '03', title: 'Verify the source', desc: 'Click the citation to open the original document. Full transparency, zero guesswork.' },
            ].map((step) => (
              <div key={step.num} className="bg-white border border-border rounded-2xl p-7">
                <div className="w-10 h-10 bg-primary rounded-full flex items-center justify-center text-white text-sm font-bold mb-5">{step.num}</div>
                <h3 className="font-bold text-foreground mb-2">{step.title}</h3>
                <p className="text-sm text-muted-foreground leading-relaxed">{step.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Trust */}
      <section id="trust" className="max-w-6xl mx-auto px-6 py-20">
        <div className="grid grid-cols-2 gap-16 items-center">
          <div>
            <div className="text-xs font-bold text-primary uppercase tracking-widest mb-3">— Trust & Safety</div>
            <h2 className="text-4xl font-extrabold tracking-tight text-foreground mb-5">
              Built for <span className="text-primary">accuracy.</span>
            </h2>
            <p className="text-muted-foreground leading-relaxed text-base">
              We don't hallucinate. If we don't know, we say so. Every answer comes with a verifiable citation directly from the source document.
            </p>
          </div>
          <div className="space-y-3">
            {[
              'Every answer cites the original AMC or SEBI source document',
              "If information isn't in our database, we tell you honestly",
              'Real-time document processing — always up to date',
              'Zero guesswork — only verified, sourced facts',
            ].map((item) => (
              <div key={item} className="flex items-start gap-3 bg-white border border-border rounded-xl p-4 hover:border-primary/30 transition-colors">
                <div className="w-5 h-5 rounded-full bg-primary flex items-center justify-center flex-shrink-0 mt-0.5">
                  <span className="text-white text-xs font-bold">✓</span>
                </div>
                <p className="text-sm text-foreground">{item}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="max-w-4xl mx-auto px-6 pb-24">
        <div className="bg-accent border border-primary/20 rounded-3xl p-14 text-center">
          <h2 className="text-3xl font-extrabold tracking-tight text-foreground mb-4">
            Ready to get instant answers?
          </h2>
          <p className="text-muted-foreground mb-8 text-base">Ask anything about mutual funds — all backed by official sources.</p>
          <button
            onClick={() => setShowApp(true)}
            className="bg-primary text-white px-10 py-4 rounded-xl font-bold text-sm hover:opacity-90 transition-opacity"
          >
            Open NovaDocs →
          </button>
        </div>
      </section>

      {/* Footer */}
      <div className="border-t border-border py-5 bg-white">
        <div className="max-w-6xl mx-auto px-6 flex items-center justify-between">
          <span className="text-sm font-bold tracking-tight">
            Nova<span className="text-primary">Docs</span>
          </span>
          <div className="flex items-center gap-6 text-xs text-muted-foreground">
            {['No hallucinations', 'Source verified', 'Real-time answers', 'Free to use'].map((item) => (
              <span key={item} className="flex items-center gap-1.5">
                <span className="w-1 h-1 rounded-full bg-primary inline-block"></span>
                {item}
              </span>
            ))}
          </div>
        </div>
      </div>

    </div>
  )
}