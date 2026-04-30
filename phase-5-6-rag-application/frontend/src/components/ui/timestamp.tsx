'use client'

import { useState, useEffect } from 'react'

interface TimestampProps {
  timestamp: number
  className?: string
}

export function Timestamp({ timestamp, className }: TimestampProps) {
  const [isClient, setIsClient] = useState(false)

  useEffect(() => {
    setIsClient(true)
  }, [])

  if (!isClient) {
    return <div className={className}>--:--:--</div>
  }

  return (
    <div className={className}>
      {new Date(timestamp * 1000).toLocaleTimeString()}
    </div>
  )
}
