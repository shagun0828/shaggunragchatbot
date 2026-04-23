/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  swcMinify: true,
  images: {
    domains: ['localhost'],
  },
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'https://rag-backend.onrender.com/api/:path*',
      },
      {
        source: '/ws',
        destination: 'wss://rag-backend.onrender.com/ws',
      },
    ],
  },
}

module.exports = nextConfig
