/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  swcMinify: true,
  images: {
    domains: ['localhost', 'rag-backend.onrender.com'],
  },
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'https://rag-backend.onrender.com/api/:path*',
      },
    ];
  },
};

module.exports = nextConfig;
