const { spawn } = require('child_process');
const path = require('path');

console.log('Starting Next.js development server...');

const server = spawn('node', ['node_modules/.bin/next', 'dev'], {
  cwd: __dirname,
  stdio: 'inherit'
});

server.on('error', (error) => {
  console.error('Failed to start server:', error);
});

server.on('close', (code) => {
  console.log(`Server process exited with code ${code}`);
});

process.on('SIGINT', () => {
  server.kill('SIGINT');
  process.exit(0);
});
