const http = require('http');
const { exec } = require('child_process');

console.log('Starting Next.js development server...');

// Start the Next.js development server
const server = exec('npx next dev --port 3000', {
  cwd: __dirname,
  stdio: 'inherit'
});

server.stdout.on('data', (data) => {
  console.log(data.toString());
});

server.stderr.on('data', (data) => {
  console.error(data.toString());
});

server.on('close', (code) => {
  console.log(`Server process exited with code ${code}`);
});

// Test if server is running
setTimeout(() => {
  http.get('http://localhost:3000', (res) => {
    if (res.statusCode === 200) {
      console.log('✅ Server is running on http://localhost:3000');
      console.log('🚀 Open your browser and navigate to http://localhost:3000');
    } else {
      console.log('❌ Server is not responding');
    }
  }).on('error', () => {
    console.log('❌ Server is not responding');
  });
}, 5000);
