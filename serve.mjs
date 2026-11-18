// Minimal static server. `node serve.mjs [port]`
// If cert/key.pem and cert/cert.pem exist, serves HTTPS (needed for the camera
// on a phone over Wi-Fi; localhost works over plain HTTP).
import http from 'node:http';
import https from 'node:https';
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import { fileURLToPath } from 'node:url';

const root = path.dirname(fileURLToPath(import.meta.url));
const port = +(process.argv[2] || process.env.PORT || 8080);
const TYPES = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.css': 'text/css',
  '.svg': 'image/svg+xml', '.json': 'application/json', '.webmanifest': 'application/manifest+json', '.png': 'image/png' };

function handler(req, res) {
  const url = decodeURIComponent(new URL(req.url, 'http://x').pathname);
  let file = path.normalize(path.join(root, url === '/' ? 'index.html' : url));
  if (!file.startsWith(root)) { res.writeHead(403).end(); return; }
  fs.readFile(file, (err, data) => {
    if (err) { res.writeHead(404).end('not found'); return; }
    res.writeHead(200, { 'Content-Type': TYPES[path.extname(file)] || 'application/octet-stream', 'Cache-Control': 'no-cache' });
    res.end(data);
  });
}

const keyFile = path.join(root, 'cert', 'key.pem'), certFile = path.join(root, 'cert', 'cert.pem');
const secure = fs.existsSync(keyFile) && fs.existsSync(certFile);
const server = secure
  ? https.createServer({ key: fs.readFileSync(keyFile), cert: fs.readFileSync(certFile) }, handler)
  : http.createServer(handler);
server.listen(port, () => {
  const proto = secure ? 'https' : 'http';
  console.log(`Serving ${root}`);
  console.log(`  ${proto}://localhost:${port}/`);
  for (const nets of Object.values(os.networkInterfaces()))
    for (const n of nets || []) if (n.family === 'IPv4' && !n.internal) console.log(`  ${proto}://${n.address}:${port}/`);
});
