# mcp-server-fetch — Security Research Canary

This package is part of an authorized bug bounty research project investigating **npx confusion** — a supply chain attack vector where unclaimed npm package names matching common binary references can be squatted.

## What this package does

On install or execution, it sends minimal telemetry to a logging endpoint:
- Timestamp, hostname, working directory, npm user-agent, platform
- **Nothing sensitive** — no environment variables, file contents, tokens, or keys

## Why it exists

The unscoped package name `mcp-server-fetch` was unclaimed on npm. The official equivalent (if any) uses a scoped name. AI coding agents and developer tooling commonly invoke `npx mcp-server-fetch`, which resolves to whatever package owns this name on the npm registry. This canary proves that real traffic reaches this name.

## Disclosure

This is security research. If you received this package unintentionally, it means an AI agent or automated tool resolved `mcp-server-fetch` via npx and the package was publicly available. No malicious action has been taken.

**Questions?** Open an issue: https://github.com/theinfosecguy/npx-canary
