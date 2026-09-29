#!/usr/bin/env node
// Read-only Salesforce MCP server for crmroute.
//   stdio (default):  node dist/index.js [--env path/to/.env]
//   HTTP:             node dist/index.js --http [--port 3333] [--host 127.0.0.1] [--env ...]
// In HTTP mode each request picks its org with the `x-crm-org` header (b2b | b2c | original).
import { existsSync } from 'node:fs';
import { createServer, type IncomingMessage, type ServerResponse } from 'node:http';
import { parseArgs } from 'node:util';

import { McpServer, createMcpHandler } from '@modelcontextprotocol/server';
import { localhostHostValidation, localhostOriginValidation, toNodeHandler } from '@modelcontextprotocol/node';
import { serveStdio } from '@modelcontextprotocol/server/stdio';

import { cacheStats } from './cache.js';
import { type OrgName, parseOrg, settings } from './config.js';
import { registerTools } from './tools.js';

const { values: flags } = parseArgs({
    options: {
        http: { type: 'boolean', default: false },
        port: { type: 'string', default: process.env.PORT ?? '3333' },
        host: { type: 'string', default: process.env.HOST ?? '127.0.0.1' },
        env: { type: 'string' }
    }
});

const envFile = flags.env ?? process.env.CRM_ENV_FILE;
if (envFile) {
    if (!existsSync(envFile)) throw new Error(`env file not found: ${envFile}`);
    process.loadEnvFile(envFile);
}

function buildServer(org: OrgName): McpServer {
    const server = new McpServer({ name: 'crmroute-salesforce', version: '0.1.0' });
    registerTools(server, org);
    return server;
}

if (!flags.http) {
    void serveStdio(() => buildServer(settings.defaultOrg));
    console.error(`[mcp-salesforce] stdio server ready (org=${settings.defaultOrg})`);
} else {
    const handler = createMcpHandler(({ requestInfo }) =>
        buildServer(parseOrg(requestInfo?.headers.get('x-crm-org')) ?? settings.defaultOrg)
    );
    const nodeHandler = toNodeHandler(handler);
    const loopback = ['127.0.0.1', 'localhost', '::1'].includes(flags.host);
    const validateHost = localhostHostValidation();
    const validateOrigin = localhostOriginValidation();
    const bearer = process.env.MCP_BEARER_TOKEN;
    const allowOpen = process.env.MCP_ALLOW_UNAUTHENTICATED === 'true';
    if (!loopback && !bearer && !allowOpen) {
        throw new Error('Binding beyond loopback needs MCP_BEARER_TOKEN, or MCP_ALLOW_UNAUTHENTICATED=true behind an authenticating proxy (e.g. Cloud Run IAM).');
    }

    const send = (res: ServerResponse, status: number, body: unknown) => {
        res.writeHead(status, { 'content-type': 'application/json' });
        res.end(JSON.stringify(body));
    };

    createServer((req: IncomingMessage, res: ServerResponse) => {
        const url = new URL(req.url ?? '/', 'http://localhost');
        if (url.pathname === '/healthz') return send(res, 200, { ok: true, cache: cacheStats });
        if (url.pathname !== '/mcp') return send(res, 404, { error: 'not found' });
        if (loopback && (!validateHost(req, res) || !validateOrigin(req, res))) return;
        if (bearer && req.headers.authorization !== `Bearer ${bearer}`) return send(res, 401, { error: 'unauthorized' });
        const orgHeader = req.headers['x-crm-org'];
        if (orgHeader !== undefined && !parseOrg(String(orgHeader))) {
            return send(res, 400, { error: `unknown org "${orgHeader}"; use b2b, b2c or original` });
        }
        void nodeHandler(req, res);
    }).listen(Number(flags.port), flags.host, () => {
        console.error(`[mcp-salesforce] http://${flags.host}:${flags.port}/mcp (default org=${settings.defaultOrg})`);
    });

    process.on('SIGINT', async () => {
        await handler.close();
        process.exit(0);
    });
}
