import assert from 'node:assert/strict';
import path from 'node:path';
import { test } from 'node:test';

import { Client } from '@modelcontextprotocol/client';
import { StdioClientTransport } from '@modelcontextprotocol/client/stdio';

test('--env config is loaded before tool schemas and org defaults are constructed', async () => {
    const transport = new StdioClientTransport({
        command: process.execPath,
        args: ['--import', 'tsx', 'src/index.ts', '--env', 'test/fixtures/settings.env'],
        cwd: path.resolve(import.meta.dirname, '..'),
        env: { PATH: process.env.PATH ?? '', SystemRoot: process.env.SystemRoot ?? '' },
        stderr: 'pipe'
    });
    const client = new Client({ name: 'offline-startup-test', version: '1.0.0' }, { versionNegotiation: { mode: 'auto' } });
    let stderr = '';
    transport.stderr?.on('data', chunk => { stderr += String(chunk); });
    try {
        await client.connect(transport);
        const { tools } = await client.listTools();
        const query = tools.find(t => t.name === 'soql_query')!;
        assert.match(JSON.stringify(query.inputSchema), /default 3/);
        assert.match(JSON.stringify(query.inputSchema), /"maximum":7/);
        assert.match(stderr, /org=b2c/);
    } finally {
        await client.close();
    }
});
