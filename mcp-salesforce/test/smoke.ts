// Live smoke test: starts the HTTP server, then calls every tool on both
// CRMArena-Pro orgs through the official MCP client.
//   npx tsx test/smoke.ts --env ../vendor/CRMArena/.env
import { spawn } from 'node:child_process';
import assert from 'node:assert/strict';
import { parseArgs } from 'node:util';

import { Client, StreamableHTTPClientTransport } from '@modelcontextprotocol/client';

const { values } = parseArgs({ options: { env: { type: 'string' }, port: { type: 'string', default: '3399' } } });
const port = values.port!;
const base = `http://127.0.0.1:${port}`;

const server = spawn(process.execPath, ['dist/index.js', '--http', '--port', port, ...(values.env ? ['--env', values.env] : [])], {
    stdio: ['ignore', 'inherit', 'inherit']
});

async function waitForHealth() {
    for (let i = 0; i < 50; i++) {
        try {
            if ((await fetch(`${base}/healthz`)).ok) return;
        } catch {
            /* not up yet */
        }
        await new Promise(r => setTimeout(r, 200));
    }
    throw new Error('server did not start');
}

function text(result: Awaited<ReturnType<Client['callTool']>>): string {
    const block = result.content?.[0];
    return block && block.type === 'text' ? block.text : '';
}

async function clientFor(org: string) {
    const client = new Client({ name: 'smoke', version: '1.0.0' }, { versionNegotiation: { mode: 'auto' } });
    await client.connect(new StreamableHTTPClientTransport(new URL(`${base}/mcp`), { requestInit: { headers: { 'x-crm-org': org } } }));
    return client;
}

try {
    await waitForHealth();

    const bad = await fetch(`${base}/mcp`, { method: 'POST', headers: { 'x-crm-org': 'prod', 'content-type': 'application/json' }, body: '{}' });
    assert.equal(bad.status, 400, 'unknown org must be rejected');
    console.log('ok  unknown org header -> 400');

    for (const org of ['b2b', 'b2c']) {
        const client = await clientFor(org);
        const { tools } = await client.listTools();
        assert.deepEqual(tools.map(t => t.name).sort(), ['describe_object', 'get_record', 'search_knowledge', 'soql_query', 'sosl_search']);
        assert.ok(tools.every(t => t.annotations?.readOnlyHint === true), 'all tools must be read-only');
        console.log(`ok  [${org}] tools/list -> 5 read-only tools`);

        const q = await client.callTool({ name: 'soql_query', arguments: { query: 'SELECT Id, Subject, Status, OwnerId FROM Case ORDER BY CreatedDate LIMIT 3' } });
        assert.ok(!q.isError, text(q));
        const rows = JSON.parse(text(q));
        assert.equal(rows.returned, 3);
        console.log(`ok  [${org}] soql_query -> ${rows.returned} of ${rows.totalSize}; first case ${rows.records[0].Id}`);

        const caseId = rows.records[0].Id as string;
        const rec = await client.callTool({ name: 'get_record', arguments: { object_name: 'Case', record_id: caseId, fields: ['Id', 'Subject', 'Status'] } });
        assert.ok(!rec.isError, text(rec));
        assert.equal(JSON.parse(text(rec)).Id, caseId);
        console.log(`ok  [${org}] get_record -> ${text(rec).slice(0, 90)}`);

        const d = await client.callTool({ name: 'describe_object', arguments: { object_name: 'VoiceCallTranscript__c' } });
        assert.ok(!d.isError, text(d));
        const fields = JSON.parse(text(d)).fields.map((f: { name: string }) => f.name);
        assert.ok(fields.includes('Body__c'));
        console.log(`ok  [${org}] describe_object -> ${fields.length} fields`);

        const s = await client.callTool({ name: 'sosl_search', arguments: { query: 'FIND {refund} IN ALL FIELDS RETURNING Knowledge__kav(Id, Title) LIMIT 3' } });
        assert.ok(!s.isError, text(s));
        console.log(`ok  [${org}] sosl_search -> ${text(s).slice(0, 110)}`);

        const k = await client.callTool({ name: 'search_knowledge', arguments: { query: 'What is the policy on discounts for large quotes?', top_k: 3 } });
        assert.ok(!k.isError, text(k));
        const arts = JSON.parse(text(k)).articles;
        console.log(`ok  [${org}] search_knowledge -> ${arts.length} articles: ${arts.map((a: { Title: string }) => a.Title).join(' | ').slice(0, 140)}`);

        const write = await client.callTool({ name: 'soql_query', arguments: { query: "DELETE FROM Case WHERE Id = '500000000000000'" } });
        assert.ok(write.isError, 'non-SELECT must be rejected');
        console.log(`ok  [${org}] write attempt rejected -> ${text(write)}`);

        const sfErr = await client.callTool({ name: 'soql_query', arguments: { query: 'SELECT NoSuchField__c FROM Case' } });
        assert.ok(sfErr.isError);
        console.log(`ok  [${org}] Salesforce error surfaced -> ${text(sfErr).slice(0, 90)}`);
        await client.close();
    }
    const health = await (await fetch(`${base}/healthz`)).json();
    console.log('cache stats', health.cache);
    console.log('SMOKE PASSED');
} finally {
    server.kill();
}
