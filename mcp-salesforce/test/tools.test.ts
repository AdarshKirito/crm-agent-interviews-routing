import assert from 'node:assert/strict';
import { test } from 'node:test';

import type { McpServer } from '@modelcontextprotocol/server';
import { Connection } from 'jsforce';

import { cachedForOrg } from '../src/cache.js';
import { settings } from '../src/config.js';
import { registerTools } from '../src/tools.js';

test('cache entries are isolated by org, credentials, API version and output budget', async () => {
    const originalEnv = { ...process.env };
    const originalSettings = { ...settings };
    settings.cacheDir = '';
    process.env.SALESFORCE_B2B_USERNAME = 'first@example.invalid';
    process.env.SALESFORCE_B2B_PASSWORD = 'offline-test';
    let reads = 0;
    const read = () => cachedForOrg('b2b', ['scope-test'], async () => ++reads);
    try {
        assert.equal(await read(), 1);
        assert.equal(await read(), 1);
        process.env.SALESFORCE_B2B_USERNAME = 'second@example.invalid';
        assert.equal(await read(), 2);
        settings.apiVersion = '63.0';
        assert.equal(await read(), 3);
        settings.maxResponseChars += 1;
        assert.equal(await read(), 4);
        assert.equal(await cachedForOrg('b2c', ['scope-test'], async () => ++reads), 5);
    } finally {
        for (const key of ['SALESFORCE_B2B_USERNAME', 'SALESFORCE_B2B_PASSWORD']) {
            if (originalEnv[key] === undefined) delete process.env[key];
            else process.env[key] = originalEnv[key];
        }
        Object.assign(settings, originalSettings);
    }
});

test('all-field record and schema results respect the total budget and support narrowing', async t => {
    const originalEnv = { ...process.env };
    const originalSettings = { ...settings };
    settings.cacheDir = '';
    settings.maxResponseChars = 500;
    process.env.SALESFORCE_B2B_USERNAME = 'budget@example.invalid';
    process.env.SALESFORCE_B2B_PASSWORD = 'offline-test';
    t.mock.method(Connection.prototype, 'login', async () => ({}));
    t.mock.method(Connection.prototype, 'retrieve', async (_object: string, id: string, options?: { fields?: string[] }) => (
        options?.fields ? { Id: id } : { Id: id, Description: 'x'.repeat(1000) }
    ));
    t.mock.method(Connection.prototype, 'describe', async () => ({
        name: 'Account', label: 'Account', queryable: true, childRelationships: [],
        fields: Array.from({ length: 20 }, (_, i) => ({ name: `Field${i}`, label: `Field ${i}`, type: 'string' }))
    }));
    type Result = { content: { text: string }[]; isError?: boolean };
    const callbacks = new Map<string, (args: Record<string, unknown>) => Promise<Result>>();
    registerTools({ registerTool: (name: string, _schema: unknown, fn: (args: Record<string, unknown>) => Promise<Result>) => {
        callbacks.set(name, fn);
    } } as unknown as McpServer, 'b2b');
    try {
        const recordArgs = { object_name: 'Account', record_id: '001000000000000AAA' };
        const oversizedRecord = await callbacks.get('get_record')!(recordArgs);
        assert.equal(oversizedRecord.isError, true);
        assert.match(oversizedRecord.content[0].text, /fields subset/);
        const smallRecord = await callbacks.get('get_record')!({ ...recordArgs, fields: ['Id'] });
        assert.ok(!smallRecord.isError);
        assert.deepEqual(JSON.parse(smallRecord.content[0].text), { Id: recordArgs.record_id });
        const oversizedSchema = await callbacks.get('describe_object')!({ object_name: 'Account' });
        assert.equal(oversizedSchema.isError, true);
        const smallSchema = await callbacks.get('describe_object')!({ object_name: 'Account', fields: ['Field0'] });
        assert.ok(!smallSchema.isError);
        assert.equal(JSON.parse(smallSchema.content[0].text).fields.length, 1);
        assert.ok(smallSchema.content[0].text.length <= 500);
        const unknownField = await callbacks.get('describe_object')!({ object_name: 'Account', fields: ['NotAField'] });
        assert.equal(unknownField.isError, true);
        assert.match(unknownField.content[0].text, /Unknown fields/);
    } finally {
        for (const key of ['SALESFORCE_B2B_USERNAME', 'SALESFORCE_B2B_PASSWORD']) {
            if (originalEnv[key] === undefined) delete process.env[key];
            else process.env[key] = originalEnv[key];
        }
        Object.assign(settings, originalSettings);
    }
});
