import assert from 'node:assert/strict';
import { test } from 'node:test';

import { Connection } from 'jsforce';

import { soqlQuery } from '../src/salesforce.js';

test('SOQL pagination caps parent rows without auto-fetching unbounded child records', async t => {
    const savedUser = process.env.SALESFORCE_USERNAME;
    const savedPassword = process.env.SALESFORCE_PASSWORD;
    process.env.SALESFORCE_USERNAME = 'offline-test@example.invalid';
    process.env.SALESFORCE_PASSWORD = 'offline-test';
    const options: unknown[] = [];
    t.mock.method(Connection.prototype, 'login', async () => ({}));
    t.mock.method(Connection.prototype, 'query', async (_query: string, opts: unknown) => {
        options.push(opts);
        return {
            totalSize: 4,
            done: false,
            nextRecordsUrl: '/services/data/v62.0/query/parent-page-2',
            records: [{ Id: 'parent1', Cases: {
                totalSize: 10000,
                done: false,
                nextRecordsUrl: '/services/data/v62.0/query/child-page-2',
                records: [{ Id: 'child1' }]
            } }]
        };
    });
    const more = t.mock.method(Connection.prototype, 'queryMore', (url: string) => {
        assert.match(url, /parent-page-2$/);
        return { execute: async (opts: unknown) => {
            options.push(opts);
            return { totalSize: 4, done: false, nextRecordsUrl: 'parent-page-3', records: [{ Id: 'parent2' }] };
        } };
    });
    try {
        const result = await soqlQuery('original', 'SELECT Id, (SELECT Id FROM Cases) FROM Account', 2);
        assert.equal(result.returned, 2);
        assert.equal(result.truncated, true);
        assert.equal(more.mock.callCount(), 1);
        assert.deepEqual(options, [{ autoFetch: false, maxFetch: 2 }, { autoFetch: false, maxFetch: 1 }]);
        assert.match(result.note ?? '', /child/i);
        assert.deepEqual(result.records[0].Cases, [{ Id: 'child1' }]);
    } finally {
        if (savedUser === undefined) delete process.env.SALESFORCE_USERNAME;
        else process.env.SALESFORCE_USERNAME = savedUser;
        if (savedPassword === undefined) delete process.env.SALESFORCE_PASSWORD;
        else process.env.SALESFORCE_PASSWORD = savedPassword;
    }
});
