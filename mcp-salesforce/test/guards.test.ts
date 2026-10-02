import assert from 'node:assert/strict';
import { test } from 'node:test';

import { cacheStats, cached } from '../src/cache.js';
import { GuardError, checkApiName, checkId, checkSoql, checkSosl, soslTerms, stripFences } from '../src/guards.js';
import { dropEmptyColumns } from '../src/salesforce.js';

test('SOQL guard allows SELECT and strips fences', () => {
    assert.equal(checkSoql('```sql\nSELECT Id FROM Case\n```'), 'SELECT Id FROM Case');
    assert.equal(checkSoql('  select Id from Lead '), 'select Id from Lead');
});

test('SOQL guard rejects anything but a single SELECT', () => {
    for (const q of ['DELETE FROM Case', 'UPDATE Case SET Status = 1', 'SELECT Id FROM Case; SELECT Id FROM Lead', 'FIND {x}']) {
        assert.throws(() => checkSoql(q), GuardError, q);
    }
});

test('SOSL guard requires FIND', () => {
    assert.equal(checkSosl('FIND {Acme} RETURNING Account(Id)'), 'FIND {Acme} RETURNING Account(Id)');
    assert.throws(() => checkSosl('SELECT Id FROM Case'), GuardError);
});

test('read-only queries reject clauses that lock records or update tracking', () => {
    for (const clause of ['FOR UPDATE', 'for view', 'FOR\nREFERENCE', 'UPDATE TRACKING', 'UPDATE VIEWSTAT']) {
        assert.throws(() => checkSoql(`SELECT Id FROM Knowledge__kav ${clause}`), GuardError, clause);
    }
    for (const clause of ['UPDATE TRACKING', 'UPDATE VIEWSTAT']) {
        assert.throws(() => checkSosl(`FIND {policy} RETURNING Knowledge__kav(Id) ${clause}`), GuardError, clause);
    }
    assert.equal(checkSoql("SELECT Id FROM Case WHERE Subject = 'FOR UPDATE'"), "SELECT Id FROM Case WHERE Subject = 'FOR UPDATE'");
    assert.equal(checkSosl('FIND {FOR UPDATE} RETURNING Knowledge__kav(Id)'), 'FIND {FOR UPDATE} RETURNING Knowledge__kav(Id)');
});

test('response budget includes wrapper metadata and never emits an oversized first row', async () => {
    const { fitToBudget } = await import('../src/salesforce.js');
    const records = [{ Id: 'a', Body__c: 'x'.repeat(5000) }];
    const out = fitToBudget({ totalSize: 1, returned: 1, truncated: false, records }, 400);
    assert.ok(JSON.stringify(out).length <= 400);
    assert.equal(out.truncated, true);
    assert.match(out.note ?? '', /Filter|select/);
    assert.equal(out.returned, 0);
});

test('Id and API-name checks', () => {
    assert.equal(checkId('500Wt00000DDeoCIAT'), '500Wt00000DDeoCIAT');
    assert.equal(checkId('500Wt00000DDeoC'), '500Wt00000DDeoC');
    assert.throws(() => checkId("500' OR Id != '"), GuardError);
    assert.equal(checkApiName('VoiceCallTranscript__c'), 'VoiceCallTranscript__c');
    assert.throws(() => checkApiName('Case WHERE 1=1'), GuardError);
});

test('soslTerms builds an OR query without reserved characters or stopwords', () => {
    assert.equal(soslTerms('What is the policy on (late) competing-offers?'), 'policy OR late OR competing OR offers');
    assert.throws(() => soslTerms('is it on?'), GuardError);
});

test('stripFences leaves plain text alone', () => {
    assert.equal(stripFences('SELECT Id FROM Case'), 'SELECT Id FROM Case');
});

test('dropEmptyColumns removes all-null columns only', () => {
    const rows = dropEmptyColumns([
        { Id: 'a', X: null, Y: 1 },
        { Id: 'b', X: null, Y: null }
    ]);
    assert.deepEqual(rows, [{ Id: 'a', Y: 1 }, { Id: 'b', Y: null }]);
});

test('cache computes once per key and does not cache errors', async () => {
    let calls = 0;
    const compute = async () => ++calls;
    const before = cacheStats.hits;
    assert.equal(await cached(['t', 1], compute), 1);
    assert.equal(await cached(['t', 1], compute), 1);
    assert.equal(cacheStats.hits, before + 1);
    await assert.rejects(cached(['t', 2], async () => { throw new Error('boom'); }));
    assert.equal(await cached(['t', 2], compute), 2);
});

test('disk cache publishes whole entries and leaves no temporary files', async () => {
    const { mkdtemp, readdir, readFile, rm } = await import('node:fs/promises');
    const { tmpdir } = await import('node:os');
    const path = await import('node:path');
    const { settings } = await import('../src/config.js');
    const dir = await mkdtemp(path.join(tmpdir(), 'sf-cache-'));
    const saved = settings.cacheDir;
    settings.cacheDir = dir;
    try {
        const value = { records: Array.from({ length: 200 }, (_, i) => ({ Id: `r${i}`, Body__c: 'x'.repeat(500) })) };
        await Promise.all([cached(['disk', 1], async () => value), cached(['disk', 2], async () => value)]);
        const files = await readdir(dir);
        assert.equal(files.filter(f => f.endsWith('.tmp')).length, 0);
        assert.equal(files.length, 2);
        for (const f of files) {
            assert.deepEqual(JSON.parse(await readFile(path.join(dir, f), 'utf8')).value, value);
        }
    } finally {
        settings.cacheDir = saved;
        await rm(dir, { recursive: true, force: true });
    }
});

test('concurrent identical cache misses share one Salesforce read', async () => {
    let calls = 0;
    const compute = async () => {
        calls++;
        await new Promise(resolve => setTimeout(resolve, 10));
        return 'result';
    };
    assert.deepEqual(await Promise.all(Array.from({ length: 4 }, () => cached(['concurrent'], compute))),
        ['result', 'result', 'result', 'result']);
    assert.equal(calls, 1);
});

test('fitToBudget keeps whole records within the response budget', async () => {
    const { fitToBudget } = await import('../src/salesforce.js');
    const records = Array.from({ length: 10 }, (_, i) => ({ Id: `r${i}`, Body__c: 'x'.repeat(100) }));
    const out = fitToBudget({ totalSize: 10, returned: 10, truncated: false, records }, 400);
    assert.ok(out.returned > 0 && out.returned < records.length);
    assert.ok(JSON.stringify(out).length <= 400);
    assert.equal(out.truncated, true);
    assert.match(out.note ?? '', new RegExp(`${out.returned} of 10 rows`));
    const small = fitToBudget({ totalSize: 1, returned: 1, truncated: false, records: records.slice(0, 1) }, 400);
    assert.equal(small.truncated, false);
    assert.throws(() => fitToBudget({ totalSize: 1, returned: 1, truncated: false, records }, 10), /budget is too small/);
});
