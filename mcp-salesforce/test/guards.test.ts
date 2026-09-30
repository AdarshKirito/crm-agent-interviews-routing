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

test('fitToBudget keeps whole records within the response budget', async () => {
    const { fitToBudget } = await import('../src/salesforce.js');
    const records = Array.from({ length: 10 }, (_, i) => ({ Id: `r${i}`, Body__c: 'x'.repeat(100) }));
    const out = fitToBudget({ totalSize: 10, returned: 10, truncated: false, records }, 400);
    assert.equal(out.returned, 3);
    assert.equal(out.truncated, true);
    assert.match(out.note ?? '', /3 of 10 rows/);
    const small = fitToBudget({ totalSize: 1, returned: 1, truncated: false, records: records.slice(0, 1) }, 400);
    assert.equal(small.truncated, false);
    // a single oversized record is still returned (never an empty answer)
    assert.equal(fitToBudget({ totalSize: 1, returned: 1, truncated: false, records }, 10).returned, 1);
});
