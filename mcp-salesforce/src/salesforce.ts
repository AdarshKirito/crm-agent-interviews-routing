import { Connection } from 'jsforce';

import { type OrgName, orgCredentials, settings } from './config.js';

type Row = Record<string, unknown>;

export class SalesforceError extends Error {}

const connections = new Map<OrgName, Promise<Connection>>();

async function login(org: OrgName): Promise<Connection> {
    const creds = orgCredentials(org);
    if (!creds) throw new SalesforceError(`No credentials configured for org "${org}".`);
    const conn = new Connection({ loginUrl: creds.loginUrl, version: settings.apiVersion });
    await conn.login(creds.username, creds.password + creds.securityToken);
    console.error(`[salesforce] logged in to ${org} at ${conn.instanceUrl}`);
    return conn;
}

function connection(org: OrgName): Promise<Connection> {
    let pending = connections.get(org);
    if (!pending) {
        pending = login(org);
        pending.catch(() => connections.delete(org));
        connections.set(org, pending);
    }
    return pending;
}

function isExpiredSession(err: unknown): boolean {
    const e = err as { errorCode?: string; name?: string };
    return e?.errorCode === 'INVALID_SESSION_ID' || e?.name === 'INVALID_SESSION_ID';
}

/** Run `fn` with the org's connection; log in again once if the session expired. */
async function withConnection<T>(org: OrgName, fn: (conn: Connection) => Promise<T>): Promise<T> {
    try {
        return await fn(await connection(org));
    } catch (err) {
        if (!isExpiredSession(err)) throw toSalesforceError(err);
        connections.delete(org);
        try {
            return await fn(await connection(org));
        } catch (retryErr) {
            throw toSalesforceError(retryErr);
        }
    }
}

/** Same "ERROR_CODE: message" shape the benchmark's connector returns to its agent. */
function toSalesforceError(err: unknown): SalesforceError {
    if (err instanceof SalesforceError) return err;
    const e = err as { errorCode?: string; name?: string; message?: string };
    const code = e?.errorCode ?? e?.name ?? 'ERROR';
    return new SalesforceError(`${code}: ${e?.message ?? String(err)}`);
}

/** Drop Salesforce "attributes" blocks, keeping the record type where it helps. */
export function cleanRecord(value: unknown, keepType = false): unknown {
    if (Array.isArray(value)) return value.map(v => cleanRecord(v));
    if (value && typeof value === 'object') {
        const out: Row = {};
        const attrs = (value as Row).attributes as { type?: string } | undefined;
        if (keepType && attrs?.type) out._type = attrs.type;
        for (const [k, v] of Object.entries(value as Row)) {
            if (k === 'attributes') continue;
            if (v && typeof v === 'object' && 'records' in (v as Row)) {
                out[k] = cleanRecord((v as Row).records); // child relationship subquery
            } else {
                out[k] = cleanRecord(v);
            }
        }
        return out;
    }
    if (typeof value === 'string' && value.length > settings.maxTextChars) {
        return `${value.slice(0, settings.maxTextChars)}... [truncated ${value.length - settings.maxTextChars} chars]`;
    }
    return value;
}

/** Remove columns that are null in every row (the benchmark's connector does the same). */
export function dropEmptyColumns(rows: Row[]): Row[] {
    if (rows.length === 0) return rows;
    const keys = new Set(rows.flatMap(r => Object.keys(r)));
    const empty = [...keys].filter(k => rows.every(r => r[k] === null || r[k] === undefined));
    if (empty.length === 0) return rows;
    return rows.map(r => Object.fromEntries(Object.entries(r).filter(([k]) => !empty.includes(k))));
}

export interface QueryOutput {
    totalSize: number;
    returned: number;
    truncated: boolean;
    records: Row[];
    note?: string;
}

/** Keep whole records until the response budget is used; say so when rows were cut. */
export function fitToBudget(out: QueryOutput, budget = settings.maxResponseChars): QueryOutput {
    if (JSON.stringify(out).length <= budget) return out;
    const shortened = (count: number): QueryOutput => ({
        totalSize: out.totalSize,
        returned: count,
        truncated: true,
        records: out.records.slice(0, count),
        note: `${out.note ? `${out.note} ` : ''}Only ${count} of ${out.records.length} rows fit in one response. Filter further, select fewer or shorter fields, or aggregate.`
    });
    if (JSON.stringify(shortened(0)).length > budget) {
        throw new SalesforceError('Response budget is too small for query metadata; increase SF_MAX_RESPONSE_CHARS.');
    }
    // Binary search avoids repeatedly serializing every prefix for large results.
    let low = 0;
    let high = out.records.length;
    while (low < high) {
        const count = Math.ceil((low + high) / 2);
        if (JSON.stringify(shortened(count)).length <= budget) low = count;
        else high = count - 1;
    }
    return shortened(low);
}

export function soqlQuery(org: OrgName, soql: string, maxRows: number): Promise<QueryOutput> {
    return withConnection(org, async conn => {
        // jsforce's autoFetch also exhausts every child subquery (without the
        // parent's maxFetch cap). Page only the outer query and mark incomplete
        // children instead of silently presenting them as complete evidence.
        let page = await conn.query(soql, { autoFetch: false, maxFetch: maxRows });
        const totalSize = page.totalSize;
        const rawRecords: unknown[] = [...page.records];
        const visited = new Set<string>();
        while (!page.done && page.nextRecordsUrl && rawRecords.length < maxRows) {
            if (visited.has(page.nextRecordsUrl)) break;
            visited.add(page.nextRecordsUrl);
            page = await conn.queryMore(page.nextRecordsUrl).execute({ autoFetch: false, maxFetch: maxRows - rawRecords.length });
            rawRecords.push(...page.records);
        }
        const selected = rawRecords.slice(0, maxRows);
        const childTruncated = selected.some(hasIncompleteChildren);
        const records = dropEmptyColumns(selected.map(r => cleanRecord(r) as Row));
        return fitToBudget({
            totalSize,
            returned: records.length,
            // COUNT() returns no rows but has done=true and is not truncated.
            truncated: page.done === false || (records.length > 0 && totalSize > records.length) || childTruncated,
            records,
            ...(childTruncated ? { note: 'Some child relationship results are incomplete. Query the child object directly with filters or aggregates.' } : {})
        });
    });
}

function hasIncompleteChildren(value: unknown): boolean {
    if (Array.isArray(value)) return value.some(hasIncompleteChildren);
    if (!value || typeof value !== 'object') return false;
    const obj = value as Row;
    if (Array.isArray(obj.records) && (obj.done === false || Number(obj.totalSize) > obj.records.length)) return true;
    return Object.values(obj).some(hasIncompleteChildren);
}

export function soslSearch(org: OrgName, sosl: string, maxRows: number): Promise<QueryOutput> {
    return withConnection(org, async conn => {
        const result = await conn.search(sosl);
        const all = (result.searchRecords ?? []) as unknown[];
        const records = all.slice(0, maxRows).map(r => cleanRecord(r, true) as Row);
        return fitToBudget({ totalSize: all.length, returned: records.length, truncated: all.length > records.length, records });
    });
}

export interface FieldInfo {
    name: string;
    label: string;
    type: string;
    referenceTo?: string[];
    relationshipName?: string;
    picklistValues?: string[];
}

export function describeObject(org: OrgName, objectName: string, fieldNames?: string[]) {
    return withConnection(org, async conn => {
        const d = await conn.describe(objectName);
        const requested = new Set((fieldNames ?? []).map(f => f.toLowerCase()));
        const selected = requested.size ? d.fields.filter(f => requested.has(f.name.toLowerCase())) : d.fields;
        if (requested.size) {
            const found = new Set(selected.map(f => f.name.toLowerCase()));
            const missing = [...requested].filter(f => !found.has(f));
            if (missing.length) throw new SalesforceError(`Unknown fields on ${objectName}: ${missing.join(', ')}`);
        }
        const fields: FieldInfo[] = selected.map(f => {
            const info: FieldInfo = { name: f.name, label: f.label, type: f.type };
            if (f.referenceTo && f.referenceTo.length > 0) info.referenceTo = f.referenceTo as string[];
            if (f.relationshipName) info.relationshipName = f.relationshipName;
            const picklist = (f.picklistValues ?? []).filter(p => p.active).map(p => p.value);
            if (picklist.length > 0) info.picklistValues = picklist.slice(0, 30);
            return info;
        });
        const children = (d.childRelationships ?? [])
            .filter(c => c.relationshipName)
            .map(c => ({ childObject: c.childSObject, field: c.field, relationshipName: c.relationshipName }));
        return { name: d.name, label: d.label, queryable: d.queryable, fields, childRelationships: children };
    });
}

export function getRecord(org: OrgName, objectName: string, id: string, fields?: string[]) {
    return withConnection(org, async conn => {
        const record = await conn.retrieve(objectName, id, fields && fields.length > 0 ? { fields } : undefined);
        const row = cleanRecord(record) as Row;
        return Object.fromEntries(Object.entries(row).filter(([, v]) => v !== null && v !== undefined));
    });
}
