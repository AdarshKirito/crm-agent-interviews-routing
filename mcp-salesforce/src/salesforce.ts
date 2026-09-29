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
}

export function soqlQuery(org: OrgName, soql: string, maxRows: number): Promise<QueryOutput> {
    return withConnection(org, async conn => {
        const result = await conn.query(soql, { autoFetch: true, maxFetch: maxRows });
        const records = dropEmptyColumns((result.records as unknown[]).slice(0, maxRows).map(r => cleanRecord(r) as Row));
        return {
            totalSize: result.totalSize,
            returned: records.length,
            // COUNT() returns no rows; only a result cut at the row cap is truncated
            truncated: records.length >= maxRows && result.totalSize > records.length,
            records
        };
    });
}

export function soslSearch(org: OrgName, sosl: string, maxRows: number): Promise<QueryOutput> {
    return withConnection(org, async conn => {
        const result = await conn.search(sosl);
        const all = (result.searchRecords ?? []) as unknown[];
        const records = all.slice(0, maxRows).map(r => cleanRecord(r, true) as Row);
        return { totalSize: all.length, returned: records.length, truncated: all.length > records.length, records };
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

export function describeObject(org: OrgName, objectName: string) {
    return withConnection(org, async conn => {
        const d = await conn.describe(objectName);
        const fields: FieldInfo[] = d.fields.map(f => {
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
