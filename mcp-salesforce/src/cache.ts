// Result cache. The benchmark orgs are shared and capped on API calls per day,
// and their data does not change, so identical reads are served from memory and,
// when SF_CACHE_DIR is set, from disk across restarts.
import { createHash, randomUUID } from 'node:crypto';
import { mkdir, readFile, rename, rm, writeFile } from 'node:fs/promises';
import path from 'node:path';

import { type OrgName, orgCredentials, settings } from './config.js';

interface Entry {
    savedAt: number;
    value: unknown;
}

const memory = new Map<string, Entry>();
const inFlight = new Map<string, Promise<unknown>>();
const MAX_MEMORY_ENTRIES = 5000;

export const cacheStats = { hits: 0, misses: 0 };

function keyOf(parts: unknown[]): string {
    return createHash('sha256').update(JSON.stringify(parts)).digest('hex');
}

function fresh(entry: Entry): boolean {
    const age = Date.now() - entry.savedAt;
    return Number.isFinite(entry.savedAt) && age >= 0 && age < settings.cacheTtlSeconds * 1000 && 'value' in entry;
}

function remember(key: string, entry: Entry): void {
    if (!memory.has(key) && memory.size >= MAX_MEMORY_ENTRIES) {
        memory.delete(memory.keys().next().value as string);
    }
    memory.set(key, entry);
}

async function readDisk(key: string): Promise<Entry | undefined> {
    if (!settings.cacheDir) return undefined;
    try {
        return JSON.parse(await readFile(path.join(settings.cacheDir, `${key}.json`), 'utf8')) as Entry;
    } catch {
        return undefined;
    }
}

async function writeDisk(key: string, entry: Entry): Promise<void> {
    if (!settings.cacheDir) return;
    const file = path.join(settings.cacheDir, `${key}.json`);
    // Parallel servers may share one cache directory: publish complete files only.
    const temporary = `${file}.${process.pid}.${randomUUID()}.tmp`;
    try {
        await mkdir(settings.cacheDir, { recursive: true });
        await writeFile(temporary, JSON.stringify(entry));
        await rename(temporary, file);
    } catch (err) {
        await rm(temporary, { force: true }).catch(() => undefined);
        console.error('[cache] write failed:', (err as Error).message);
    }
}

/** Return the cached value for `parts`, or compute, store and return it. Errors are not cached. */
export async function cached<T>(parts: unknown[], compute: () => Promise<T>): Promise<T> {
    const key = keyOf(parts);
    const pending = inFlight.get(key);
    if (pending) {
        cacheStats.hits++;
        return pending as Promise<T>;
    }
    const work = (async () => {
        const hit = memory.get(key) ?? (await readDisk(key));
        if (hit && fresh(hit)) {
            cacheStats.hits++;
            remember(key, hit);
            return hit.value as T;
        }
        cacheStats.misses++;
        const value = await compute();
        const entry = { savedAt: Date.now(), value };
        remember(key, entry);
        await writeDisk(key, entry);
        return value;
    })();
    inFlight.set(key, work);
    try {
        return await work;
    } finally {
        inFlight.delete(key);
    }
}

/** Disk entries must not survive a change of org credentials or output limits.
 * Only the hash is used as the cache filename; no credentials are persisted. */
export function cachedForOrg<T>(org: OrgName, parts: unknown[], compute: () => Promise<T>): Promise<T> {
    const creds = orgCredentials(org);
    return cached([
        'salesforce-v2', org, creds?.loginUrl, creds?.username, settings.apiVersion,
        settings.maxTextChars, settings.maxResponseChars, ...parts
    ], compute);
}
