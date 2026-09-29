// Result cache. The benchmark orgs are shared and capped on API calls per day,
// and their data does not change, so identical reads are served from memory and,
// when SF_CACHE_DIR is set, from disk across restarts.
import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';

import { settings } from './config.js';

interface Entry {
    savedAt: number;
    value: unknown;
}

const memory = new Map<string, Entry>();
const MAX_MEMORY_ENTRIES = 5000;

export const cacheStats = { hits: 0, misses: 0 };

function keyOf(parts: unknown[]): string {
    return createHash('sha256').update(JSON.stringify(parts)).digest('hex');
}

function fresh(entry: Entry): boolean {
    return Date.now() - entry.savedAt < settings.cacheTtlSeconds * 1000;
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
    try {
        await mkdir(settings.cacheDir, { recursive: true });
        await writeFile(path.join(settings.cacheDir, `${key}.json`), JSON.stringify(entry));
    } catch (err) {
        console.error('[cache] write failed:', (err as Error).message);
    }
}

/** Return the cached value for `parts`, or compute, store and return it. Errors are not cached. */
export async function cached<T>(parts: unknown[], compute: () => Promise<T>): Promise<T> {
    const key = keyOf(parts);
    const hit = memory.get(key) ?? (await readDisk(key));
    if (hit && fresh(hit)) {
        cacheStats.hits++;
        memory.set(key, hit);
        return hit.value as T;
    }
    cacheStats.misses++;
    const value = await compute();
    const entry = { savedAt: Date.now(), value };
    if (memory.size >= MAX_MEMORY_ENTRIES) memory.delete(memory.keys().next().value as string);
    memory.set(key, entry);
    await writeDisk(key, entry);
    return value;
}
