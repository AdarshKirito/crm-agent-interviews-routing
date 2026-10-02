export type OrgName = 'b2b' | 'b2c' | 'original';

export const ORG_NAMES: readonly OrgName[] = ['b2b', 'b2c', 'original'];

export interface OrgCredentials {
    username: string;
    password: string;
    securityToken: string;
    loginUrl: string;
}

// Same variable names as the CRMArena benchmark's .env, so one file serves both.
const ENV_PREFIX: Record<OrgName, string> = {
    b2b: 'SALESFORCE_B2B_',
    b2c: 'SALESFORCE_B2C_',
    original: 'SALESFORCE_'
};

export function orgCredentials(org: OrgName): OrgCredentials | undefined {
    const prefix = ENV_PREFIX[org];
    const username = process.env[`${prefix}USERNAME`];
    const password = process.env[`${prefix}PASSWORD`];
    const securityToken = process.env[`${prefix}SECURITY_TOKEN`] ?? '';
    if (!username || !password) return undefined;
    return {
        username,
        password,
        securityToken,
        loginUrl: process.env.SALESFORCE_LOGIN_URL ?? 'https://login.salesforce.com'
    };
}

export function parseOrg(value: string | null | undefined): OrgName | undefined {
    const v = (value ?? '').trim().toLowerCase();
    return (ORG_NAMES as readonly string[]).includes(v) ? (v as OrgName) : undefined;
}

function intEnv(name: string, fallback: number): number {
    const raw = process.env[name];
    const n = raw ? Number(raw) : Number.NaN;
    return Number.isSafeInteger(n) && n > 0 ? n : fallback;
}

const hardMaxRows = intEnv('SF_HARD_MAX_ROWS', 2000);

export const settings = {
    defaultOrg: parseOrg(process.env.CRM_DEFAULT_ORG) ?? 'b2b',
    apiVersion: process.env.SALESFORCE_API_VERSION ?? '62.0',
    maxRows: Math.min(intEnv('SF_MAX_ROWS', 200), hardMaxRows),
    hardMaxRows,
    maxTextChars: intEnv('SF_MAX_TEXT_CHARS', 4000),
    // total size of one tool result; keeps the model's context (and free-tier token use) bounded
    maxResponseChars: intEnv('SF_MAX_RESPONSE_CHARS', 16000),
    cacheDir: process.env.SF_CACHE_DIR ?? '',
    cacheTtlSeconds: intEnv('SF_CACHE_TTL_S', 7 * 24 * 3600)
};
