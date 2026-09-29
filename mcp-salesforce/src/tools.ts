import type { McpServer } from '@modelcontextprotocol/server';
import * as z from 'zod/v4';

import { cached } from './cache.js';
import { type OrgName, settings } from './config.js';
import { GuardError, checkApiName, checkId, checkSoql, checkSosl, soslTerms } from './guards.js';
import { SalesforceError, describeObject, getRecord, soqlQuery, soslSearch } from './salesforce.js';

export const TOOL_NAMES = ['soql_query', 'sosl_search', 'describe_object', 'get_record', 'search_knowledge'] as const;

const READ_ONLY = { readOnlyHint: true, destructiveHint: false, idempotentHint: true, openWorldHint: false } as const;

type ToolResult = { content: { type: 'text'; text: string }[]; isError?: boolean };

function ok(value: unknown): ToolResult {
    return { content: [{ type: 'text', text: JSON.stringify(value) }] };
}

async function run(fn: () => Promise<unknown>): Promise<ToolResult> {
    try {
        return ok(await fn());
    } catch (err) {
        if (err instanceof GuardError || err instanceof SalesforceError) {
            return { content: [{ type: 'text', text: err.message }], isError: true };
        }
        console.error('[tool] unexpected error:', err);
        return { content: [{ type: 'text', text: `Unexpected error: ${(err as Error).message}` }], isError: true };
    }
}

const rowsArg = z
    .number()
    .int()
    .min(1)
    .max(settings.hardMaxRows)
    .optional()
    .describe(`Maximum rows to return (default ${settings.maxRows}).`);

/** Register the five read-only tools, bound to one org for this request. */
export function registerTools(server: McpServer, org: OrgName): void {
    server.registerTool(
        'soql_query',
        {
            title: 'Run a SOQL query',
            description:
                'Run one read-only SOQL SELECT query against the CRM org and return matching records as JSON ' +
                '({totalSize, returned, truncated, records}). Supports relationship fields, subqueries, ' +
                'aggregates (COUNT, AVG, GROUP BY) and date literals. Columns that are empty in every row are dropped.',
            inputSchema: z.object({
                query: z.string().min(8).describe("A SOQL SELECT statement, e.g. SELECT Id, Subject FROM Case WHERE Status = 'Closed'"),
                max_rows: rowsArg
            }),
            annotations: READ_ONLY
        },
        async ({ query, max_rows }) =>
            run(() => {
                const soql = checkSoql(query);
                const rows = max_rows ?? settings.maxRows;
                return cached([org, 'soql', soql, rows], () => soqlQuery(org, soql, rows));
            })
    );

    server.registerTool(
        'sosl_search',
        {
            title: 'Run a SOSL search',
            description:
                'Run one SOSL text search (FIND {terms} IN ... RETURNING Object(fields)) across the org search index. ' +
                'Use it to find records by free text such as names, subjects or transcript wording.',
            inputSchema: z.object({
                query: z.string().min(6).describe('A SOSL statement starting with FIND, e.g. FIND {Joe Smith} IN NAME FIELDS RETURNING Contact(Id, Name)'),
                max_rows: rowsArg
            }),
            annotations: READ_ONLY
        },
        async ({ query, max_rows }) =>
            run(() => {
                const sosl = checkSosl(query);
                const rows = max_rows ?? settings.maxRows;
                return cached([org, 'sosl', sosl, rows], () => soslSearch(org, sosl, rows));
            })
    );

    server.registerTool(
        'describe_object',
        {
            title: 'Describe an object',
            description:
                'List the fields of one Salesforce object (name, label, type, referenced objects, active picklist values) ' +
                'and its child relationships. Use it before writing a query against an unfamiliar object.',
            inputSchema: z.object({
                object_name: z.string().describe('Object API name, e.g. Case, Opportunity, VoiceCallTranscript__c')
            }),
            annotations: READ_ONLY
        },
        async ({ object_name }) =>
            run(() => {
                const name = checkApiName(object_name);
                return cached([org, 'describe', name], () => describeObject(org, name));
            })
    );

    server.registerTool(
        'get_record',
        {
            title: 'Get one record',
            description: 'Fetch one record by Id. Returns the requested fields, or every non-empty field when none are given.',
            inputSchema: z.object({
                object_name: z.string().describe('Object API name, e.g. Quote'),
                record_id: z.string().describe('15- or 18-character Salesforce Id'),
                fields: z.array(z.string()).max(100).optional().describe('Field API names to return')
            }),
            annotations: READ_ONLY
        },
        async ({ object_name, record_id, fields }) =>
            run(() => {
                const name = checkApiName(object_name);
                const id = checkId(record_id);
                const cols = (fields ?? []).map(f => checkApiName(f, 'field'));
                return cached([org, 'get', name, id, cols], () => getRecord(org, name, id, cols));
            })
    );

    server.registerTool(
        'search_knowledge',
        {
            title: 'Search knowledge articles',
            description:
                "Keyword search over the org's knowledge articles (Knowledge__kav) using the Salesforce search index. " +
                'Returns Id, Title, Summary and the article text for the best matches.',
            inputSchema: z.object({
                query: z.string().min(2).describe('What to look for, in plain words'),
                top_k: z.number().int().min(1).max(20).optional().describe('Number of articles to return (default 5)')
            }),
            annotations: READ_ONLY
        },
        async ({ query, top_k }) =>
            run(() => {
                const k = top_k ?? 5;
                const sosl =
                    `FIND {${soslTerms(query)}} IN ALL FIELDS ` +
                    `RETURNING Knowledge__kav(Id, Title, UrlName, Summary, FAQ_Answer__c) LIMIT ${k}`;
                return cached([org, 'knowledge', sosl], async () => {
                    const out = await soslSearch(org, sosl, k);
                    return { returned: out.returned, articles: out.records.map(({ _type, ...rest }) => rest) };
                });
            })
    );
}
