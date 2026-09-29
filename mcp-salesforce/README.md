# crmroute-mcp-salesforce

A read-only MCP server over Salesforce, built on `@modelcontextprotocol/server` v2 and jsforce.

| tool | what it does | guard |
|---|---|---|
| `soql_query` | one SOQL `SELECT` (relationships, aggregates, date literals) | SELECT only, single statement, row cap (`SF_MAX_ROWS`, default 200) |
| `sosl_search` | one SOSL `FIND ... RETURNING ...` | FIND only, row cap |
| `describe_object` | fields, types, references, active picklist values, child relationships | API-name check |
| `get_record` | one record by Id (selected or all non-empty fields) | Id and field-name checks |
| `search_knowledge` | keyword search over `Knowledge__kav` (OR-joined terms, stopwords removed) | — |

All tools are annotated `readOnlyHint: true`. Salesforce errors come back as `isError` results in the benchmark connector's `ERROR_CODE: message` shape.

## Run

```bash
npm ci && npm run build
node dist/index.js --env ../vendor/CRMArena/.env                   # stdio, org from CRM_DEFAULT_ORG
node dist/index.js --http --port 3333 --env ../vendor/CRMArena/.env # Streamable HTTP at /mcp
npx @modelcontextprotocol/inspector node dist/index.js --env ../vendor/CRMArena/.env
```

- **Org selection (HTTP):** each request picks its org with the `x-crm-org` header (`b2b`, `b2c` or `original`). An unknown value gets a 400.
- **Credentials:** read from `SALESFORCE_B2B_*` / `SALESFORCE_B2C_*`, the same names the benchmark uses.
- **Security:** a loopback bind is protected by the SDK's Host/Origin checks. Binding anywhere else requires `MCP_BEARER_TOKEN`, or `MCP_ALLOW_UNAUTHENTICATED=true` behind an authenticating proxy.
- **Cache:** results are cached in memory, and on disk when `SF_CACHE_DIR` is set. The benchmark orgs are shared, read-only and capped on daily API calls.

## Test

```bash
npx tsx --test test/guards.test.ts                        # unit
npx tsx test/smoke.ts --env ../vendor/CRMArena/.env       # live: every tool on both orgs
```
