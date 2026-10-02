// Input checks that keep every tool read-only and bounded.

const SF_ID = /^[a-zA-Z0-9]{15}(?:[a-zA-Z0-9]{3})?$/;
const API_NAME = /^[A-Za-z][A-Za-z0-9_]{0,79}$/;

export class GuardError extends Error {}

/** Remove a ```sql ... ``` fence, as the benchmark's own connector does. */
export function stripFences(query: string): string {
    const fenced = /```(?:sql|SQL|soql|SOQL|sosl|SOSL)?([\s\S]+?)```/.exec(query);
    return (fenced ? fenced[1] : query).trim();
}

export function checkSoql(query: string): string {
    const q = stripFences(query);
    if (!/^select\s/i.test(q)) throw new GuardError('Only SELECT queries are allowed.');
    if (q.includes(';')) throw new GuardError('Only one statement is allowed; remove ";".');
    checkReadOnlyClauses(q);
    return q;
}

export function checkSosl(query: string): string {
    const q = stripFences(query);
    if (!/^find\s/i.test(q)) throw new GuardError('SOSL searches must start with FIND.');
    if (q.includes(';')) throw new GuardError('Only one statement is allowed; remove ";".');
    checkReadOnlyClauses(q.replace(/^find\s*\{(?:\\.|[^}\\])*\}/i, 'FIND {}'));
    return q;
}

/** SELECT/FIND can still change view statistics or lock rows. Ignore quoted
 * literals so an ordinary subject/search phrase is not mistaken for a clause. */
function checkReadOnlyClauses(query: string): void {
    const code = query.replace(/'(?:\\.|[^'\\])*'/g, "''");
    if (/\b(?:FOR\s+(?:UPDATE|VIEW|REFERENCE)|UPDATE\s+(?:TRACKING|VIEWSTAT))\b/i.test(code)) {
        throw new GuardError('Only read-only queries are allowed; locking and view/tracking updates are forbidden.');
    }
}

export function checkId(id: string): string {
    const v = id.trim();
    if (!SF_ID.test(v)) throw new GuardError(`"${id}" is not a 15- or 18-character Salesforce Id.`);
    return v;
}

export function checkApiName(name: string, what = 'object'): string {
    const v = name.trim();
    if (!API_NAME.test(v)) throw new GuardError(`"${name}" is not a valid ${what} API name.`);
    return v;
}

// Characters SOSL treats as operators inside FIND {...}.
const SOSL_RESERVED = /[?&|!{}[\]()^~*:\\"'+-]/g;

const STOPWORDS = new Set(
    (
        'a an and are as at be by can could did do does for from has have how i if in is it its of on or our ' +
        'should so that the their them there these this to was were what when where which who why will with ' +
        'would you your about any into than then also may might must not no yes we us my me'
    ).split(' ')
);

/** Turn a natural-language question into an OR-joined SOSL search term. */
export function soslTerms(text: string, maxTerms = 10): string {
    const words = text
        .replace(SOSL_RESERVED, ' ')
        .toLowerCase()
        .split(/\s+/)
        .filter(w => w.length >= 3 && !STOPWORDS.has(w));
    const unique = [...new Set(words)].slice(0, maxTerms);
    if (unique.length === 0) throw new GuardError('The search text has no usable keywords.');
    return unique.join(' OR ');
}
