// Single source of truth for runtime release metadata: read the incumbent registry and
// the canonical crossfit summary, then write the generated runtime resource that the
// visual application fetches. Never duplicate release numbers by hand.
import { readFile, mkdir, writeFile } from 'node:fs/promises';
import { resolve, join } from 'node:path';
const root = resolve(import.meta.dirname, '../../..');
const promoted = JSON.parse(await readFile(join(root, 'release/event_stack_incumbent.json'), 'utf8'));
const summary = JSON.parse(await readFile(join(root, `outputs/crossfit/summary_${promoted.run_key}.json`), 'utf8'));
const output = { run_key: promoted.run_key, outer_metrics: summary.outer_metrics };
const target = join(root, 'dist/visual/runtime/release-metadata.json');
await mkdir(resolve(target, '..'), { recursive: true });
await writeFile(target, JSON.stringify(output, null, 2) + '\n');
console.log(`Prepared release metadata for ${promoted.run_key}`);
