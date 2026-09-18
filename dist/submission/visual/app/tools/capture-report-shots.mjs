// Capture the demo-data screenshots used by the competition report
// (scripts/report/build_report.py reads them from scripts/report/assets/).
//
// The page is opened as a local file from the shipped standalone bundle and it
// carries only DEMO DATA (synthetic) — no real subject data is ever involved.
//
// Usage: node dist/visual/app/tools/capture-report-shots.mjs
//   BME_VISUAL_PAGE  override the page URL (default: the built submission's
//                    standalone visual, falling back to dist/visual when the
//                    submission bundle is not present)
//   BME_SHOTS_OUT    override the output directory (default scripts/report/assets)
import { existsSync, mkdirSync } from 'node:fs';
import { resolve } from 'node:path';
import { chromium } from 'playwright-core';

const app = resolve(import.meta.dirname, '..');
const repo = resolve(app, '../../..');
const outDir = process.env.BME_SHOTS_OUT
  ? resolve(process.env.BME_SHOTS_OUT)
  : resolve(repo, 'scripts/report/assets');

function pageUrl() {
  if (process.env.BME_VISUAL_PAGE) return process.env.BME_VISUAL_PAGE;
  const built = resolve(repo, 'dist/submission/visual/index.html');
  const workspace = resolve(repo, 'dist/visual/index.html');
  return 'file:///' + (existsSync(built) ? built : workspace).replace(/\\/g, '/');
}

mkdirSync(outDir, { recursive: true });
const url = pageUrl();
console.log('page:', url);

const browser = await chromium.launch({ channel: 'msedge', headless: true })
  .catch(() => chromium.launch({ channel: 'chrome', headless: true }));
try {
  const page = await browser.newPage({ viewport: { width: 1560, height: 1020 }, deviceScaleFactor: 1.6 });
  const errors = [];
  page.on('pageerror', error => errors.push(String(error)));
  await page.goto(url);
  await page.waitForSelector('.timeline-panel', { timeout: 60000 });
  await page.waitForTimeout(2500);
  await page.screenshot({ path: resolve(outDir, 'ui_monitor.png') });
  console.log('monitor captured, badge:', (await page.locator('.top-meta').innerText()).replace(/\s+/g, ' '));

  await page.click('nav button:has-text("Events")');
  await page.waitForSelector('.event-record', { timeout: 30000 });
  await page.waitForTimeout(600);
  await page.screenshot({ path: resolve(outDir, 'ui_events.png') });
  console.log('events captured:', await page.locator('.event-record').count());

  await page.click('nav button:has-text("Model")');
  await page.waitForSelector('.model-page', { timeout: 30000 });
  await page.waitForTimeout(600);
  await page.screenshot({ path: resolve(outDir, 'ui_model.png') });
  console.log('model captured');

  if (errors.length) throw new Error('page errors: ' + errors.join(' | '));
} finally {
  await browser.close();
}
