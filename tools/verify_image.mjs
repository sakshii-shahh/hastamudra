// Verify the reference-image bug fix over file:// using the pre-installed Chromium.
import pw from '/opt/node22/lib/node_modules/playwright/index.js';
import path from 'node:path';
const { chromium } = pw;

const file = 'file://' + path.resolve(process.argv[2] || 'index.html');
const browser = await chromium.launch({ executablePath: '/opt/pw-browsers/chromium' });
const page = await browser.newPage();
const errors = [];
page.on('pageerror', e => errors.push(String(e)));
await page.goto(file, { waitUntil: 'domcontentloaded' });

// showMudra is a top-level lexical binding of the inline classic script.
const result = await page.evaluate(async () => {
  if (typeof showMudra !== 'function') return { ok: false, why: 'showMudra not defined' };
  showMudra('pataka');
  const img = document.getElementById('refImg');
  // wait for the image to finish loading (or error)
  await new Promise(res => {
    if (img.complete && img.naturalWidth) return res();
    img.addEventListener('load', res, { once: true });
    img.addEventListener('error', res, { once: true });
    setTimeout(res, 3000);
  });
  const cs = getComputedStyle(img);
  return {
    ok: true,
    src: img.getAttribute('src'),
    inlineDisplay: img.style.display,
    computedDisplay: cs.display,
    naturalWidth: img.naturalWidth,
    naturalHeight: img.naturalHeight,
    phVisible: getComputedStyle(document.getElementById('refPh')).display !== 'none',
    phText: document.getElementById('refPh').textContent,
  };
});

const pass = result.ok && result.naturalWidth > 0 &&
             result.computedDisplay !== 'none' && !result.phVisible;
console.log(JSON.stringify({ pass, result, pageErrors: errors }, null, 2));
await browser.close();
process.exit(pass ? 0 : 1);
