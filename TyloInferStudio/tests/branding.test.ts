import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { expect, it } from 'vitest';

const read = (path: string) => readFileSync(resolve(process.cwd(), path), 'utf8');

it('uses the TyloInfer mark across the desktop and web shell', () => {
  const mark = read('src/assets/tylogi-ai-lab-mark.svg');
  expect(mark).toContain('aria-label="Tylogi AI Lab"');
  expect(mark).toContain('M48 48H86M67 48V77C67 84 71.5 88 79 88H84');
  expect(mark).not.toContain('M16 44V20L32 37 48 20V44');
  expect(read('index.html')).toContain('/src/assets/tylogi-ai-lab-mark.svg');
  expect(read('src/app/StudioSidebar.tsx')).toContain('<div><strong>TyloInfer</strong></div>');
  for (const component of ['StudioSidebar', 'LoadingPage', 'NotFoundPage', 'DesktopWindowFrame', 'LoginPage', 'FailurePage', 'display']) {
    const source = read(`src/app/${component}.tsx`);
    expect(source).toContain('../assets/tylogi-ai-lab-mark.svg');
    expect(source).not.toContain('mfq-mark.svg');
  }
});

it('uses TyloInfer in context-management descriptions and exported filenames', () => {
  const context = read('src/features/runtime/ModelContextSettings.tsx');
  expect(context).toContain('留空由 TyloInfer 管理');
  expect(context).not.toContain('由 MFQ 管理');
  expect(read('src/features/evaluations/EvaluationDetails.tsx')).toContain('tyloinfer-evaluation-');
  expect(read('src/features/runtime/LogsPage.tsx')).toContain('`tyloinfer-${tab}-');
});

it('keeps native Dock icons padded and uses the Tylogi TL mark', () => {
  const build = read('src-tauri/build.rs');
  expect(build).toContain('const MACOS_ICON_SCALE: f64 = 0.79;');
  expect(build).toContain('(10.0, 12.0, 29.0, 12.0)');
  expect(build).toContain('(19.5, 12.0, 19.5, 26.5)');
  expect(build).toContain('(25.5, 32.0, 28.0, 32.0)');
  expect(build).toContain('(x - 31.5).hypot(y - 32.0) <= 2.25');
});
