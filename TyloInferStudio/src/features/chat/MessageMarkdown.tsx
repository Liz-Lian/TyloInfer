
import { lazy, Suspense } from 'react';
const Markdown = lazy(() => import('./markdown/Markdown').then((module) => ({ default: module.Markdown })));


export function renderMarkdown(text: string, live = false, normalizeEscapedLineBreaks = false) {
  return <Suspense fallback={<p className='rich-text'>{text}</p>}>
    <Markdown live={live} normalizeEscapedLineBreaks={normalizeEscapedLineBreaks} text={text} />
  </Suspense>;
}
