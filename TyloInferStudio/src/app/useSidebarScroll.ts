import { useLayoutEffect, useRef, useState } from 'react';

export function useSidebarScroll() {
  const scrollRef = useRef<HTMLDivElement>(null);
  const [scrollAll, setScrollAll] = useState(false);
  useLayoutEffect(() => {
    const outer = scrollRef.current;
    if (!outer) return;
    let frame = 0;
    let disposed = false;
    const observed = new Set<Element>();
    const schedule = () => { if (!frame && !disposed) frame = requestAnimationFrame(measure); };
    const resize = typeof ResizeObserver === 'undefined' ? null : new ResizeObserver(schedule);
    function measure() {
      frame = 0;
      if (disposed || !outer!.getClientRects().length) return;
      const list = outer!.querySelector<HTMLElement>('.agent-workspace-list');
      if (!list) return;
      const first = list.querySelector<HTMLElement>('.agent-workspace-heading, .agent-workspace-session-row, .agent-workspace-notice');
      const targets = new Set<Element>([outer!, ...outer!.querySelectorAll('.sectioned-nav > section, .agent-workspace-controls'), ...(first ? [first] : [])]);
      for (const element of observed) if (!targets.has(element)) { resize?.unobserve(element); observed.delete(element); }
      for (const element of targets) if (!observed.has(element)) { resize?.observe(element); observed.add(element); }
      const fixedHeight = list.getBoundingClientRect().top - outer!.getBoundingClientRect().top + outer!.scrollTop;
      const firstRow = first ? first.getBoundingClientRect().bottom - list.getBoundingClientRect().top + list.scrollTop : 0;
      setScrollAll(fixedHeight + firstRow > outer!.clientHeight + .5);
    }
    const mutations = new MutationObserver(schedule);
    mutations.observe(outer, { childList: true, subtree: true, characterData: true, attributes: true, attributeFilter: ['hidden'] });
    window.addEventListener('resize', schedule);
    measure();
    return () => { disposed = true; cancelAnimationFrame(frame); resize?.disconnect(); mutations.disconnect(); window.removeEventListener('resize', schedule); };
  }, []);
  useLayoutEffect(() => { if (!scrollAll && scrollRef.current) scrollRef.current.scrollTop = 0; }, [scrollAll]);
  return { scrollRef, scrollAll };
}
