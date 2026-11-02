/** 验证深链接、路由生成与回退行为在页面拆分后保持一致。 */
import { describe, expect, it } from 'vitest';
import {
  dashboardPath,
  isStudioPath,
  labPath,
  normalizeStudioPath,
  primaryConversationPath,
  resolveStudioLocation,
} from '../src/navigation';

describe('Studio 页面路由', () => {
  it('uses text chat for native services and preserves the Agent entry for other services', () => {
    expect(primaryConversationPath('native-instances')).toBe('/chat');
    expect(primaryConversationPath()).toBe('/agent');
  });
  it.each([
    ['/', 'dashboard', 'overview'],
    ['/models', 'dashboard', 'models'],
    ['/runtime', 'dashboard', 'connections'],
    ['/resources', 'dashboard', 'cache'],
    ['/logs', 'dashboard', 'logs'],
    ['/model-hub', 'lab', 'models'],
    ['/evaluations', 'lab', 'evaluations'],
    ['/analysis', 'lab', 'analysis'],
    ['/chat', 'chat', 'overview'],
    ['/agent', 'agent', 'overview'],
    ['/versions', 'versions', 'overview'],
  ])('解析深链接 %s', (path, view, page) => {
    const location = resolveStudioLocation(path);
    expect(location.view).toBe(view);
    expect(view === 'lab' ? location.labPage : location.dashboardPage).toBe(page);
  });

  it('生成的业务路径可回读，兼容尾部斜杠', () => {
    expect(resolveStudioLocation(`${dashboardPath('connections')}/`).dashboardPage).toBe(
      'connections',
    );
    expect(resolveStudioLocation(`${labPath('evaluations')}///`).labPage).toBe('evaluations');
    expect(normalizeStudioPath('/')).toBe('/');
  });

  it('未知地址不会被视为业务页面', () => {
    expect(isStudioPath('/chat')).toBe(true);
    expect(isStudioPath('/chat/')).toBe(true);
    expect(isStudioPath('/agent/')).toBe(true);
    expect(isStudioPath('/applications/')).toBe(false);
    expect(isStudioPath('/settings')).toBe(false);
    expect(isStudioPath('/missing')).toBe(false);
    expect(isStudioPath('/quantization')).toBe(false);
    expect(isStudioPath('/quantization/')).toBe(false);
    expect(isStudioPath('/chat/extra')).toBe(false);
  });
});
