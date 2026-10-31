import type { HubModelVariant } from '../../shared/api/types';

const SUPPORT_FILES = [
  '*.json', '*.txt', '*.model', '*.tiktoken', '*.jinja',
  'tokenizer*', 'processor*', 'preprocessor*',
];

export function isSharedEngram(name: string): boolean {
  return /(?:^|\/)Engram\/.*\.(?:tq|mfq)$/i.test(name);
}

export function downloadPatterns(variant: HubModelVariant | null): string[] {
  if (!variant) return [];
  if (variant.format === 'unknown') return variant.files;
  const primary = variant.files.filter(name => !isSharedEngram(name));
  const weights = primary.length <= 48 || variant.format !== 'tq' ? primary
    : Array.from(new Set(primary.flatMap(name => {
      const suffix = name.match(/\.(?:tq|mfq)$/i)?.[0];
      return suffix ? [`${variant.label}*${suffix}`] : [];
    })));
  return Array.from(new Set([...weights, ...variant.files.filter(isSharedEngram), ...SUPPORT_FILES]));
}
