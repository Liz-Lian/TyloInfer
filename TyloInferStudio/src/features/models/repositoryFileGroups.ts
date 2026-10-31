import type { HubModelInfo, HubModelVariant } from '../../shared/api/types';

type ModelFile = HubModelInfo['files'][number];
type FileGroup = { id: string; label: string; files: ModelFile[]; expected?: number; missing: number };

export function downloadSize(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return '—';
  const units = ['B', 'KiB', 'MiB', 'GiB', 'TiB'];
  let amount = bytes;
  let unit = 0;
  while (amount >= 1024 && unit < units.length - 1) {
    amount /= 1024;
    unit += 1;
  }
  return `${amount.toLocaleString(undefined, { maximumFractionDigits: unit ? 1 : 0 })} ${units[unit]}`;
}

function shard(name: string) {
  const raw = /^(.*\.gguf)\.part-(\d{1,5})-of-(\d{1,5})$/.exec(name);
  const split = raw || /^(.*)-(\d{5})-of-(\d{5})\.(tq|mfq|gguf|safetensors|bin)$/i.exec(name);
  if (!split) return null;
  const index = Number(split[2]);
  const count = Number(split[3]);
  if (index < 1 || index > count) return null;
  const label = raw ? raw[1] : `${split[1]}.${split[4]}`;
  return { id: `${raw ? 'parts' : 'shards'}:${count}:${label}`, label, index, count };
}

export function repositoryFileGroups(files: ModelFile[], variants: HubModelVariant[] = []): FileGroup[] {
  const groups = new Map<string, FileGroup>();
  const byName = new Map(files.map(file => [file.name, file]));
  for (const file of files) {
    const part = shard(file.name);
    const id = part?.id || file.name;
    const group = groups.get(id) || { id, label: part?.label || file.name, files: [], expected: part?.count, missing: 0 };
    group.files.push(file);
    groups.set(id, group);
  }
  for (const group of groups.values()) {
    if (!group.expected) continue;
    if (/\.(tq|mfq)$/i.test(group.label) && !/(?:^|\/)Engram\/|(?:^|\/).+-Engram-/i.test(group.label)) {
      const variant = variants.find(item => item.format === 'tq' && group.files.every(file => item.files.includes(file.name)));
      if (variant) group.files = variant.files.flatMap(name => byName.get(name) ? [byName.get(name)!] : []);
    }
    const indices = new Set(group.files.map(file => shard(file.name)).filter(part => part?.count === group.expected).map(part => part!.index));
    group.missing = group.expected - indices.size;
  }
  return [...groups.values()];
}
