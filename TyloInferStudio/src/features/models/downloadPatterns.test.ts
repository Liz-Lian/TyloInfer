import { describe, expect, it } from 'vitest';
import type { HubModelVariant } from '../../shared/api/types';
import { downloadPatterns, isSharedEngram } from './downloadPatterns';

function variant(files: string[]): HubModelVariant {
  return { id: 'tq:TQ2.5/model', label: 'TQ2.5/model', format: 'tq', files, byte_size: 1024,
    configuration: { status: 'unknown', recommendation: 'unknown', reasons: [] } };
}

describe('TQ model download selection', () => {
  it.each(['mfq', 'tq', 'MFQ'])('preserves .%s for a 65-shard tier', (suffix) => {
    const files = Array.from({ length: 65 }, (_, index) => `TQ2.5/model-${String(index + 1).padStart(5, '0')}-of-00065.${suffix}`);
    const patterns = downloadPatterns(variant(files));
    expect(patterns).toContain(`TQ2.5/model*.${suffix}`);
    expect(patterns.filter(pattern => /\.(?:mfq|tq)$/i.test(pattern))).toEqual([`TQ2.5/model*.${suffix}`]);
    expect(patterns).not.toContain('*.mfq');
    expect(patterns).not.toContain('*.tq');
  });

  it('keeps both suffixes when a tier lists both', () => {
    const files = Array.from({ length: 50 }, (_, index) => `TQ2.5/model-${index}.${index % 2 ? 'tq' : 'mfq'}`);
    expect(downloadPatterns(variant(files))).toEqual(expect.arrayContaining(['TQ2.5/model*.tq', 'TQ2.5/model*.mfq']));
  });

  it.each(['mfq', 'tq'])('retains shared Engram .%s files explicitly', (suffix) => {
    const shared = `Engram/DeepSeek-Engram-MXFP8-00002-of-00043.${suffix}`;
    expect(isSharedEngram(shared)).toBe(true);
    expect(isSharedEngram(`TQ2.5/model.${suffix}`)).toBe(false);
    expect(downloadPatterns(variant([`TQ2.5/model.${suffix}`, shared]))).toEqual(
      expect.arrayContaining([`TQ2.5/model.${suffix}`, shared]));
  });

  it('keeps single-file tiers explicit', () => {
    const file = 'UD-IQ2_M/Qwen3.5-9B-TQ-UD-IQ2_M-GateUpJoint.mfq';
    expect(downloadPatterns({ ...variant([file]), label: 'UD-IQ2_M/Qwen3.5-9B-TQ-UD-IQ2_M-GateUpJoint' })).toContain(file);
    expect(downloadPatterns(null)).toEqual([]);
  });
});
