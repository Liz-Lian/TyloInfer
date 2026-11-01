import { expect, it } from 'vitest';
import type { ModelArtifact } from '../../shared/api/types';
import { checkpointModelName, groupCheckpoints } from './checkpointGroups';

const artifact = (name: string, architecture = 'qwen4_exp_text', format: ModelArtifact['format'] = 'tq') => ({
  id: name, name, architecture, format,
}) as ModelArtifact;

it.each([
  ['Qwen3.8-Flash-Next-EW-S2-L', 'Qwen3.8-Flash-Next'],
  ['Qwen3.8-Flash-Next-EWQ-MFQ-V3-XXS', 'Qwen3.8-Flash-Next'],
  ['Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS', 'Qwen3.8-Flash-Next'],
  ['Qwen3.8-Flash-Next-Gyro-L-TQ3_0', 'Qwen3.8-Flash-Next'],
  ['Qwen3.8-Flash-Next-oQ2.5e-mtp', 'Qwen3.8-Flash-Next'],
  ['Qwen3.8-27B-V1_M', 'Qwen3.8-27B'],
  ['Qwen3.5-9B-TQ4', 'Qwen3.5-9B'],
  ['Qwen3.5-9B-MXFP4', 'Qwen3.5-9B'],
  ['DeepSeek-V4.1-Flash-1017-UD-Q3_K_M.gguf', 'DeepSeek-V4.1-Flash-1017'],
  ['MiniCPM-o-4.5-Q4_K_M', 'MiniCPM-o-4.5'],
  ['Gemma-3-27B-it-4bit', 'Gemma-3-27B-it'],
  ['Qwen3.5-9B-DFlash2-Q8_0', 'Qwen3.5-9B-DFlash2'],
  ['Qwen3.8-27B-Instruct-2507-EXL3-4.0bpw', 'Qwen3.8-27B-Instruct-2507'],
  ['Custom model', 'Custom model'],
])('extracts model identity without quantization suffix: %s', (name, expected) => {
  expect(checkpointModelName(name)).toBe(expected);
});

it('groups architecture families and base models across precision formats without losing variant identities', () => {
  const items = [
    artifact('Qwen3.8-Flash-Next-EW-S2-L', 'qwen4_exp-hf-mfq-alphaq-ud'),
    artifact('DeepSeek-V4.1-Flash-1017-TQ-Q3', 'deepseek_v41_text'),
    artifact('Qwen3.8-27B-V1_M', 'qwen3_5-hf-mfq-nint-recipe'),
    artifact('Qwen3.8-Flash-Next-Gyro-L-TQ3_0', 'qwen4_exp_text', 'gguf'),
    artifact('Qwen3.8-27B-Q4_K_M', 'qwen3_5_text', 'gguf'),
    artifact('Qwen3.8-Flash-Next-oQ2.5e-mtp', 'qwen4_exp_text', 'hf'),
  ];
  const groups = groupCheckpoints(items);
  expect(groups.map(group => group.name)).toEqual(['Qwen', 'DeepSeek']);
  expect(groups[0].models.map(group => group.name)).toEqual(['Qwen3.8-Flash-Next', 'Qwen3.8-27B']);
  expect(groups[0].models[0].artifacts).toEqual([items[0], items[3], items[5]]);
  expect(groups[0].models[1].artifacts).toEqual([items[2], items[4]]);
  expect(groups.map(group => group.count)).toEqual([5, 1]);
});

it('never infers parameter size from file bytes or combines different generations and model roles', () => {
  const groups = groupCheckpoints([
    artifact('Qwen3.8-27B-Q4_K_M'), artifact('Qwen3.8-32B-Q4_K_M'),
    artifact('Qwen3.5-27B-Q4_K_M'), artifact('Qwen3.8-27B-Instruct-Q4_K_M'),
    artifact('Qwen3.8-27B-DFlash2-Q8_0'),
  ]);
  expect(groups[0].models).toHaveLength(5);
});

it('uses architecture before names and preserves unrecognized model names in safe fallback groups', () => {
  const groups = groupCheckpoints([
    artifact('Local alias A', 'deepseek_v41_text'), artifact('Local alias B', 'deepseek_v41_text'),
    artifact('Qwen3.8-27B-Q4', 'llama'), artifact('Gemma-3-27B-Q4', 'unknown'),
    artifact('Unidentified A', ''), artifact('Unidentified B', 'unknown'),
  ]);
  expect(groups.map(group => group.name)).toEqual(['DeepSeek', 'llama', 'Gemma', null]);
  expect(groups[0].models).toHaveLength(2);
  expect(groups[3].models).toHaveLength(2);
});

it('keeps the caller’s sorting order inside each group, including incomplete models', () => {
  const items = [artifact('Qwen3.8-27B-Q8'), { ...artifact('Qwen3.8-27B-Q4'), complete: false, missing_shards: 1 }];
  expect(groupCheckpoints(items)[0].models[0].artifacts).toEqual(items);
  expect(groupCheckpoints([])).toEqual([]);
});

it('places independently usable tables under the same base model as its quantized versions', () => {
  const items = [artifact('Qwen3.8-Flash-Next-EW-S2-L'),
    { ...artifact('Qwen3.8-Flash-Next-PLE'), asset_kind: 'ple' as const },
    { ...artifact('Qwen3.8-Flash-Next-Engram-MXFP8'), asset_kind: 'engram' as const }];
  const groups = groupCheckpoints(items);
  expect(groups).toHaveLength(1);
  expect(groups[0].models).toHaveLength(1);
  expect(groups[0].models[0].name).toBe('Qwen3.8-Flash-Next');
  expect(groups[0].models[0].artifacts).toEqual(items);
});
