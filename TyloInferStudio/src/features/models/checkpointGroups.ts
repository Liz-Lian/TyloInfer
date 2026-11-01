import { modelVendor } from '../../app/ModelVendorMark';
import type { ModelArtifact } from '../../shared/api/types';

export interface CheckpointModelGroup {
  key: string;
  name: string;
  artifacts: ModelArtifact[];
}

export interface CheckpointFamilyGroup {
  key: string;
  name: string | null;
  models: CheckpointModelGroup[];
  count: number;
}

const vendorNames = { qwen: 'Qwen', deepseek: 'DeepSeek', zai: 'GLM', minicpm: 'MiniCPM', gemma: 'Gemma' };
const quantizationSuffix = /(?:^|[-_\s.])(?:EWQ?|MFQ|GGUF|MLX|EXL[23]|GSQ(?:[-_]RCO)?|UD|Gyro|BF16|FP(?:8|16|32)|MXFP[48]|NVFP4|F(?:16|32)|NINT(?:v\d+)?\d*|NVQ\d+J?|TQ(?:K|P)?\d*(?:[._]\d+)?|I?Q\d+(?:\.\d+)?|oQ\d+(?:\.\d+)?e|S\d+[-_](?:XXS|XS|XL|S|M|L)|V\d+[-_](?:XXS|XS|XL|S|M|L)|\d+(?:\.\d+)?(?:bits?|bpw))(?=$|[-_\s.])/i;

export function checkpointModelName(name: string): string {
  const basename = name.split(/[\\/]/).pop()?.trim() || name;
  const clean = basename.replace(/\.(?:gguf|mfq|tq|safetensors)$/i, '');
  const suffix = quantizationSuffix.exec(clean);
  const base = suffix && suffix.index > 0 ? clean.slice(0, suffix.index) : clean;
  return base.replace(/[-_\s]+$/, '');
}

export function groupCheckpoints(artifacts: ModelArtifact[]): CheckpointFamilyGroup[] {
  const families = new Map<string, CheckpointFamilyGroup>();
  const modelGroups = new Map<CheckpointFamilyGroup, Map<string, CheckpointModelGroup>>();
  for (const artifact of artifacts) {
    const vendor = modelVendor(artifact.name, artifact.architecture);
    const architecture = artifact.architecture?.trim();
    const declared = architecture && architecture.toLowerCase() !== 'unknown' ? architecture : null;
    const key = vendor ?? (declared ? `architecture:${declared.toLowerCase()}` : 'other');
    let family = families.get(key);
    if (!family) {
      family = { key, name: vendor ? vendorNames[vendor] : declared, models: [], count: 0 };
      families.set(key, family);
      modelGroups.set(family, new Map());
    }
    const identity = artifact.asset_kind && artifact.asset_kind !== 'model'
      ? artifact.name.replace(/[-_](?:engram|ple|per[-_]?layer[-_]?(?:token[-_]?)?emb(?:eddings?|d)?)(?:[-_].*)?$/i, '') : artifact.name;
    const name = checkpointModelName(identity || artifact.name);
    const modelKey = name.replace(/[_\s]+/g, '-').toLowerCase();
    let model = modelGroups.get(family)!.get(modelKey);
    if (!model) {
      model = { key: modelKey, name, artifacts: [] };
      family.models.push(model);
      modelGroups.get(family)!.set(modelKey, model);
    }
    model.artifacts.push(artifact);
    family.count++;
  }
  return [...families.values()];
}
