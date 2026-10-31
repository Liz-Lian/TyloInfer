import { fireEvent, render, screen } from '@testing-library/react';
import { expect, it, vi } from 'vitest';
import { RepositoryFiles } from './RepositoryFiles';
import type { HubModelVariant } from '../../shared/api/types';

const tr = (_zh: string, en: string) => en;
const file = (name: string) => ({ name, byte_size: 1024 });

it.each(['gguf', 'tq', 'mfq', 'safetensors', 'bin'])('selects all .%s shards together without renaming them', suffix => {
  const files = [1, 2, 3].map(index => file(`model-${String(index).padStart(5, '0')}-of-00003.${suffix}`));
  const other = file(`other.${suffix}`);
  const onDownload = vi.fn();
  render(<RepositoryFiles files={[...files, other]} disabled={false} onDownload={onDownload} tr={tr} expanded />);
  expect(screen.getAllByRole('checkbox')).toHaveLength(2);
  expect(screen.getByText('3/3 shards')).toBeVisible();
  fireEvent.click(screen.getByRole('checkbox', { name: new RegExp(`model\\.${suffix}`) }));
  fireEvent.click(screen.getByRole('button', { name: 'Download 3 selected files' }));
  expect(onDownload).toHaveBeenCalledExactlyOnceWith(files, 'selected-files', expect.any(Object));
  fireEvent.click(screen.getByRole('checkbox', { name: new RegExp(`model\\.${suffix}`) }));
  expect(screen.getByRole('button', { name: 'Download 0 selected files' })).toBeDisabled();
});

it('groups raw GGUF parts and reports an incomplete set', () => {
  const files = [file('model.gguf.part-1-of-3'), file('model.gguf.part-3-of-3')];
  const onDownload = vi.fn();
  render(<RepositoryFiles files={files} disabled={false} onDownload={onDownload} tr={tr} expanded />);
  expect(screen.getByText('2/3 shards · Incomplete')).toBeVisible();
  fireEvent.click(screen.getByRole('checkbox', { name: /model.gguf/ }));
  fireEvent.click(screen.getByRole('button', { name: 'Download 2 selected files' }));
  expect(onDownload.mock.calls[0][0]).toEqual(files);
});

it('downloads a folder directly, preserving nested files and excluding sibling precisions', () => {
  const files = [file('4bit/model.safetensors'), file('4bit/config.json'), file('4bit/tokenizer/vocab.json')];
  const onDownload = vi.fn();
  render(<RepositoryFiles files={[...files, file('8bit/model.safetensors')]} disabled={false} onDownload={onDownload} tr={tr} expanded />);
  fireEvent.click(screen.getByRole('button', { name: 'Download folder 4bit' }));
  expect(onDownload).toHaveBeenCalledExactlyOnceWith(files, '4bit', expect.any(Object));
  fireEvent.click(screen.getByRole('button', { name: 'Open folder 4bit' }));
  expect(screen.getByText('/4bit/')).toBeVisible();
  expect(screen.getByRole('checkbox', { name: /model.safetensors/ })).toBeVisible();
});

it('keeps shared Engram dependencies in a native shard group without marking it incomplete', () => {
  const files = [file('TQ3/model-00001-of-00003.tq'), file('TQ3/model-00003-of-00003.tq'),
    file('Engram/model-Engram-MXFP8-00002-of-00003.tq')];
  const variants: HubModelVariant[] = [{ id: 'tq:TQ3/model', label: 'TQ3/model', format: 'tq',
    files: files.map(item => item.name), byte_size: 3072,
    configuration: { status: 'unknown', recommendation: 'unknown', reasons: [] } }];
  const onDownload = vi.fn();
  render(<RepositoryFiles files={files} variants={variants} disabled={false} onDownload={onDownload} tr={tr} expanded />);
  fireEvent.click(screen.getByRole('button', { name: 'Open folder TQ3' }));
  expect(screen.getAllByRole('checkbox')).toHaveLength(1);
  expect(screen.getByText('3/3 shards')).toBeVisible();
  fireEvent.click(screen.getByRole('checkbox', { name: /model.tq/ }));
  fireEvent.click(screen.getByRole('button', { name: 'Download 3 selected files' }));
  expect(onDownload.mock.calls[0][0]).toEqual(files);
  fireEvent.click(screen.getByRole('button', { name: 'Up' }));
  fireEvent.click(screen.getByRole('button', { name: 'Open folder Engram' }));
  expect(screen.getAllByRole('checkbox')).toHaveLength(1);
  expect(screen.getByRole('checkbox', { name: /model-Engram-MXFP8.tq/ })).toBeVisible();
});

it('keeps disabled downloads disabled and renders small file sizes without zero GiB', () => {
  render(<RepositoryFiles files={[file('folder/model.gguf'), { name: 'config.json', byte_size: 12 }]} disabled onDownload={vi.fn()} tr={tr} expanded />);
  expect(screen.getByRole('button', { name: 'Download folder folder' })).toBeDisabled();
  expect(screen.getByText('12 B')).toBeVisible();
  fireEvent.click(screen.getByRole('checkbox', { name: /config.json/ }));
  expect(screen.getByRole('button', { name: 'Download 1 selected files' })).toBeDisabled();
});
