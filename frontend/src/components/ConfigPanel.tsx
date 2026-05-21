import { useState, useEffect } from 'react';
import { MODELS } from '../data';
import type { ModelUi } from '../types';
import Icon from './Icon';
import { TissueSvg } from './WsiImage';

const PREVIEWABLE = ['image/jpeg', 'image/png', 'image/webp', 'image/tiff'];

interface UploadCardProps {
  file: File | null;
  onPick: (f?: File) => void;
  onClear: () => void;
}

export function UploadCard({ file, onPick, onClear }: UploadCardProps) {
  const [dragOver, setDrag] = useState(false);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  useEffect(() => {
    if (!file || !PREVIEWABLE.includes(file.type)) { setPreviewUrl(null); return; }
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  if (file) {
    return (
      <div className="card" style={{ padding: '24px 24px', display: 'flex', gap: 20, alignItems: 'center', height: 180 }}>
        <div style={{ width: 108, height: 108, borderRadius: 14, overflow: 'hidden', flexShrink: 0 }}>
          {previewUrl
            ? <img src={previewUrl} alt={file.name} style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
            : <TissueSvg seed={7} mode="dim"/>}
        </div>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ fontSize: 17, fontWeight: 600, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{file.name}</div>
          <div style={{ fontSize: 14, color: 'var(--text-muted)', marginTop: 6, fontFamily: 'var(--font-mono)' }}>
            {(file.size / 1024 / 1024).toFixed(1)} MB
          </div>
        </div>
        <button className="icon-btn" onClick={onClear} title="제거">
          <Icon name="x" size={18}/>
        </button>
      </div>
    );
  }

  return (
    <div
      className="card"
      style={{
        padding: 24, textAlign: 'center',
        height: 180,
        display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
        border: `1.5px dashed ${dragOver ? 'var(--accent)' : 'var(--border-strong)'}`,
        background: dragOver ? 'var(--accent-50)' : 'var(--panel)',
        transition: 'border-color 140ms, background 140ms',
      }}
      onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
      onDragLeave={() => setDrag(false)}
      onDrop={(e) => { e.preventDefault(); setDrag(false); const f = e.dataTransfer.files?.[0]; onPick(f); }}
    >
      <div style={{ width: 44, height: 44, margin: '0 auto 12px', borderRadius: 12, background: 'var(--accent-50)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--accent)' }}>
        <Icon name="upload" size={20}/>
      </div>
      <div style={{ fontSize: 16, fontWeight: 600, marginBottom: 6 }}>WSI 이미지 업로드</div>
      <div style={{ fontSize: 13, color: 'var(--text-muted)', marginBottom: 14 }}>
        파일을 끌어다 놓거나 아래 버튼을 눌러주세요
      </div>
      <button className="btn outline sm" onClick={(e) => { e.stopPropagation(); onPick(); }} style={{ margin: '0 auto' }}>
        <Icon name="upload" size={13}/> 파일 선택
      </button>
    </div>
  );
}

interface ModelCardProps {
  model: ModelUi;
  selected: boolean;
  onToggle: () => void;
}

function ModelCard({ model, selected, onToggle }: ModelCardProps) {
  const [hover, setHover] = useState(false);
  const isClassical = model.category === 'Classical';
  return (
    <div
      onClick={onToggle}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      style={{
        position: 'relative', padding: 12,
        borderRadius: 'var(--r-md)',
        border: `1.5px solid ${selected ? model.tint : 'var(--border)'}`,
        background: selected ? `color-mix(in oklab, ${model.tint} 5%, var(--panel))` : 'var(--panel)',
        cursor: 'pointer',
        transition: 'border-color 140ms, background 140ms, transform 140ms',
        transform: hover && !selected ? 'translateY(-1px)' : 'none',
        boxShadow: selected ? `0 0 0 3px color-mix(in oklab, ${model.tint} 15%, transparent)` : 'var(--shadow-sm)',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'flex-start', gap: 10 }}>
        <div style={{ width: 24, height: 24, borderRadius: 6, background: `color-mix(in oklab, ${model.tint} 15%, var(--panel))`, color: model.tint, display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
          <Icon name={isClassical ? 'cpu' : 'sparkle'} size={13}/>
        </div>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ fontSize: 15, fontWeight: 600 }}>{model.name}</div>
          <div style={{ fontSize: 13, color: 'var(--text-muted)', marginTop: 3, lineHeight: 1.5 }}>{model.description}</div>
        </div>
        <div style={{ width: 16, height: 16, borderRadius: 4, border: `1.5px solid ${selected ? model.tint : 'var(--border-strong)'}`, background: selected ? model.tint : 'transparent', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0, marginTop: 2 }}>
          {selected && <Icon name="check" size={11} color="#fff" strokeWidth={2.5}/>}
        </div>
      </div>
    </div>
  );
}

interface ModelPickerProps {
  selected: Set<number>;
  onToggle: (id: number) => void;
}

export function ModelPicker({ selected, onToggle }: ModelPickerProps) {
  const classical = MODELS.filter(m => m.category === 'Classical');
  const learning  = MODELS.filter(m => m.category === 'Learning-based');

  const categoryHeader = (title: string, badge: string) => (
    <div style={{ display: 'flex', alignItems: 'baseline', gap: 10, marginBottom: 14, whiteSpace: 'nowrap' }}>
      <div style={{ fontSize: 16, fontWeight: 700, color: 'var(--text)', letterSpacing: '-0.01em' }}>{title}</div>
      <div style={{ fontSize: 13, color: 'var(--text-dim)', fontWeight: 500 }}>{badge}</div>
    </div>
  );

  return (
    <div className="model-categories">
      <div>
        {categoryHeader('알고리즘 기반 방법', 'CPU')}
        <div style={{ display: 'grid', gap: 8 }}>
          {classical.map(m => <ModelCard key={m.id} model={m} selected={selected.has(m.id)} onToggle={() => onToggle(m.id)}/>)}
        </div>
      </div>
      <div className="model-categories-divider"/>
      <div>
        {categoryHeader('딥러닝 모델', 'GPU')}
        <div style={{ display: 'grid', gap: 8 }}>
          {learning.map(m => <ModelCard key={m.id} model={m} selected={selected.has(m.id)} onToggle={() => onToggle(m.id)}/>)}
        </div>
      </div>
    </div>
  );
}
