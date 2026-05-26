import { useState } from 'react';
import { MODELS } from '../data';
import type { UiJob } from '../types';
import Icon from './Icon';

interface AddModelModalProps {
  job: UiJob;
  onConfirm: (modelIds: number[]) => void;
  onCancel: () => void;
}

export function AddModelModal({ job, onConfirm, onCancel }: AddModelModalProps) {
  const available = MODELS.filter(m => !job.modelIds.includes(m.id));
  const [selected, setSelected] = useState<Set<number>>(new Set());

  const toggle = (id: number) => setSelected(prev => {
    const s = new Set(prev);
    s.has(id) ? s.delete(id) : s.add(id);
    return s;
  });

  return (
    <div style={{ position: 'fixed', inset: 0, zIndex: 200, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
      <div
        style={{ position: 'absolute', inset: 0, background: 'rgba(0,0,0,0.45)', backdropFilter: 'blur(3px)' }}
        onClick={onCancel}
      />
      <div className="card" style={{ position: 'relative', zIndex: 1, width: 480, padding: 24, display: 'flex', flexDirection: 'column', gap: 20 }}>
        <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between' }}>
          <div>
            <div style={{ fontSize: 16, fontWeight: 700, letterSpacing: '-0.01em' }}>모델 추가</div>
            <div style={{ fontSize: 13, color: 'var(--text-muted)', marginTop: 3 }}>{job.wsi}</div>
          </div>
          <button className="icon-btn" onClick={onCancel} style={{ marginTop: -2 }}>
            <Icon name="x" size={16} />
          </button>
        </div>

        {available.length === 0 ? (
          <div style={{ textAlign: 'center', color: 'var(--text-dim)', fontSize: 13, padding: '20px 0' }}>
            추가할 수 있는 모델이 없습니다.
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {available.map(m => {
              const isSelected = selected.has(m.id);
              return (
                <div
                  key={m.id}
                  onClick={() => toggle(m.id)}
                  style={{
                    display: 'flex', alignItems: 'center', gap: 12,
                    padding: '10px 14px', borderRadius: 'var(--r-md)', cursor: 'pointer',
                    border: `1.5px solid ${isSelected ? m.tint : 'var(--border)'}`,
                    background: isSelected ? `color-mix(in oklab, ${m.tint} 6%, var(--panel))` : 'var(--panel)',
                    transition: 'border-color 120ms, background 120ms',
                  }}
                >
                  <div style={{ width: 10, height: 10, borderRadius: '50%', background: m.tint, flexShrink: 0 }} />
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ fontSize: 14, fontWeight: 600 }}>{m.name}</div>
                    <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 2 }}>{m.description}</div>
                  </div>
                  <div style={{
                    width: 16, height: 16, borderRadius: 4, flexShrink: 0,
                    border: `1.5px solid ${isSelected ? m.tint : 'var(--border-strong)'}`,
                    background: isSelected ? m.tint : 'transparent',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                  }}>
                    {isSelected && <Icon name="check" size={11} color="#fff" strokeWidth={2.5} />}
                  </div>
                </div>
              );
            })}
          </div>
        )}

        <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
          <button className="btn ghost" onClick={onCancel}>취소</button>
          <button
            className="btn primary"
            disabled={selected.size === 0}
            onClick={() => onConfirm([...selected])}
          >
            <Icon name="play" size={13} />
            {selected.size > 0 ? `${selected.size}개 모델 실행` : '모델 선택'}
          </button>
        </div>
      </div>
    </div>
  );
}
