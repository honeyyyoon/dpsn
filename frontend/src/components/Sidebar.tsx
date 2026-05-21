import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { MODELS } from '../data';
import type { UiJob } from '../types';
import Icon from './Icon';

function JobStatusBadge({ status }: { status: UiJob['status'] }) {
  if (status === 'done')
    return <span style={{ width: 18, height: 18, borderRadius: '50%', background: 'color-mix(in oklab, var(--success) 15%, var(--panel))', color: 'var(--success)', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}><Icon name="check" size={10} strokeWidth={2.5}/></span>;
  if (status === 'running')
    return <span style={{ width: 18, height: 18, borderRadius: '50%', background: 'var(--accent-50)', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}><span style={{ width: 6, height: 6, borderRadius: '50%', background: 'var(--accent)', animation: 'pulse-dot 1.2s infinite' }}/></span>;
  if (status === 'failed')
    return <span style={{ width: 18, height: 18, borderRadius: '50%', background: 'color-mix(in oklab, var(--danger) 15%, var(--panel))', color: 'var(--danger)', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0, fontSize: 10, fontWeight: 700 }}>!</span>;
  if (status === 'cancelled')
    return <span style={{ width: 18, height: 18, borderRadius: '50%', background: 'var(--bg-sunken)', color: 'var(--text-dim)', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}><Icon name="x" size={8} strokeWidth={2.5}/></span>;
  return <span style={{ width: 18, height: 18, borderRadius: '50%', border: '1.5px dashed var(--border-strong)', flexShrink: 0 }}/>;
}

interface MenuItem {
  label: string;
  danger?: boolean;
  disabled?: boolean;
  action: () => void;
}

function DropdownMenu({ items, onClose }: { items: MenuItem[]; onClose: () => void }) {
  return (
    <>
      <div
        style={{ position: 'fixed', inset: 0, zIndex: 100 }}
        onClick={(e) => { e.stopPropagation(); onClose(); }}
      />
      <div style={{
        position: 'absolute', right: 4, top: 'calc(100% - 4px)',
        background: 'var(--panel)',
        border: '1px solid var(--border)',
        borderRadius: 'var(--r-md)',
        boxShadow: '0 8px 24px rgba(0,0,0,0.18)',
        zIndex: 101,
        minWidth: 148,
        overflow: 'hidden',
        padding: '4px 0',
      }}>
        {items.map((item, i) => (
          <button
            key={i}
            disabled={item.disabled}
            onClick={(e) => { e.stopPropagation(); item.action(); onClose(); }}
            style={{
              display: 'block', width: '100%', textAlign: 'left',
              padding: '7px 12px', fontSize: 13, border: 'none', cursor: item.disabled ? 'default' : 'pointer',
              color: item.danger ? 'var(--danger)' : item.disabled ? 'var(--text-dim)' : 'var(--text)',
              background: 'transparent',
            }}
            onMouseEnter={(e) => { if (!item.disabled) (e.currentTarget as HTMLButtonElement).style.background = 'var(--bg-sunken)'; }}
            onMouseLeave={(e) => { (e.currentTarget as HTMLButtonElement).style.background = 'transparent'; }}
          >
            {item.label}
          </button>
        ))}
      </div>
    </>
  );
}

interface JobItemProps {
  job: UiJob;
  active: boolean;
  onClick: () => void;
  onJobTerminate: (jobId: string) => void;
}

function JobItem({ job, active, onClick, onJobTerminate }: JobItemProps) {
  const [hover, setHover] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);

  const isRunning = job.status === 'running' || job.status === 'pending';
  const statusLabel: Record<UiJob['status'], string> = {
    done: '완료', running: '실행 중', pending: '대기중', failed: '실패', cancelled: '취소됨',
  };
  const modelCount = job.modelIds?.length || 0;
  const modelSummary = modelCount === 1
    ? (MODELS.find(m => m.id === job.modelIds[0])?.name || '')
    : `방법 ${modelCount}개`;

  const getMenuItems = (): MenuItem[] => {
    if (isRunning) return [
      { label: '중단', danger: true, action: () => {
        if (window.confirm(`"${job.wsi}" 작업을 중단하시겠습니까?`)) onJobTerminate(job.id);
      }},
    ];
    if (job.status === 'done') return [
      { label: '결과 보기', action: onClick },
      { label: '모델 추가', disabled: true, action: () => {} },
      { label: '결과 다운로드', disabled: true, action: () => {} },
      { label: '삭제', danger: true, action: () => {
        if (window.confirm(`"${job.wsi}" 작업을 삭제하시겠습니까?`)) onJobTerminate(job.id);
      }},
    ];
    return [
      { label: '재시도', disabled: true, action: () => {} },
      { label: '삭제', danger: true, action: () => {
        if (window.confirm(`"${job.wsi}" 작업을 삭제하시겠습니까?`)) onJobTerminate(job.id);
      }},
    ];
  };

  return (
    <div
      onClick={onClick}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      style={{
        position: 'relative',
        display: 'flex', alignItems: 'flex-start', gap: 12,
        padding: '12px 14px',
        paddingRight: 36,
        borderRadius: 'var(--r-md)',
        background: active ? 'var(--accent-50)' : hover ? 'var(--bg-sunken)' : 'transparent',
        cursor: 'pointer',
        margin: '2px 0',
      }}
    >
      <JobStatusBadge status={job.status}/>
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ fontSize: 15, fontWeight: 600, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', color: active ? 'var(--accent-600)' : 'var(--text)' }}>
          {job.wsi}
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, color: 'var(--text-muted)', marginTop: 3, whiteSpace: 'nowrap' }}>
          <span style={{ overflow: 'hidden', textOverflow: 'ellipsis' }}>{modelSummary}</span>
          <span>·</span>
          <span>{statusLabel[job.status]}</span>
          {job.status === 'running' && job.progress != null && (
            <><span>·</span><span className="num">{Math.round(job.progress * 100)}%</span></>
          )}
          <span style={{ marginLeft: 'auto', color: 'var(--text-dim)' }}>{job.when}</span>
        </div>
        {job.status === 'running' && job.progress != null && (
          <div style={{ height: 3, background: 'var(--bg-sunken)', borderRadius: 999, marginTop: 6, overflow: 'hidden' }}>
            <div style={{ width: `${job.progress * 100}%`, height: '100%', background: 'var(--accent)', borderRadius: 999, transition: 'width 300ms' }}/>
          </div>
        )}
      </div>

      <button
        aria-label="작업 메뉴 열기"
        onClick={(e) => { e.stopPropagation(); setMenuOpen(m => !m); }}
        style={{
          position: 'absolute', right: 6, top: '50%', transform: 'translateY(-50%)',
          opacity: hover || menuOpen ? 1 : 0,
          pointerEvents: hover || menuOpen ? 'auto' : 'none',
          transition: 'opacity 120ms',
          padding: 4, border: 'none', borderRadius: 'var(--r-sm)',
          background: menuOpen ? 'var(--bg-sunken)' : 'transparent',
          cursor: 'pointer', color: 'var(--text-muted)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
        }}
      >
        <Icon name="dot-menu" size={14}/>
      </button>

      {menuOpen && <DropdownMenu items={getMenuItems()} onClose={() => setMenuOpen(false)}/>}
    </div>
  );
}

interface SidebarProps {
  jobs: UiJob[];
  activeJobId: string | null;
  onSelectJob: (jobId: string) => void;
  onJobTerminate: (jobId: string) => void;
}

export default function Sidebar({ jobs, activeJobId, onSelectJob, onJobTerminate }: SidebarProps) {
  const navigate = useNavigate();
  return (
    <aside className="sidebar">
      <button
        className="sb-brand"
        style={{ padding: '12px 16px', justifyContent: 'center', width: '100%', cursor: 'pointer' }}
        onClick={() => navigate('/')}
        title="메인 화면으로"
      >
        <img src="/mainImage.png" alt="Stain Normalization 비교 플랫폼" style={{ height: 36, width: 'auto', display: 'block' }}/>
      </button>

      <div className="sb-body">
        <div className="sb-section">
          <span style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <span style={{ width: 18, height: 18, display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
              <Icon name="layers" size={16} strokeWidth={2}/>
            </span>
            작업 목록
          </span>
        </div>
        {jobs.length === 0 && (
          <div style={{ padding: '16px 14px', fontSize: 13, color: 'var(--text-dim)', textAlign: 'center' }}>
            아직 작업이 없습니다.
          </div>
        )}
        {jobs.map(j => (
          <JobItem
            key={j.id}
            job={j}
            active={j.id === activeJobId}
            onClick={() => onSelectJob(j.id)}
            onJobTerminate={onJobTerminate}
          />
        ))}
      </div>
    </aside>
  );
}
