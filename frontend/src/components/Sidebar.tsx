import { MODELS } from '../data';
import type { UiJob } from '../types';
import Icon from './Icon';

interface JobStatusBadgeProps {
  status: UiJob['status'];
}

function JobStatusBadge({ status }: JobStatusBadgeProps) {
  if (status === 'done') {
    return (
      <span style={{ width: 14, height: 14, borderRadius: '50%', background: 'color-mix(in oklab, var(--success) 15%, var(--panel))', color: 'var(--success)', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
        <Icon name="check" size={10} strokeWidth={2.5}/>
      </span>
    );
  }
  if (status === 'running') {
    return (
      <span style={{ width: 14, height: 14, borderRadius: '50%', background: 'var(--accent-50)', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
        <span style={{ width: 6, height: 6, borderRadius: '50%', background: 'var(--accent)', animation: 'pulse-dot 1.2s infinite' }}/>
      </span>
    );
  }
  if (status === 'failed') {
    return <span style={{ width: 14, height: 14, borderRadius: '50%', background: 'color-mix(in oklab, var(--danger) 15%, var(--panel))', color: 'var(--danger)', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0, fontSize: 10, fontWeight: 700 }}>!</span>;
  }
  return (
    <span style={{ width: 14, height: 14, borderRadius: '50%', border: '1.5px dashed var(--border-strong)', flexShrink: 0 }}/>
  );
}

interface JobItemProps {
  job: UiJob;
  active: boolean;
  onClick: () => void;
}

function JobItem({ job, active, onClick }: JobItemProps) {
  const statusLabel: Record<UiJob['status'], string> = {
    done: '완료', running: '실행 중', pending: '대기중', failed: '실패',
  };
  const modelCount = job.modelIds?.length || 0;
  const modelSummary = modelCount === 1
    ? (MODELS.find(m => m.id === job.modelIds[0])?.name || '')
    : `방법 ${modelCount}개`;

  return (
    <div
      onClick={onClick}
      style={{
        display: 'flex', alignItems: 'flex-start', gap: 10,
        padding: '10px 12px',
        borderRadius: 'var(--r-md)',
        background: active ? 'var(--accent-50)' : 'transparent',
        cursor: 'pointer',
        margin: '1px 0',
      }}
      onMouseEnter={(e) => { if (!active) (e.currentTarget as HTMLDivElement).style.background = 'var(--bg-sunken)'; }}
      onMouseLeave={(e) => { if (!active) (e.currentTarget as HTMLDivElement).style.background = 'transparent'; }}
    >
      <JobStatusBadge status={job.status}/>
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ fontSize: 13, fontWeight: 500, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', color: active ? 'var(--accent-600)' : 'var(--text)' }}>
          {job.wsi}
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 11, color: 'var(--text-dim)', marginTop: 2, whiteSpace: 'nowrap' }}>
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
    </div>
  );
}

interface SidebarProps {
  jobs: UiJob[];
  activeJobId: string | null;
  onSelectJob: (jobId: string) => void;
  onNewRun: () => void;
}

export default function Sidebar({ jobs, activeJobId, onSelectJob, onNewRun }: SidebarProps) {
  return (
    <aside className="sidebar">
      <div className="sb-brand">
        <div className="mark">S</div>
        <div style={{ minWidth: 0 }}>
          <div className="name">Stain Normalization</div>
          <div className="org">비교 플랫폼</div>
        </div>
      </div>

      <div style={{ padding: '12px 12px 4px' }}>
        <button className="btn outline" style={{ width: '100%', justifyContent: 'flex-start' }} onClick={onNewRun}>
          <Icon name="plus" size={14}/> 새 작업
        </button>
      </div>

      <div className="sb-body">
        <div className="sb-section">작업</div>
        {jobs.length === 0 && (
          <div style={{ padding: '16px 12px', fontSize: 12, color: 'var(--text-dim)', textAlign: 'center' }}>
            아직 작업이 없습니다.
          </div>
        )}
        {jobs.map(j => (
          <JobItem key={j.id} job={j} active={j.id === activeJobId} onClick={() => onSelectJob(j.id)}/>
        ))}
      </div>
    </aside>
  );
}
