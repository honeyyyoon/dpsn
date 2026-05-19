import { MODELS } from '../data';
import type { UiJob } from '../types';

interface Props {
  fileName: string;
  matchingJob: UiJob;
  onDifferent: () => void;
  onAddModels: () => void;
  onClose: () => void;
}

const STATUS_LABEL: Record<string, string> = {
  done: '완료', running: '실행 중', pending: '대기중', failed: '실패', cancelled: '취소됨',
};

export default function SameImageModal({ fileName, matchingJob, onDifferent, onAddModels, onClose }: Props) {
  const modelSummary = matchingJob.modelIds
    .map(id => MODELS.find(m => m.id === id)?.name ?? '')
    .filter(Boolean)
    .join(' · ');

  return (
    <>
      <div
        style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.4)', zIndex: 200, backdropFilter: 'blur(2px)' }}
        onClick={onClose}
      />
      <div style={{
        position: 'fixed', left: '50%', top: '50%',
        transform: 'translate(-50%, -50%)',
        background: 'var(--panel)',
        border: '1px solid var(--border)',
        borderRadius: 'var(--r-md)',
        boxShadow: '0 20px 60px rgba(0,0,0,0.3)',
        zIndex: 201, width: 400, padding: 24,
      }}>
        <div style={{ fontSize: 16, fontWeight: 700, marginBottom: 8 }}>이 이미지가 맞나요?</div>
        <div style={{ fontSize: 13, color: 'var(--text-muted)', marginBottom: 16, lineHeight: 1.5 }}>
          <span style={{ fontWeight: 600, color: 'var(--text)' }}>"{fileName}"</span>로 실행한 이전 작업이 있어요.
        </div>

        <div style={{
          padding: '10px 12px', borderRadius: 'var(--r-md)',
          border: '1px solid var(--border)', background: 'var(--bg-sunken)', marginBottom: 20,
        }}>
          <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 4, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
            {matchingJob.wsi}
          </div>
          <div style={{ fontSize: 11, color: 'var(--text-dim)' }}>
            {modelSummary} · {STATUS_LABEL[matchingJob.status] ?? matchingJob.status} · {matchingJob.when}
          </div>
        </div>

        <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
          <button className="btn ghost" onClick={onDifferent}>다른 이미지예요</button>
          <button className="btn primary" onClick={onAddModels}>네, 모델 추가하기 →</button>
        </div>
      </div>
    </>
  );
}
