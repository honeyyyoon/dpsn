// 백엔드 Pydantic 스키마와 1:1 대응
export interface Model {
  id: number;
  name: string;
  category: 'Classical' | 'Learning-based';
  description: string;
}

export interface JobResponse {
  job_id: string;
  image_id: string;
}

export interface JobStatusResponse {
  job_id: string;
  status: 'pending' | 'running' | 'done' | 'failed';
}

export interface JobResultResponse {
  job_id: string;
  status: string;
  result_image_id: string;
  metrics: { ssim: number; psnr: number; fid: number };
}

// UI 전용 확장 타입
export interface ModelUi extends Model {
  tint: string;
  fast: boolean;
}

export interface MetricDef {
  key: 'psnr' | 'ssim' | 'fid';
  label: string;
  unit: string;
  higherBetter: boolean;
  desc: string;
}

export interface JobResult {
  metrics: JobResultResponse['metrics'];
  result_image_id: string;
}

export type JobStatus = 'pending' | 'running' | 'done' | 'failed';

export interface UiJob {
  id: string;
  wsi: string;
  modelIds: number[];
  status: JobStatus;
  when: string;
  progress?: number;
  src_image_id?: string;
  results?: Record<number, JobResult>;
}
