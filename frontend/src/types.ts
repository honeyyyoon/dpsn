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
  status: 'pending' | 'running' | 'done' | 'failed' | 'cancelled';
  progress: number;
  message: string;
  error_detail?: string;
}

export interface JobResultResponse {
  job_id: string;
  status: string;
  result_image_id: string;
  metrics: {
    ssim: number;
    psnr: number;
    fid: number;
    gaussian_color_dist?: number;
    gaussian_color_gain?: number;
  };
  elapsed_seconds: number;
}

// UI 전용 확장 타입
export interface ModelUi extends Model {
  tint: string;
  fast: boolean;
}

export interface MetricDef {
  key: keyof JobResultResponse['metrics'];
  label: string;
  unit: string;
  higherBetter: boolean;
  desc: string;
  ref: number;
  precision?: number;
}

export interface JobResult {
  metrics: JobResultResponse['metrics'];
  result_image_id: string;
  elapsed_seconds?: number;
}

export type JobStatus = 'pending' | 'running' | 'done' | 'failed' | 'cancelled';

export interface FailedJobInfo {
  message: string;
  error_detail?: string;
}

export interface JobListItem {
  id: string;
  model_id: number;
  status: JobStatus;
  progress: number;
  message?: string;
  error_detail?: string;
  result_image_id: string | null;
  metrics: JobResultResponse['metrics'] | null;
  elapsed_seconds: number | null;
}

export interface JobGroupResponse {
  group_id: string;
  wsi_name: string;
  image_id: string;
  created_at: string;
  jobs: JobListItem[];
}

export interface UiJob {
  id: string;
  wsi: string;
  modelIds: number[];
  status: JobStatus;
  when: string;
  progress?: number;
  src_image_id?: string;
  results?: Record<number, JobResult>;
  failedJobInfo?: Record<number, FailedJobInfo>;
  imageFingerprint?: string;
}
