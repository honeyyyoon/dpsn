import type { Model, JobResponse, JobStatusResponse, JobResultResponse } from './types';

const BASE = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000';

export async function getModels(): Promise<Model[]> {
  const res = await fetch(`${BASE}/models`);
  if (!res.ok) throw new Error('Failed to fetch models');
  return res.json();
}

export async function createJobs(imageFile: File, modelIds: number[]): Promise<JobResponse[]> {
  const form = new FormData();
  form.append('image', imageFile);
  form.append('model_ids', modelIds.join(','));
  const res = await fetch(`${BASE}/jobs`, { method: 'POST', body: form });
  if (!res.ok) throw new Error('Failed to create jobs');
  return res.json();
}

export async function getJobStatus(jobId: string): Promise<JobStatusResponse> {
  const res = await fetch(`${BASE}/jobs/${jobId}`);
  if (!res.ok) throw new Error(`Failed to get job status: ${jobId}`);
  return res.json();
}

export async function getJobResult(jobId: string): Promise<JobResultResponse> {
  const res = await fetch(`${BASE}/jobs/${jobId}/results`);
  if (!res.ok) throw new Error(`Failed to get job result: ${jobId}`);
  return res.json();
}

export function getImageUrl(imageId: string, thumbnail: boolean = false): string {
  return `${BASE}/images/${imageId}${thumbnail ? '?thumbnail=true' : ''}`;
}
