import type { ModelUi, MetricDef } from './types';

export const MODELS: ModelUi[] = [
  { id: 1, name: 'Reinhard',  category: 'Classical',      description: '색 분포 변환을 통한 source-target간 통계량 유지', tint: '#0ea5e9', fast: true  },
  { id: 2, name: 'Macenko',   category: 'Classical',      description: 'stain vector를 추정하고 이를 기반으로 정규화',    tint: '#14b8a6', fast: true  },
  { id: 3, name: 'Vahadane',  category: 'Classical',      description: '구조 보존이 뛰어난 stain 분리 기법',             tint: '#6366f1', fast: true  },
  { id: 4, name: 'StainGAN',  category: 'Learning-based', description: 'source-target 간 Image-to-Image 변환',         tint: '#8b5cf6', fast: false },
  { id: 5, name: 'StainNet',  category: 'Learning-based', description: '경량 CNN 기반 stain 정규화',                    tint: '#f97316', fast: false },
  { id: 6, name: 'StainSWIN', category: 'Learning-based', description: 'Vision Transformer 기반 모델',                  tint: '#ec4899', fast: false },
];

export const METRIC_DEFS: MetricDef[] = [
  { key: 'psnr', label: 'PSNR',  unit: 'dB', higherBetter: true,  desc: 'Peak signal-to-noise ratio', ref: 22   },
  { key: 'ssim', label: 'SSIM',  unit: '',   higherBetter: true,  desc: 'Structural similarity',       ref: 0.90 },
  { key: 'fid',  label: 'FID',   unit: '',   higherBetter: false, desc: 'Fréchet inception distance',  ref: 60   },
];
