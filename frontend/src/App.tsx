import { useState, useEffect, useRef } from "react";
import type { ReactNode } from "react";
import { useNavigate, useMatch } from "react-router-dom";
import "./styles.css";
import { MODELS } from "./data";
import type { UiJob, JobResult, FailedJobInfo } from "./types";
import { createJobs, getJobStatus, getJobResult, deleteJob, fetchJobs } from "./api";
import Sidebar from "./components/Sidebar";
import Icon from "./components/Icon";
import { UploadCard, ModelPicker } from "./components/ConfigPanel";
import { SingleResult, MultiDashboard } from "./components/ResultsViews";

function Topbar({
  file,
  selectedCount,
  onRun,
  running,
  onReset,
  title,
  viewingJob,
  onBack,
}: {
  file: File | null;
  selectedCount: number;
  onRun: () => void;
  running: boolean;
  onReset: () => void;
  title: string;
  viewingJob: boolean;
  onBack: () => void;
}) {
  const canRun = file && selectedCount > 0;
  return (
    <div className="topbar">
      <div className="topbar-left">
        {viewingJob && (
          <button
            className="icon-btn"
            onClick={onBack}
            style={{ flexShrink: 0 }}
          >
            <Icon name="chevron-left" size={16} />
          </button>
        )}
        <div
          style={{
            fontSize: 20,
            fontWeight: 600,
            letterSpacing: "-0.02em",
            whiteSpace: "nowrap",
            overflow: "hidden",
            textOverflow: "ellipsis",
            minWidth: 0,
            flex: 1,
          }}
        >
          {title}
        </div>
      </div>
      <div className="topbar-right">
        {running && (
          <div
            className="chip"
            style={{
              background: "var(--accent-50)",
              color: "var(--accent-600)",
            }}
          >
            <span
              style={{
                width: 6,
                height: 6,
                borderRadius: "50%",
                background: "var(--accent)",
                animation: "pulse-dot 1.4s infinite",
                display: "inline-block",
              }}
            />
            분석 중…
          </div>
        )}
        {!viewingJob && (
          <>
            <button className="btn ghost" onClick={onReset}>
              <Icon name="history" size={15} /> 초기화
            </button>
            <button
              className="btn primary"
              disabled={!canRun || running}
              onClick={onRun}
            >
              <Icon name="play" size={14} />
              {running
                ? "실행 중…"
                : selectedCount > 1
                  ? `정규화 ${selectedCount}개 실행`
                  : "정규화 실행"}
            </button>
          </>
        )}
      </div>
    </div>
  );
}

function StatusLine({ ok, children }: { ok?: boolean; children: ReactNode }) {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 8,
        fontSize: 14,
        color: ok ? "var(--text)" : "var(--text-muted)",
      }}
    >
      <span
        style={{
          width: 14,
          height: 14,
          borderRadius: "50%",
          background: ok
            ? "color-mix(in oklab, var(--success) 15%, var(--panel))"
            : "var(--bg-sunken)",
          color: ok ? "var(--success)" : "var(--text-dim)",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          flexShrink: 0,
        }}
      >
        {ok ? (
          <Icon name="check" size={10} strokeWidth={2.5} />
        ) : (
          <span
            style={{
              width: 4,
              height: 4,
              background: "currentColor",
              borderRadius: "50%",
            }}
          />
        )}
      </span>
      <span
        style={{
          overflow: "hidden",
          textOverflow: "ellipsis",
          whiteSpace: "nowrap",
        }}
      >
        {children}
      </span>
    </div>
  );
}

function ConfigColumn({
  file,
  onPickFile,
  onClearFile,
  selected,
  onToggleModel,
  onRun,
  running,
  fileInputRef,
}: {
  file: File | null;
  onPickFile: (f: File) => void;
  onClearFile: () => void;
  selected: Set<number>;
  onToggleModel: (id: number) => void;
  onRun: () => void;
  running: boolean;
  fileInputRef: { current: HTMLInputElement | null };
}) {
  const canRun = file && selected.size > 0;
  const selectedModels = [...selected]
    .map((id) => MODELS.find((m) => m.id === id)!)
    .filter(Boolean);

  return (
    <div
      style={{
        flex: 1,
        minWidth: 0,
        minHeight: 0,
        display: "flex",
        flexDirection: "column",
        background: "var(--bg)",
        overflow: "hidden",
      }}
    >
      <input
        ref={fileInputRef}
        type="file"
        accept=".svs,.tiff,.tif,.ndpi,.scn,.mrxs,.jpg,.jpeg,.png"
        style={{ display: "none" }}
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) onPickFile(f);
        }}
      />
      <div style={{ flex: 1, overflow: "auto" }}>
        <div
          style={{
            maxWidth: 960,
            width: "100%",
            margin: "0 auto",
            padding: "28px 32px 28px",
          }}
        >
          <div style={{ display: "flex", flexDirection: "column", gap: 28 }}>
            <div>
              <div
                style={{
                  fontSize: 15,
                  fontWeight: 600,
                  color: "var(--text-muted)",
                  letterSpacing: "-0.01em",
                  marginBottom: 14,
                }}
              >
                1. WSI 이미지 업로드
              </div>
              <UploadCard
                file={file}
                onPick={(f) =>
                  f ? onPickFile(f) : fileInputRef.current?.click()
                }
                onClear={onClearFile}
              />
            </div>
            <div>
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                  gap: 8,
                  marginBottom: 14,
                  minHeight: 22,
                }}
              >
                <div
                  style={{
                    fontSize: 15,
                    fontWeight: 600,
                    color: "var(--text-muted)",
                    letterSpacing: "-0.01em",
                  }}
                >
                  2. 정규화 방법 선택
                </div>
                <span
                  className="chip accent"
                  style={{
                    visibility: selected.size > 0 ? "visible" : "hidden",
                    fontSize: 14,
                    height: 28,
                    padding: "0 12px",
                  }}
                >
                  {selected.size}개 선택됨
                </span>
              </div>
              <ModelPicker selected={selected} onToggle={onToggleModel} />
            </div>
          </div>
        </div>
      </div>

      <div
        style={{
          background: "color-mix(in oklab, var(--panel) 85%, transparent)",
          backdropFilter: "blur(8px)",
          borderTop: "1px solid var(--border)",
          padding: "14px 32px",
          display: "flex",
          alignItems: "center",
          gap: 16,
          flexShrink: 0,
        }}
      >
        <div
          style={{
            maxWidth: 960,
            width: "100%",
            margin: "0 auto",
            display: "flex",
            alignItems: "center",
            gap: 16,
          }}
        >
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: 16,
              flex: 1,
              minWidth: 0,
            }}
          >
            {file ? (
              <StatusLine ok>{file.name}</StatusLine>
            ) : (
              <StatusLine>WSI 이미지가 없습니다</StatusLine>
            )}
            <div style={{ width: 1, height: 16, background: "var(--border-strong)", flexShrink: 0 }} />
            {selected.size > 0 ? (
              <StatusLine ok>
                방법 {selected.size}개:{" "}
                {selectedModels.map((m) => m.name).join(", ")}
              </StatusLine>
            ) : (
              <StatusLine>선택된 방법 없음</StatusLine>
            )}
          </div>
          <button
            className="btn primary lg"
            disabled={!canRun || running}
            onClick={onRun}
            style={{ flexShrink: 0 }}
          >
            <Icon name="play" size={14} />
            {running
              ? "실행 중…"
              : selected.size > 1
                ? `정규화 ${selected.size}개 실행`
                : "정규화 실행"}
          </button>
        </div>
      </div>
    </div>
  );
}

const MOCK_JOBS: UiJob[] = [
  {
    id: "mock-run",
    wsi: "CAMELYON17-042",
    modelIds: [1, 2],
    status: "running",
    when: "now",
    progress: 60,
  },
  {
    id: "mock-pending",
    wsi: "PAIP-liver-089",
    modelIds: [5],
    status: "pending",
    when: "대기중",
  },
  // 1 model
  {
    id: "mock-1m",
    wsi: "PAIP-ovary-014",
    modelIds: [5],
    status: "done",
    when: "30m",
    results: {
      5: { metrics: { ssim: 0.912, psnr: 31.47, fid: 9.8 }, result_image_id: "", elapsed_seconds: 91 },
    },
  },
  // 2 models — PSNR best: Macenko, SSIM best: Reinhard
  {
    id: "mock-2m",
    wsi: "GTEx-stomach-5",
    modelIds: [1, 2],
    status: "done",
    when: "1h",
    results: {
      1: { metrics: { ssim: 0.934, psnr: 26.81, fid: 13.4 }, result_image_id: "", elapsed_seconds: 18 },
      2: { metrics: { ssim: 0.897, psnr: 29.43, fid: 11.7 }, result_image_id: "", elapsed_seconds: 24 },
    },
  },
  // 3 models — PSNR best: StainNet, SSIM best: Reinhard, FID best: Macenko
  {
    id: "mock-3m",
    wsi: "TCGA-BRCA-A2K4",
    modelIds: [1, 2, 5],
    status: "done",
    when: "2h",
    results: {
      1: { metrics: { ssim: 0.948, psnr: 25.67, fid: 14.2 }, result_image_id: "", elapsed_seconds: 15 },
      2: { metrics: { ssim: 0.876, psnr: 27.34, fid: 8.9  }, result_image_id: "", elapsed_seconds: 21 },
      5: { metrics: { ssim: 0.902, psnr: 32.15, fid: 11.3 }, result_image_id: "", elapsed_seconds: 94 },
    },
  },
  // 4 models — PSNR best: StainNet, SSIM best: Vahadane, FID best: Macenko
  {
    id: "mock-4m",
    wsi: "TCGA-LUAD-B41C",
    modelIds: [1, 2, 3, 5],
    status: "done",
    when: "3h",
    results: {
      1: { metrics: { ssim: 0.883, psnr: 27.14, fid: 15.8 }, result_image_id: "", elapsed_seconds: 12 },
      2: { metrics: { ssim: 0.871, psnr: 26.58, fid: 7.4  }, result_image_id: "", elapsed_seconds: 19 },
      3: { metrics: { ssim: 0.961, psnr: 24.92, fid: 12.3 }, result_image_id: "", elapsed_seconds: 28 },
      5: { metrics: { ssim: 0.914, psnr: 33.47, fid: 10.1 }, result_image_id: "", elapsed_seconds: 87 },
    },
  },
  // 5 models — PSNR best: StainNet, SSIM best: Vahadane, FID best: StainGAN
  {
    id: "mock-5m",
    wsi: "CPTAC-CCRCC-C3L-00004",
    modelIds: [1, 2, 3, 4, 5],
    status: "done",
    when: "1d",
    results: {
      1: { metrics: { ssim: 0.881, psnr: 27.52, fid: 15.1 }, result_image_id: "", elapsed_seconds: 14 },
      2: { metrics: { ssim: 0.869, psnr: 26.34, fid: 13.6 }, result_image_id: "", elapsed_seconds: 22 },
      3: { metrics: { ssim: 0.963, psnr: 23.87, fid: 12.4 }, result_image_id: "", elapsed_seconds: 31 },
      4: { metrics: { ssim: 0.921, psnr: 29.64, fid: 6.2  }, result_image_id: "", elapsed_seconds: 143 },
      5: { metrics: { ssim: 0.934, psnr: 34.21, fid: 9.7  }, result_image_id: "", elapsed_seconds: 89 },
    },
  },
  // 6 models — PSNR best: StainNet, SSIM best: Vahadane, FID best: StainGAN
  {
    id: "mock-6m",
    wsi: "TCGA-OV-A5KX",
    modelIds: [1, 2, 3, 4, 5, 6],
    status: "done",
    when: "2d",
    results: {
      1: { metrics: { ssim: 0.874, psnr: 27.31, fid: 16.2 }, result_image_id: "", elapsed_seconds: 11 },
      2: { metrics: { ssim: 0.858, psnr: 26.04, fid: 14.5 }, result_image_id: "", elapsed_seconds: 18 },
      3: { metrics: { ssim: 0.957, psnr: 23.41, fid: 13.1 }, result_image_id: "", elapsed_seconds: 27 },
      4: { metrics: { ssim: 0.918, psnr: 28.76, fid: 5.8  }, result_image_id: "", elapsed_seconds: 158 },
      5: { metrics: { ssim: 0.929, psnr: 35.08, fid: 10.3 }, result_image_id: "", elapsed_seconds: 92 },
      6: { metrics: { ssim: 0.943, psnr: 31.54, fid: 8.6  }, result_image_id: "", elapsed_seconds: 234 },
    },
  },
  {
    id: "mock-partial-fail",
    wsi: "TCGA-BRCA-A3L1",
    modelIds: [1, 2, 5],
    status: "done",
    when: "1d",
    results: {
      1: { metrics: { ssim: 0.934, psnr: 26.81, fid: 13.4 }, result_image_id: "", elapsed_seconds: 18 },
    },
    failedJobInfo: {
      2: { message: "오류가 발생했습니다.\n문제가 지속되면 담당자에게 문의해주세요.", error_detail: "No tissue found in slide thumbnail: /data/uploads/TCGA-BRCA-A3L1.tif" },
      5: { message: "오류가 발생했습니다.\n문제가 지속되면 담당자에게 문의해주세요.", error_detail: "StainNet checkpoint not found: /data/checkpoints/stainnet_v2.pth" },
    },
  },
  {
    id: "mock-fail",
    wsi: "NLST-lung-00891",
    modelIds: [2, 5],
    status: "failed",
    when: "2d",
    failedJobInfo: {
      2: { message: "오류가 발생했습니다.\n문제가 지속되면 담당자에게 문의해주세요.", error_detail: "Macenko pipeline requires a target image." },
      5: { message: "오류가 발생했습니다.\n문제가 지속되면 담당자에게 문의해주세요.", error_detail: "StainNet checkpoint not found: /data/checkpoints/stainnet_latest.pth" },
    },
  },
  {
    id: "mock-cancel",
    wsi: "PAIP-colon-112",
    modelIds: [1, 3],
    status: "cancelled",
    when: "3d",
  },
];

function formatRelativeTime(isoStr: string): string {
  const normalized = isoStr.includes('T') ? isoStr : isoStr.replace(' ', 'T') + 'Z';
  const diff = Math.floor((Date.now() - new Date(normalized).getTime()) / 1000);
  if (diff < 60) return '방금';
  if (diff < 3600) return `${Math.floor(diff / 60)}분`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h`;
  return `${Math.floor(diff / 86400)}d`;
}

export default function App() {
  const navigate = useNavigate();
  const jobMatch = useMatch('/jobs/:jobId');
  const activeJobId = jobMatch?.params.jobId ?? null;

  const [file, setFile] = useState<File | null>(null);
  const [selected, setSelected] = useState<Set<number>>(new Set());
  const [running, setRunning] = useState(false);
  const [jobs, setJobs] = useState<UiJob[]>(MOCK_JOBS);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const pollingRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const toggleModel = (id: number) =>
    setSelected((prev) => {
      const s = new Set(prev);
      s.has(id) ? s.delete(id) : s.add(id);
      return s;
    });

  const reset = () => {
    setFile(null);
    setSelected(new Set());
    setRunning(false);
    navigate('/');
  };

  const handleJobTerminate = async (jobId: string) => {
    const job = jobs.find(j => j.id === jobId);
    if (!job) return;
    const isRunning = job.status === 'running' || job.status === 'pending';
    try { await deleteJob(jobId); } catch { /* 서버에 없어도 UI 반영 */ }
    if (isRunning) {
      if (pollingRef.current) { clearInterval(pollingRef.current); pollingRef.current = null; }
      setRunning(false);
      setJobs(prev => prev.map(j => j.id === jobId ? { ...j, status: 'cancelled' } : j));
    } else {
      setJobs(prev => prev.filter(j => j.id !== jobId));
      if (activeJobId === jobId) navigate('/');
    }
  };

  const startPolling = (uiJobId: string, jobIds: string[], modelIds: number[]) => {
    const newResults: Record<number, JobResult> = {};
    const newFailedInfo: Record<number, FailedJobInfo> = {};
    const finishedSet = new Set<number>();

    pollingRef.current = setInterval(async () => {
      let progress = 0;
      for (let i = 0; i < jobIds.length; i++) {
        const jobId = jobIds[i];
        const modelId = modelIds[i];
        if (finishedSet.has(modelId)) { progress += 100; continue; }
        try {
          const status = await getJobStatus(jobId);
          if (status.status === "done") {
            const result = await getJobResult(jobId);
            newResults[modelId] = { metrics: result.metrics, result_image_id: result.result_image_id, elapsed_seconds: result.elapsed_seconds };
            finishedSet.add(modelId);
            progress += 100;
          } else if (status.status === "failed") {
            newFailedInfo[modelId] = { message: status.message, error_detail: status.error_detail };
            finishedSet.add(modelId);
            progress += 100;
          } else {
            progress += status.progress;
          }
        } catch (err) {
          console.warn("Polling error, will retry:", err);
        }
      }

      setJobs((prev) =>
        prev.map((j) => j.id === uiJobId ? { ...j, progress: progress / jobIds.length } : j),
      );

      if (finishedSet.size >= jobIds.length) {
        if (pollingRef.current) clearInterval(pollingRef.current);
        setRunning(false);
        const allFailed = Object.keys(newFailedInfo).length === jobIds.length;
        setJobs((prev) =>
          prev.map((j) =>
            j.id === uiJobId
              ? {
                  ...j,
                  status: allFailed ? "failed" : "done",
                  results: { ...(j.results ?? {}), ...newResults },
                  failedJobInfo: { ...(j.failedJobInfo ?? {}), ...newFailedInfo },
                  when: "방금",
                }
              : j,
          ),
        );
        navigate(`/jobs/${uiJobId}`);
      }
    }, 1500);
  };

  const runAsNew = async () => {
    if (!file || selected.size === 0) return;
    setRunning(true);
    const wsiName = file.name.replace(/\.[^/.]+$/, "");
    const modelIds = [...selected];
    const tempId = `uploading-${Date.now()}`;
    setJobs((prev) => [{
      id: tempId, wsi: wsiName, modelIds,
      status: "pending" as const, when: "업로드 중", progress: 0,
    }, ...prev]);
    try {
      const responses = await createJobs(file, modelIds);
      const jobIds = responses.map((r) => r.job_id);
      const uiJobId = jobIds[0];
      setJobs((prev) => prev.map(j => j.id === tempId
        ? { ...j, id: uiJobId, status: "running" as const, when: "now", src_image_id: responses[0].image_id }
        : j
      ));
      startPolling(uiJobId, jobIds, modelIds);
    } catch (err) {
      console.error(err);
      setRunning(false);
      setJobs((prev) => prev.filter(j => j.id !== tempId));
    }
  };

  const run = () => {
    if (!file || selected.size === 0 || running) return;
    const wsiName = file.name.replace(/\.[^/.]+$/, "");
    const sameWsiJob = jobs.find(j => j.wsi === wsiName && j.status !== 'failed' && j.status !== 'cancelled');
    if (sameWsiJob) {
      const ok = window.confirm(`"${wsiName}" 이미지로 이미 실행한 작업이 있어요.\n새로 정규화를 실행할까요?`);
      if (!ok) return;
    }
    runAsNew();
  };

  useEffect(() => {
    fetchJobs()
      .then(groups => {
        const serverJobs: UiJob[] = groups.map(g => {
          const anyActive = g.jobs.some(j => j.status === 'running' || j.status === 'pending');
          const allFailed = g.jobs.length > 0 && g.jobs.every(j => j.status === 'failed');
          const allSettled = g.jobs.every(j => ['done', 'failed', 'cancelled'].includes(j.status));
          const status = allFailed ? 'failed' : anyActive ? 'running' : allSettled ? 'done' : 'pending';
          const results: Record<number, JobResult> = {};
          const failedJobInfo: Record<number, FailedJobInfo> = {};
          for (const j of g.jobs) {
            if (j.status === 'done' && j.result_image_id && j.metrics) {
              results[j.model_id] = { metrics: j.metrics, result_image_id: j.result_image_id, elapsed_seconds: j.elapsed_seconds ?? undefined };
            } else if (j.status === 'failed') {
              failedJobInfo[j.model_id] = { message: j.message ?? "오류가 발생했습니다.\n문제가 지속되면 담당자에게 문의해주세요.", error_detail: j.error_detail };
            }
          }
          return {
            id: g.group_id,
            wsi: g.wsi_name,
            modelIds: g.jobs.map(j => j.model_id),
            status,
            when: formatRelativeTime(g.created_at),
            progress: g.jobs.length > 0
              ? g.jobs.reduce((sum, j) => sum + (j.status === 'done' || j.status === 'failed' ? 100 : j.progress), 0) / g.jobs.length
              : undefined,
            src_image_id: g.image_id,
            results: Object.keys(results).length > 0 ? results : undefined,
            failedJobInfo: Object.keys(failedJobInfo).length > 0 ? failedJobInfo : undefined,
          };
        });
        setJobs([...serverJobs, ...MOCK_JOBS]);
      })
      .catch(() => {});
    return () => {
      if (pollingRef.current) clearInterval(pollingRef.current);
    };
  }, []);

  const activeJob = activeJobId ? jobs.find((j) => j.id === activeJobId) : null;
  const viewingJob = !!activeJob;
  const activeModels = activeJob
    ? activeJob.modelIds
        .map((id) => MODELS.find((m) => m.id === id)!)
        .filter(Boolean)
    : [];
  const headerTitle = viewingJob
    ? activeJob!.wsi
    : running
      ? "분석 실행 중"
      : "Stain Normalization 비교 플랫폼";

  return (
    <div className="app">
      <Sidebar
        jobs={jobs}
        activeJobId={activeJobId}
        onSelectJob={(id) => {
          const job = jobs.find((j) => j.id === id);
          if (job?.status === "done" || job?.status === "failed") navigate(`/jobs/${id}`);
        }}
        onJobTerminate={handleJobTerminate}
      />
      <div className="main">
        <Topbar
          file={file}
          selectedCount={selected.size}
          onRun={run}
          running={running}
          onReset={reset}
          title={headerTitle}
          viewingJob={viewingJob}
          onBack={reset}
        />
        <div
          className="content"
          style={{
            overflow: "hidden",
            display: "flex",
            flexDirection: "column",
          }}
        >
          {viewingJob && (activeJob?.results || activeJob?.failedJobInfo) ? (
            <div style={{ flex: 1, overflow: "auto" }}>
              {activeModels.length === 1 && !activeJob.failedJobInfo?.[activeModels[0].id] ? (
                <SingleResult
                  model={activeModels[0]}
                  result={activeJob.results![activeModels[0].id]}
                  srcImageId={activeJob.src_image_id}
                />
              ) : (
                <MultiDashboard
                  key={activeJob.id}
                  models={activeModels}
                  results={activeJob.results ?? {}}
                  failedJobs={activeJob.failedJobInfo ?? {}}
                  srcImageId={activeJob.src_image_id}
                />
              )}
            </div>
          ) : (
            <ConfigColumn
              file={file}
              onPickFile={setFile}
              onClearFile={() => setFile(null)}
              selected={selected}
              onToggleModel={toggleModel}
              onRun={run}
              running={running}
              fileInputRef={fileInputRef}
            />
          )}
        </div>
      </div>
    </div>
  );
}
