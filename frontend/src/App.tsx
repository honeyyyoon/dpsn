import { useState, useEffect, useRef } from "react";
import type { ReactNode } from "react";
import "./styles.css";
import { MODELS } from "./data";
import type { UiJob, JobResult } from "./types";
import { createJobs, getJobStatus, getJobResult, deleteJob } from "./api";
import Sidebar from "./components/Sidebar";
import Icon from "./components/Icon";
import { UploadCard, ModelPicker } from "./components/ConfigPanel";
import { SingleResult, MultiDashboard } from "./components/ResultsViews";
import SameImageModal from "./components/SameImageModal";

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
            fontSize: 14,
            fontWeight: 600,
            letterSpacing: "-0.01em",
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
            <button className="btn ghost sm" onClick={onReset}>
              <Icon name="history" size={14} /> 초기화
            </button>
            <button
              className="btn primary"
              disabled={!canRun || running}
              onClick={onRun}
            >
              <Icon name="play" size={13} />
              {running
                ? "실행 중…"
                : selectedCount > 1
                  ? `정규화 ${selectedCount}개 실행`
                  : "실행"}
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
        fontSize: 12,
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
  matchingJob,
}: {
  file: File | null;
  onPickFile: (f: File) => void;
  onClearFile: () => void;
  selected: Set<number>;
  onToggleModel: (id: number) => void;
  onRun: () => void;
  running: boolean;
  fileInputRef: { current: HTMLInputElement | null };
  matchingJob?: UiJob | null;
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
                  fontSize: 11,
                  fontWeight: 600,
                  color: "var(--text-muted)",
                  textTransform: "uppercase",
                  letterSpacing: "0.04em",
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
              {matchingJob && (
                <div style={{ fontSize: 12, color: 'var(--accent-600)', marginTop: 8 }}>
                  ⚡ "{matchingJob.wsi}"로 실행한 이전 작업이 있어요.
                </div>
              )}
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
                    fontSize: 11,
                    fontWeight: 600,
                    color: "var(--text-muted)",
                    textTransform: "uppercase",
                    letterSpacing: "0.04em",
                  }}
                >
                  2. 정규화 방법 선택
                </div>
                <span
                  className="chip accent"
                  style={{
                    visibility: selected.size > 0 ? "visible" : "hidden",
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
              flexDirection: "column",
              gap: 4,
              flex: 1,
              minWidth: 0,
            }}
          >
            {file ? (
              <StatusLine ok>{file.name}</StatusLine>
            ) : (
              <StatusLine>WSI 이미지가 없습니다</StatusLine>
            )}
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
    progress: 0.6,
  },
  {
    id: "mock-pending",
    wsi: "PAIP-liver-089",
    modelIds: [5],
    status: "pending",
    when: "대기중",
  },
  // 2 models
  {
    id: "mock-2m",
    wsi: "GTEx-stomach-5",
    modelIds: [1, 2],
    status: "done",
    when: "1h",
    results: {
      1: { metrics: { ssim: 0.905, psnr: 29.03, fid: 11.9 }, result_image_id: "" },
      2: { metrics: { ssim: 0.921, psnr: 29.84, fid: 10.2 }, result_image_id: "" },
    },
  },
  // 3 models
  {
    id: "mock-3m",
    wsi: "TCGA-BRCA-A2K4",
    modelIds: [1, 2, 5],
    status: "done",
    when: "2h",
    results: {
      1: { metrics: { ssim: 0.891, psnr: 28.42, fid: 12.6 }, result_image_id: "" },
      2: { metrics: { ssim: 0.921, psnr: 29.84, fid: 10.2 }, result_image_id: "" },
      5: { metrics: { ssim: 0.934, psnr: 30.87, fid: 8.9 }, result_image_id: "" },
    },
  },
  // 4 models
  {
    id: "mock-4m",
    wsi: "TCGA-LUAD-B41C",
    modelIds: [1, 2, 3, 5],
    status: "done",
    when: "3h",
    results: {
      1: { metrics: { ssim: 0.876, psnr: 27.91, fid: 14.3 }, result_image_id: "" },
      2: { metrics: { ssim: 0.898, psnr: 28.76, fid: 12.1 }, result_image_id: "" },
      3: { metrics: { ssim: 0.912, psnr: 29.45, fid: 10.8 }, result_image_id: "" },
      5: { metrics: { ssim: 0.941, psnr: 31.44, fid: 8.2 }, result_image_id: "" },
    },
  },
  // 5 models
  {
    id: "mock-5m",
    wsi: "CPTAC-CCRCC-C3L-00004",
    modelIds: [1, 2, 3, 4, 5],
    status: "done",
    when: "1d",
    results: {
      1: { metrics: { ssim: 0.871, psnr: 27.52, fid: 15.1 }, result_image_id: "" },
      2: { metrics: { ssim: 0.893, psnr: 28.34, fid: 12.8 }, result_image_id: "" },
      3: { metrics: { ssim: 0.907, psnr: 29.12, fid: 11.2 }, result_image_id: "" },
      4: { metrics: { ssim: 0.928, psnr: 30.21, fid: 9.4 }, result_image_id: "" },
      5: { metrics: { ssim: 0.945, psnr: 31.78, fid: 7.9 }, result_image_id: "" },
    },
  },
  // 6 models
  {
    id: "mock-6m",
    wsi: "TCGA-OV-A5KX",
    modelIds: [1, 2, 3, 4, 5, 6],
    status: "done",
    when: "2d",
    results: {
      1: { metrics: { ssim: 0.868, psnr: 27.31, fid: 15.8 }, result_image_id: "" },
      2: { metrics: { ssim: 0.889, psnr: 28.15, fid: 13.2 }, result_image_id: "" },
      3: { metrics: { ssim: 0.904, psnr: 28.97, fid: 11.6 }, result_image_id: "" },
      4: { metrics: { ssim: 0.923, psnr: 30.04, fid: 9.7 }, result_image_id: "" },
      5: { metrics: { ssim: 0.938, psnr: 31.22, fid: 8.4 }, result_image_id: "" },
      6: { metrics: { ssim: 0.952, psnr: 32.41, fid: 7.1 }, result_image_id: "" },
    },
  },
  {
    id: "mock-fail",
    wsi: "NLST-lung-00891",
    modelIds: [2, 5],
    status: "failed",
    when: "2d",
  },
  {
    id: "mock-cancel",
    wsi: "PAIP-colon-112",
    modelIds: [1, 3],
    status: "cancelled",
    when: "3d",
  },
];

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [selected, setSelected] = useState<Set<number>>(new Set());
  const [running, setRunning] = useState(false);
  const [jobs, setJobs] = useState<UiJob[]>(MOCK_JOBS);
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [showSameImageModal, setShowSameImageModal] = useState(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const pollingRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const fingerprint = file ? `${file.name}:${file.size}` : null;
  const matchingJob = fingerprint ? (jobs.find(j => j.imageFingerprint === fingerprint) ?? null) : null;

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
    setActiveJobId(null);
  };

  const handleJobTerminate = async (jobId: string) => {
    const job = jobs.find(j => j.id === jobId);
    if (!job) return;
    const isRunning = job.status === 'running' || job.status === 'pending';
    try { await deleteJob(jobId); } catch { /* 서버에 없어도 UI 반영 */ }
    if (isRunning) {
      setJobs(prev => prev.map(j => j.id === jobId ? { ...j, status: 'cancelled' } : j));
    } else {
      setJobs(prev => prev.filter(j => j.id !== jobId));
      if (activeJobId === jobId) setActiveJobId(null);
    }
  };

  const startPolling = (uiJobId: string, jobIds: string[], modelIds: number[]) => {
    const newResults: Record<number, JobResult> = {};
    const finishedSet = new Set<number>();
    const failedSet = new Set<number>();

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
            newResults[modelId] = { metrics: result.metrics, result_image_id: result.result_image_id };
            finishedSet.add(modelId);
            progress += 100;
          } else if (status.status === "failed") {
            failedSet.add(modelId);
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
        prev.map((j) => j.id === uiJobId ? { ...j, progress: progress / 100 / jobIds.length } : j),
      );

      if (finishedSet.size >= jobIds.length) {
        if (pollingRef.current) clearInterval(pollingRef.current);
        setRunning(false);
        const allFailed = failedSet.size === jobIds.length;
        setJobs((prev) =>
          prev.map((j) =>
            j.id === uiJobId
              ? { ...j, status: allFailed ? "failed" : "done", results: { ...(j.results ?? {}), ...newResults }, when: "방금" }
              : j,
          ),
        );
        if (!allFailed) setActiveJobId(uiJobId);
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
      imageFingerprint: fingerprint ?? undefined,
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

  const runWithExtraModels = async (baseJob: UiJob) => {
    setShowSameImageModal(false);
    if (!file) return;
    const existingIds = new Set(baseJob.modelIds);
    const newModelIds = [...selected].filter(id => !existingIds.has(id));
    if (newModelIds.length === 0) {
      alert("이미 모두 실행된 모델입니다.");
      return;
    }
    setRunning(true);
    setJobs((prev) => {
      const rest = prev.filter(j => j.id !== baseJob.id);
      return [{ ...baseJob, status: "running" as const, modelIds: [...baseJob.modelIds, ...newModelIds] }, ...rest];
    });
    setActiveJobId(baseJob.id);
    try {
      const responses = await createJobs(file, newModelIds);
      const jobIds = responses.map((r) => r.job_id);
      startPolling(baseJob.id, jobIds, newModelIds);
    } catch (err) {
      console.error(err);
      setRunning(false);
      setJobs((prev) => prev.map(j => j.id === baseJob.id ? { ...j, status: "failed" as const } : j));
    }
  };

  const run = () => {
    if (!file || selected.size === 0 || running) return;
    if (matchingJob) {
      setShowSameImageModal(true);
      return;
    }
    runAsNew();
  };

  useEffect(() => {
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
          if (job?.status === "done") setActiveJobId(id);
        }}
        onNewRun={reset}
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
          onBack={() => setActiveJobId(null)}
        />
        <div
          className="content"
          style={{
            overflow: "hidden",
            display: "flex",
            flexDirection: "column",
          }}
        >
          {viewingJob && activeJob?.results ? (
            <div style={{ flex: 1, overflow: "auto" }}>
              {activeModels.length === 1 ? (
                <SingleResult
                  model={activeModels[0]}
                  result={activeJob.results[activeModels[0].id]}
                  srcImageId={activeJob.src_image_id}
                />
              ) : (
                <MultiDashboard
                  models={activeModels}
                  results={activeJob.results}
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
              matchingJob={matchingJob}
            />
          )}
        </div>
      </div>
      {showSameImageModal && matchingJob && (
        <SameImageModal
          fileName={file!.name}
          matchingJob={matchingJob}
          onDifferent={() => { setShowSameImageModal(false); runAsNew(); }}
          onAddModels={() => runWithExtraModels(matchingJob)}
          onClose={() => setShowSameImageModal(false)}
        />
      )}
    </div>
  );
}
