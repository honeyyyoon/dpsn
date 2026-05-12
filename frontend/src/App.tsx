import { useState, useEffect, useRef } from "react";
import type { ReactNode } from "react";
import "./styles.css";
import { MODELS } from "./data";
import type { UiJob, JobResult } from "./types";
import { createJobs, getJobStatus, getJobResult } from "./api";
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
    id: "mock-0",
    wsi: "CAMELYON17-042",
    modelIds: [1, 2],
    status: "running",
    when: "now",
    progress: 0.6,
  },
  {
    id: "mock-x",
    wsi: "PAIP-liver-089",
    modelIds: [4],
    status: "pending",
    when: "대기중",
  },
  {
    id: "mock-1",
    wsi: "TCGA-BRCA-A2K4",
    modelIds: [4, 5, 6],
    status: "done",
    when: "2m",
    results: {
      4: {
        metrics: { ssim: 0.927, psnr: 30.12, fid: 9.4 },
        result_image_id: "",
      },
      5: {
        metrics: { ssim: 0.934, psnr: 30.87, fid: 8.9 },
        result_image_id: "",
      },
      6: {
        metrics: { ssim: 0.941, psnr: 31.44, fid: 8.2 },
        result_image_id: "",
      },
    },
  },
  {
    id: "mock-2",
    wsi: "GTEx-stomach-5",
    modelIds: [3],
    status: "done",
    when: "3h",
    results: {
      3: {
        metrics: { ssim: 0.905, psnr: 29.03, fid: 11.9 },
        result_image_id: "",
      },
    },
  },
  {
    id: "mock-3",
    wsi: "TCGA-LUAD-B41C",
    modelIds: [1, 3, 4, 6],
    status: "done",
    when: "1d",
    results: {
      1: {
        metrics: { ssim: 0.891, psnr: 28.42, fid: 12.6 },
        result_image_id: "",
      },
      3: {
        metrics: { ssim: 0.905, psnr: 29.03, fid: 11.9 },
        result_image_id: "",
      },
      4: {
        metrics: { ssim: 0.927, psnr: 30.12, fid: 9.4 },
        result_image_id: "",
      },
      6: {
        metrics: { ssim: 0.941, psnr: 31.44, fid: 8.2 },
        result_image_id: "",
      },
    },
  },
];

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [selected, setSelected] = useState<Set<number>>(new Set());
  const [running, setRunning] = useState(false);
  const [jobs, setJobs] = useState<UiJob[]>(MOCK_JOBS);
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
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
    setActiveJobId(null);
  };

  const run = async () => {
    if (!file || selected.size === 0) return;
    setRunning(true);

    const wsiName = file.name.replace(/\.[^/.]+$/, "");
    const modelIds = [...selected];

    try {
      const responses = await createJobs(file, modelIds);
      const jobIds = responses.map((r) => r.job_id);

      const uiJobId = jobIds[0];
      const newJob: UiJob = {
        id: uiJobId,
        wsi: wsiName,
        modelIds,
        status: "running",
        when: "now",
        progress: 0,
        src_image_id: responses[0].image_id,
      };
      setJobs((prev) => [newJob, ...prev]);

      const results: Record<number, JobResult> = {};
      const finishedSet = new Set<number>();
      const failedSet = new Set<number>();

      pollingRef.current = setInterval(async () => {
        for (let i = 0; i < jobIds.length; i++) {
          const jobId = jobIds[i];
          const modelId = modelIds[i];
          if (finishedSet.has(modelId)) continue;
          try {
            const status = await getJobStatus(jobId);
            if (status.status === "done") {
              const result = await getJobResult(jobId);
              results[modelId] = {
                metrics: result.metrics,
                result_image_id: result.result_image_id,
              };
              finishedSet.add(modelId);
            } else if (status.status === "failed") {
              failedSet.add(modelId);
              finishedSet.add(modelId);
            }
          } catch (err) {
            console.warn("Polling error, will retry:", err);
          }
        }

        setJobs((prev) =>
          prev.map((j) =>
            j.id === uiJobId
              ? { ...j, progress: finishedSet.size / jobIds.length }
              : j,
          ),
        );

        if (finishedSet.size >= jobIds.length) {
          if (pollingRef.current) clearInterval(pollingRef.current);
          setRunning(false);
          const allFailed = failedSet.size === jobIds.length;
          setJobs((prev) =>
            prev.map((j) =>
              j.id === uiJobId
                ? {
                    ...j,
                    status: allFailed ? "failed" : "done",
                    results,
                    when: "방금",
                  }
                : j,
            ),
          );
          if (!allFailed) setActiveJobId(uiJobId);
        }
      }, 1500);
    } catch (err) {
      console.error(err);
      setRunning(false);
    }
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
      : "새 작업";

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
            />
          )}
        </div>
      </div>
    </div>
  );
}
