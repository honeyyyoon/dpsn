import { useState } from "react";
import type { CSSProperties } from "react";
import { METRIC_DEFS } from "../data";
import type { MetricDef, ModelUi, JobResult } from "../types";
import Icon from "./Icon";
import { WsiView } from "./WsiImage";
import { getImageUrl, getTargetImageUrl } from '../api';

function metricColor(def: MetricDef, value: number): string {
  const passed = def.higherBetter ? value >= def.ref : value <= def.ref;
  return passed ? "var(--success)" : "#f97316";
}

function refLabel(def: MetricDef): string {
  const sign = def.higherBetter ? "≥" : "≤";
  return `${sign}${def.ref}${def.unit ? " " + def.unit : ""}`;
}

function formatElapsed(seconds?: number | null): string {
  if (seconds == null) return "-";
  const s = Math.round(seconds);
  if (s < 60) return `${s}초`;
  return `${Math.floor(s / 60)}분 ${s % 60}초`;
}

interface MetricCardProps {
  def: MetricDef;
  value: number;
}

function MetricCard({ def, value }: MetricCardProps) {
  const color = metricColor(def, value);
  return (
    <div className="card" style={{ padding: 14 }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 4 }}>
        <div style={{ fontSize: 11, fontWeight: 600, color: "var(--text-muted)", letterSpacing: "0.04em", textTransform: "uppercase" }}>
          {def.label}
        </div>
        <span style={{ fontSize: 10, color: "var(--text-dim)" }}>{refLabel(def)}</span>
      </div>
      <div style={{ display: "flex", alignItems: "baseline", gap: 6 }}>
        <div className="num" style={{ fontSize: 26, fontWeight: 600, letterSpacing: "-0.02em", color }}>
          {def.key === "ssim" ? value.toFixed(3) : value.toFixed(2)}
        </div>
        {def.unit && <div style={{ fontSize: 12, color: "var(--text-muted)" }}>{def.unit}</div>}
      </div>
      <div style={{ fontSize: 11, marginTop: 4, color: "var(--text-dim)" }}>{def.desc}</div>
    </div>
  );
}

export function EmptyState({ hasFile, selectedCount }: { hasFile: boolean; selectedCount: number }) {
  const reasons = [];
  if (!hasFile) reasons.push({ icon: "upload", label: "WSI 이미지 업로드" });
  if (selectedCount === 0) reasons.push({ icon: "layers", label: "정규화 방법 1개 이상 선택" });
  return (
    <div style={{ height: "100%", display: "flex", alignItems: "center", justifyContent: "center" }}>
      <div style={{ textAlign: "center", maxWidth: 420 }}>
        <div style={{ width: 64, height: 64, margin: "0 auto 18px", borderRadius: 16, background: "var(--bg-sunken)", display: "flex", alignItems: "center", justifyContent: "center", color: "var(--text-muted)" }}>
          <Icon name="layers" size={28} />
        </div>
        <div style={{ fontSize: 18, fontWeight: 600, letterSpacing: "-0.01em" }}>정규화 준비 완료</div>
        <div style={{ fontSize: 13, color: "var(--text-muted)", marginTop: 6 }}>
          왼쪽 패널에서 WSI와 모델을 선택하면 이곳에서 변환 전/후 비교와 메트릭을 확인할 수 있습니다.
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 8, marginTop: 22, maxWidth: 280, marginInline: "auto" }}>
          {reasons.map((r, i) => (
            <div key={i} style={{ display: "flex", alignItems: "center", gap: 10, padding: "10px 14px", borderRadius: "var(--r-md)", background: "var(--panel)", border: "1px solid var(--border)", fontSize: 13, color: "var(--text)", textAlign: "left" }}>
              <div style={{ width: 22, height: 22, borderRadius: 6, background: "var(--accent-50)", color: "var(--accent)", display: "flex", alignItems: "center", justifyContent: "center" }}>
                <Icon name={r.icon} size={13} />
              </div>
              {r.label}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// 왼쪽 패널: 원본 + 타겟 (세로 2개)
function LeftPanel({ srcImageId, seed }: { srcImageId?: string; seed: number }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      <div className="card fade-up" style={{ padding: 12 }}>
        <WsiView seed={seed} src={srcImageId ? getImageUrl(srcImageId, true) : undefined} mode="dim" chip="원본" />
      </div>
      <div className="card fade-up" style={{ padding: 12 }}>
        <WsiView seed={seed} src={getTargetImageUrl()} mode="dim" chip="타겟" />
      </div>
    </div>
  );
}

// 왼쪽↔오른쪽 구분선
function Divider() {
  return <div style={{ width: 1, background: "var(--border)", alignSelf: "stretch" }} />;
}

// 결과 이미지 카드 — 메트릭 오버레이 포함
function ResultCard({
  model,
  result,
  best,
  seed,
  onRatioDetected,
  onDownload,
  style,
}: {
  model: ModelUi;
  result?: JobResult;
  best: Record<string, number>;
  seed: number;
  onRatioDetected?: (r: number) => void;
  onDownload: (imageId: string, name: string) => void;
  style?: CSSProperties;
}) {
  return (
    <div className="card fade-up" style={{ padding: 12, ...style }}>
      <WsiView
        seed={seed}
        src={result?.result_image_id ? getImageUrl(result.result_image_id, true) : undefined}
        mode="norm"
        tint={model.tint}
        intensity={0.8}
        chip={model.name}
        chipColor={model.tint}
        onRatioDetected={onRatioDetected}
      >
        <div style={{ position: "absolute", bottom: 8, left: 8, right: 8, display: "flex", alignItems: "center", gap: 4 }}>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 4, flex: 1 }}>
            {METRIC_DEFS.map((def) => {
              const isBest = best[def.key] === model.id;
              const val = result?.metrics[def.key as keyof JobResult["metrics"]] ?? 0;
              return (
                <div key={def.key} style={{ padding: "5px 7px", borderRadius: "var(--r-sm)", background: "rgba(15,22,41,0.75)", backdropFilter: "blur(6px)" }}>
                  <div style={{ fontSize: 8, color: "rgba(255,255,255,0.55)", textTransform: "uppercase", letterSpacing: "0.04em", fontWeight: 600 }}>
                    {def.label}{isBest && <span style={{ marginLeft: 3 }}>★</span>}
                  </div>
                  <div className="num" style={{ fontSize: 12, fontWeight: 600, color: metricColor(def, val) }}>
                    {def.key === "ssim" ? val.toFixed(3) : val.toFixed(2)}
                  </div>
                </div>
              );
            })}
          </div>
          <button
            className="icon-btn"
            style={{ flexShrink: 0, background: "rgba(15,22,41,0.75)", backdropFilter: "blur(6px)", color: "#fff", borderRadius: 6 }}
            disabled={!result?.result_image_id}
            onClick={() => result?.result_image_id && onDownload(result.result_image_id, model.name)}
            title="결과 이미지 다운로드"
          >
            <Icon name="download" size={14} />
          </button>
        </div>
      </WsiView>
    </div>
  );
}

export function SingleResult({ model, result, srcImageId }: { model: ModelUi; result: JobResult; srcImageId?: string }) {
  const seed = 7;
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16, padding: 24 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
        <div style={{ fontSize: 16, fontWeight: 600, letterSpacing: "-0.01em" }}>결과 비교 대시보드</div>
        <span className="chip accent dot">{model.name}</span>
        {result.elapsed_seconds != null && (
          <span style={{ fontSize: 12, color: "var(--text-muted)" }}>처리 시간 {formatElapsed(result.elapsed_seconds)}</span>
        )}
      </div>

      {/* 타겟 | 원본 | 결과 — 1행 3열, 이미지 크기 W/3 */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 12 }}>
        <div className="card fade-up" style={{ padding: 12 }}>
          <WsiView seed={seed} src={getTargetImageUrl()} mode="dim" chip="타겟" />
        </div>
        <div className="card fade-up" style={{ padding: 12 }}>
          <WsiView seed={seed} src={srcImageId ? getImageUrl(srcImageId, true) : undefined} mode="dim" chip="원본" />
        </div>
        <div className="card fade-up" style={{ padding: 12 }}>
          <WsiView
            seed={seed}
            src={result.result_image_id ? getImageUrl(result.result_image_id, true) : undefined}
            mode="norm"
            tint={model.tint}
            intensity={0.8}
            chip={model.name}
            chipColor={model.tint}
          />
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 12 }}>
        {METRIC_DEFS.map((def) => (
          <MetricCard key={def.key} def={def} value={result.metrics[def.key as keyof typeof result.metrics] ?? 0} />
        ))}
      </div>
    </div>
  );
}

export function MultiDashboard({ models, results, srcImageId }: { models: ModelUi[]; results: Record<number, JobResult>; srcImageId?: string }) {
  const [sortKey, setSortKey] = useState<"psnr" | "ssim" | "fid">("psnr");
  const [hiddenModels, setHiddenModels] = useState<Set<number>>(new Set());
  const seed = 7;

  const toggleModel = (id: number) => {
    setHiddenModels((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const sorted = [...models].sort((a, b) => {
    const def = METRIC_DEFS.find((d) => d.key === sortKey);
    const missing = def?.higherBetter ? -Infinity : Infinity;
    const A = results[a.id]?.metrics[sortKey] ?? missing;
    const B = results[b.id]?.metrics[sortKey] ?? missing;
    return def?.higherBetter ? B - A : A - B;
  });

  const visibleSorted = sorted.filter(m => !hiddenModels.has(m.id));
  const visibleCount = visibleSorted.length;

  // 열 수 & 외부 그리드 비율 — "1fr 1px Nfr" 공식으로 양쪽 이미지 크기 = W/(N+1)
  // 2개: 우측 2열 그리드에 gridColumn:1 강제 → 상하 배치, 이미지 W/3
  // 3-4개: 2열, 5-6개: 3열
  const rightCols = visibleCount <= 4 ? 2 : 3;
  const outerCols = `1fr 1px ${rightCols}fr`;

  const best: Record<string, number> = {};
  METRIC_DEFS.forEach((def) => {
    const vals = models
      .filter((m) => results[m.id]?.metrics[def.key as keyof (typeof results)[number]["metrics"]] != null)
      .map((m) => ({ id: m.id, v: results[m.id].metrics[def.key as keyof (typeof results)[number]["metrics"]] }));
    vals.sort((x, y) => (def.higherBetter ? y.v - x.v : x.v - y.v));
    if (vals.length > 0) best[def.key] = vals[0].id;
  });

  const handleDownload = async (imageId: string, modelName: string) => {
    try {
      const res = await fetch(getImageUrl(imageId));
      if (!res.ok) throw new Error(`${res.status}`);
      const blob = await res.blob();
      const blobUrl = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = blobUrl;
      a.download = `${modelName}_normalized.png`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(blobUrl);
    } catch {
      alert('다운로드에 실패했습니다.');
    }
  };

  const thStyle: CSSProperties = { textAlign: "left", padding: "12px 20px", fontWeight: 600, fontSize: 12 };
  const tdStyle: CSSProperties = { padding: "14px 20px", verticalAlign: "middle" };

  const tableSorted = [
    ...sorted.filter(m => !hiddenModels.has(m.id)),
    ...sorted.filter(m => hiddenModels.has(m.id)),
  ];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16, padding: 24 }}>
      {/* 헤더: 모델 chip에 눈 토글 내장 */}
      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
        <div style={{ fontSize: 16, fontWeight: 600, letterSpacing: "-0.01em" }}>결과 비교 대시보드</div>
        <div style={{ display: "flex", gap: 6 }}>
          {sorted.map((m) => {
            const isHidden = hiddenModels.has(m.id);
            return (
              <div
                key={m.id}
                className="chip"
                style={{
                  background: `color-mix(in oklab, ${m.tint} 12%, var(--panel))`,
                  color: m.tint,
                  borderColor: `color-mix(in oklab, ${m.tint} 30%, transparent)`,
                  display: "flex", alignItems: "center", gap: 4,
                  opacity: isHidden ? 0.55 : 1,
                  transition: "opacity 150ms",
                }}
              >
                {m.name}
                <button
                  onClick={() => toggleModel(m.id)}
                  title={isHidden ? `${m.name} 표시` : `${m.name} 숨기기`}
                  style={{ display: "flex", alignItems: "center", justifyContent: "center", width: 14, height: 14, padding: 0, background: "none", border: "none", cursor: "pointer", color: "inherit" }}
                >
                  <Icon name={isHidden ? "eye-off" : "eye"} size={11} strokeWidth={2} />
                </button>
              </div>
            );
          })}
        </div>
      </div>

      {/* 왼쪽: 원본+타겟 | 구분선 | 오른쪽: 결과 그리드 */}
      <div style={{ display: "grid", gridTemplateColumns: outerCols, gap: "0 14px", alignItems: "start" }}>
        <LeftPanel srcImageId={srcImageId} seed={seed} />
        <Divider />
        <div style={{ display: "grid", gridTemplateColumns: `repeat(${rightCols}, 1fr)`, gap: 12, alignContent: "start" }}>
          {visibleSorted.map((m, i) => (
            <ResultCard
              key={m.id}
              model={m}
              result={results[m.id]}
              best={best}
              seed={seed}
              onRatioDetected={undefined}
              onDownload={handleDownload}
              style={visibleCount === 2 ? { gridColumn: 1 } : undefined}
            />
          ))}
        </div>
      </div>

      {/* 비교표 */}
      <div className="card" style={{ padding: 0, overflow: "hidden" }}>
        <div style={{ padding: "12px 20px", borderBottom: "1px solid var(--border)", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <div style={{ fontSize: 13, fontWeight: 600 }}>비교표</div>
          <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 11, color: "var(--text-muted)" }}>
            정렬
            {METRIC_DEFS.map((def) => (
              <button
                key={def.key}
                onClick={() => setSortKey(def.key)}
                className="btn sm"
                style={{ background: sortKey === def.key ? "var(--accent-50)" : "transparent", color: sortKey === def.key ? "var(--accent-600)" : "var(--text-muted)", height: 24, padding: "0 8px", fontWeight: 500 }}
              >
                {def.label}
              </button>
            ))}
          </div>
        </div>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
          <thead>
            <tr style={{ color: "var(--text-muted)", fontSize: 12, textTransform: "uppercase", letterSpacing: "0.04em" }}>
              <th style={thStyle}>모델</th>
              <th style={thStyle}>분류</th>
              {METRIC_DEFS.map((def) => (
                <th key={def.key} style={{ ...thStyle, textAlign: "right" }}>
                  {def.label} <span style={{ color: "var(--text-dim)", fontWeight: 400 }}>({refLabel(def)})</span>
                </th>
              ))}
              <th style={{ ...thStyle, textAlign: "right" }}>처리 시간</th>
            </tr>
          </thead>
          <tbody>
            {tableSorted.map((m) => {
              const r = results[m.id];
              const hidden = hiddenModels.has(m.id);
              if (hidden) {
                return (
                  <tr key={m.id} style={{ borderTop: "1px solid var(--divider)" }}>
                    <td style={tdStyle}>
                      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                        <div style={{ width: 8, height: 28, borderRadius: 2, background: m.tint, opacity: 0.4 }} />
                        <div style={{ fontWeight: 700, fontSize: 15, color: "var(--text-dim)" }}>{m.name}</div>
                        <button className="icon-btn" onClick={() => toggleModel(m.id)} title={`${m.name} 표시`} style={{ color: "var(--text-dim)" }}>
                          <Icon name="eye-off" size={14} />
                        </button>
                      </div>
                    </td>
                    <td style={tdStyle} />
                    {METRIC_DEFS.map((def) => <td key={def.key} style={{ ...tdStyle, textAlign: "right" }} />)}
                    <td style={{ ...tdStyle, textAlign: "right" }} />
                  </tr>
                );
              }
              return (
                <tr key={m.id} style={{ borderTop: "1px solid var(--divider)" }}>
                  <td style={tdStyle}>
                    <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                      <div style={{ width: 8, height: 28, borderRadius: 2, background: m.tint }} />
                      <div style={{ fontWeight: 700, fontSize: 15 }}>{m.name}</div>
                      <button className="icon-btn" onClick={() => toggleModel(m.id)} title={`${m.name} 숨기기`} style={{ color: "var(--text-muted)" }}>
                        <Icon name="eye" size={14} />
                      </button>
                    </div>
                  </td>
                  <td style={tdStyle}>
                    <span className="chip">{m.category === "Classical" ? "알고리즘 기반" : "딥러닝 모델"}</span>
                  </td>
                  {METRIC_DEFS.map((def) => {
                    const isBest = best[def.key] === m.id;
                    const val = r?.metrics[def.key as keyof JobResult["metrics"]] ?? 0;
                    return (
                      <td key={def.key} style={{ ...tdStyle, textAlign: "right" }} className="num">
                        <span style={{ fontWeight: isBest ? 700 : 500, color: metricColor(def, val) }}>
                          {def.key === "ssim" ? val.toFixed(3) : val.toFixed(2)}
                          {isBest && <span style={{ marginLeft: 4, color: "var(--text-muted)" }}>★</span>}
                        </span>
                      </td>
                    );
                  })}
                  <td style={{ ...tdStyle, textAlign: "right" }} className="num">
                    <span style={{ color: "var(--text-muted)", fontWeight: 500 }}>{formatElapsed(r?.elapsed_seconds)}</span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
