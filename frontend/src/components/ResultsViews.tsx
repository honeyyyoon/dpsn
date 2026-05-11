import { useState } from "react";
import type { CSSProperties } from "react";
import { METRIC_DEFS } from "../data";
import type { MetricDef, ModelUi, JobResult } from "../types";
import Icon from "./Icon";
import { WsiView } from "./WsiImage";
import { getImageUrl } from '../api';

interface MetricCardProps {
  def: MetricDef;
  value: number;
}

function MetricCard({ def, value }: MetricCardProps) {
  return (
    <div className="card" style={{ padding: 14 }}>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          marginBottom: 4,
        }}
      >
        <div
          style={{
            fontSize: 11,
            fontWeight: 600,
            color: "var(--text-muted)",
            letterSpacing: "0.04em",
            textTransform: "uppercase",
          }}
        >
          {def.label}
        </div>
        {def.higherBetter ? (
          <span style={{ fontSize: 10, color: "var(--text-dim)" }}>
            높을수록 ↑
          </span>
        ) : (
          <span style={{ fontSize: 10, color: "var(--text-dim)" }}>
            낮을수록 ↓
          </span>
        )}
      </div>
      <div style={{ display: "flex", alignItems: "baseline", gap: 6 }}>
        <div
          className="num"
          style={{ fontSize: 26, fontWeight: 600, letterSpacing: "-0.02em" }}
        >
          {def.key === "ssim" ? value.toFixed(3) : value.toFixed(2)}
        </div>
        {def.unit && (
          <div style={{ fontSize: 12, color: "var(--text-muted)" }}>
            {def.unit}
          </div>
        )}
      </div>
      <div style={{ fontSize: 11, marginTop: 4, color: "var(--text-dim)" }}>
        {def.desc}
      </div>
    </div>
  );
}

interface EmptyStateProps {
  hasFile: boolean;
  selectedCount: number;
}

export function EmptyState({ hasFile, selectedCount }: EmptyStateProps) {
  const reasons = [];
  if (!hasFile) reasons.push({ icon: "upload", label: "WSI 이미지 업로드" });
  if (selectedCount === 0)
    reasons.push({ icon: "layers", label: "정규화 방법 1개 이상 선택" });
  return (
    <div
      style={{
        height: "100%",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
      }}
    >
      <div style={{ textAlign: "center", maxWidth: 420 }}>
        <div
          style={{
            width: 64,
            height: 64,
            margin: "0 auto 18px",
            borderRadius: 16,
            background: "var(--bg-sunken)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            color: "var(--text-muted)",
          }}
        >
          <Icon name="layers" size={28} />
        </div>
        <div
          style={{ fontSize: 18, fontWeight: 600, letterSpacing: "-0.01em" }}
        >
          정규화 준비 완료
        </div>
        <div style={{ fontSize: 13, color: "var(--text-muted)", marginTop: 6 }}>
          왼쪽 패널에서 WSI와 모델을 선택하면 이곳에서 변환 전/후 비교와
          메트릭을 확인할 수 있습니다.
        </div>
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            gap: 8,
            marginTop: 22,
            maxWidth: 280,
            marginInline: "auto",
          }}
        >
          {reasons.map((r, i) => (
            <div
              key={i}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 10,
                padding: "10px 14px",
                borderRadius: "var(--r-md)",
                background: "var(--panel)",
                border: "1px solid var(--border)",
                fontSize: 13,
                color: "var(--text)",
                textAlign: "left",
              }}
            >
              <div
                style={{
                  width: 22,
                  height: 22,
                  borderRadius: 6,
                  background: "var(--accent-50)",
                  color: "var(--accent)",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                }}
              >
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

interface SingleResultProps {
  model: ModelUi;
  result: JobResult;
  srcImageId?: string;
}

export function SingleResult({ model, result, srcImageId }: SingleResultProps) {
  const [showGrid, setShowGrid] = useState(false);
  const [zoom, setZoom] = useState(1);
  const seed = 7;
  const inner = {
    width: "100%",
    height: "100%",
    transform: `scale(${zoom})`,
    transformOrigin: "center" as const,
    transition: "transform 200ms",
  };
  // const imageUrl = getImageUrl(result.result_image_id);

  return (
    <div
      style={{ display: "flex", flexDirection: "column", gap: 16, padding: 24 }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: 12,
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <div
            style={{ fontSize: 16, fontWeight: 600, letterSpacing: "-0.01em" }}
          >
            결과 비교 대시보드
          </div>
          <span className="chip accent dot">{model.name}</span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
          <button
            className="icon-btn"
            onClick={() => setZoom((z) => Math.max(1, z - 0.25))}
            disabled={zoom <= 1}
          >
            <Icon name="zoom-out" size={16} />
          </button>
          <div
            className="num"
            style={{
              fontSize: 12,
              color: "var(--text-muted)",
              width: 48,
              textAlign: "center",
            }}
          >
            {Math.round(zoom * 100)}%
          </div>
          <button
            className="icon-btn"
            onClick={() => setZoom((z) => Math.min(3, z + 0.25))}
          >
            <Icon name="zoom-in" size={16} />
          </button>
          <div
            style={{
              width: 1,
              height: 18,
              background: "var(--border)",
              margin: "0 6px",
            }}
          />
          <button
            className="icon-btn"
            onClick={() => setShowGrid((g) => !g)}
            style={
              showGrid
                ? { background: "var(--accent-50)", color: "var(--accent)" }
                : undefined
            }
          >
            <Icon name="grid" size={16} />
          </button>
        </div>
      </div>

      <div className="card fade-up" style={{ padding: 14 }}>
        <div
          style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}
        >
          <div
            style={{
              position: "relative",
              width: "100%",
              height: "100%",
              overflow: "hidden",
            }}
          >
            <div style={inner}>
              <WsiView
                seed={seed}
                src={srcImageId ? getImageUrl(srcImageId, true) : undefined}
                mode="dim"
                label="원본"
                chip="원본"
                showGrid={showGrid}
              />
            </div>
          </div>
          <div
            style={{
              position: "relative",
              width: "100%",
              height: "100%",
              overflow: "hidden",
            }}
          >
            <div style={inner}>
              <WsiView
                seed={seed}
                src={result.result_image_id ? getImageUrl(result.result_image_id, true) : undefined}
                mode="norm"
                tint={model.tint}
                intensity={0.8}
                label="정규화 결과"
                sublabel={model.name}
                chip={model.name}
                chipColor={model.tint}
                showGrid={showGrid}
              />
            </div>
          </div>
        </div>
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(3, 1fr)",
          gap: 12,
        }}
      >
        {METRIC_DEFS.map((def) => (
          <MetricCard
            key={def.key}
            def={def}
            value={result.metrics[def.key as keyof typeof result.metrics] ?? 0}
          />
        ))}
      </div>
    </div>
  );
}

interface MultiDashboardProps {
  models: ModelUi[];
  results: Record<number, JobResult>;
  srcImageId?: string;
}

export function MultiDashboard({ models, results, srcImageId }: MultiDashboardProps) {
  const [sortKey, setSortKey] = useState<"psnr" | "ssim" | "fid">("psnr");
  const seed = 7;

  const sorted = [...models].sort((a, b) => {
    const A = results[a.id]?.metrics[sortKey] ?? 0;
    const B = results[b.id]?.metrics[sortKey] ?? 0;
    const def = METRIC_DEFS.find((d) => d.key === sortKey);
    return def?.higherBetter ? B - A : A - B;
  });

  const best: Record<string, number> = {};
  METRIC_DEFS.forEach((def) => {
    const vals = models.map((m) => ({
      id: m.id,
      v:
        results[m.id]?.metrics[
          def.key as keyof (typeof results)[number]["metrics"]
        ] ?? 0,
    }));
    vals.sort((x, y) => (def.higherBetter ? y.v - x.v : x.v - y.v));
    best[def.key] = vals[0].id;
  });

  const thStyle: CSSProperties = {
    textAlign: "left",
    padding: "10px 16px",
    fontWeight: 600,
  };
  const tdStyle: CSSProperties = {
    padding: "14px 16px",
    verticalAlign: "middle",
  };

  return (
    <div
      style={{ display: "flex", flexDirection: "column", gap: 16, padding: 24 }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <div
            style={{ fontSize: 16, fontWeight: 600, letterSpacing: "-0.01em" }}
          >
            결과 비교 대시보드
          </div>
          <span className="chip accent dot">방법 {models.length}개</span>
        </div>
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(3, 1fr)",
          gap: 14,
        }}
      >
        <div className="card fade-up" style={{ padding: 12 }}>
          <WsiView
            seed={seed}
            src={srcImageId ? getImageUrl(srcImageId, true) : undefined}
            mode="dim"
            chip="원본"
            style={{ aspectRatio: "1 / 1" }}
          />
        </div>
        {sorted.map((m) => {
          const r = results[m.id];
          return (
            <div key={m.id} className="card fade-up" style={{ padding: 12 }}>
              <WsiView
                seed={seed}
                src={r?.result_image_id ? getImageUrl(r.result_image_id, true) : undefined}
                mode="norm"
                tint={m.tint}
                intensity={0.8}
                chip={m.name}
                chipColor={m.tint}
                style={{ aspectRatio: "1 / 1" }}
              />
              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "repeat(3, 1fr)",
                  gap: 6,
                  marginTop: 10,
                }}
              >
                {METRIC_DEFS.map((def) => {
                  const isBest = best[def.key] === m.id;
                  const val =
                    r?.metrics[def.key as keyof JobResult["metrics"]] ?? 0;
                  return (
                    <div
                      key={def.key}
                      style={{
                        padding: "6px 8px",
                        borderRadius: "var(--r-sm)",
                        background: isBest
                          ? "color-mix(in oklab, var(--success) 10%, var(--panel))"
                          : "var(--bg-sunken)",
                        border: isBest
                          ? "1px solid color-mix(in oklab, var(--success) 30%, transparent)"
                          : "1px solid transparent",
                      }}
                    >
                      <div
                        style={{
                          fontSize: 9,
                          color: "var(--text-muted)",
                          textTransform: "uppercase",
                          letterSpacing: "0.04em",
                          fontWeight: 600,
                        }}
                      >
                        {def.label}
                        {isBest && (
                          <span
                            style={{ color: "var(--success)", marginLeft: 4 }}
                          >
                            ★
                          </span>
                        )}
                      </div>
                      <div
                        className="num"
                        style={{ fontSize: 13, fontWeight: 600 }}
                      >
                        {def.key === "ssim" ? val.toFixed(3) : val.toFixed(2)}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          );
        })}
      </div>

      <div className="card" style={{ padding: 0, overflow: "hidden" }}>
        <div
          style={{
            padding: "12px 16px",
            borderBottom: "1px solid var(--border)",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
          }}
        >
          <div style={{ fontSize: 13, fontWeight: 600 }}>비교표</div>
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: 6,
              fontSize: 11,
              color: "var(--text-muted)",
            }}
          >
            정렬
            {METRIC_DEFS.map((def) => (
              <button
                key={def.key}
                onClick={() => setSortKey(def.key)}
                className="btn sm"
                style={{
                  background:
                    sortKey === def.key ? "var(--accent-50)" : "transparent",
                  color:
                    sortKey === def.key
                      ? "var(--accent-600)"
                      : "var(--text-muted)",
                  height: 24,
                  padding: "0 8px",
                  fontWeight: 500,
                }}
              >
                {def.label}
              </button>
            ))}
          </div>
        </div>
        <table
          style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}
        >
          <thead>
            <tr
              style={{
                color: "var(--text-muted)",
                fontSize: 11,
                textTransform: "uppercase",
                letterSpacing: "0.04em",
              }}
            >
              <th style={thStyle}>모델</th>
              <th style={thStyle}>분류</th>
              {METRIC_DEFS.map((def) => (
                <th key={def.key} style={{ ...thStyle, textAlign: "right" }}>
                  {def.label}{" "}
                  <span style={{ color: "var(--text-dim)" }}>
                    ({def.unit || (def.higherBetter ? "↑" : "↓")})
                  </span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {sorted.map((m) => {
              const r = results[m.id];
              return (
                <tr
                  key={m.id}
                  style={{ borderTop: "1px solid var(--divider)" }}
                >
                  <td style={tdStyle}>
                    <div
                      style={{ display: "flex", alignItems: "center", gap: 10 }}
                    >
                      <div
                        style={{
                          width: 8,
                          height: 28,
                          borderRadius: 2,
                          background: m.tint,
                        }}
                      />
                      <div style={{ fontWeight: 600 }}>{m.name}</div>
                    </div>
                  </td>
                  <td style={tdStyle}>
                    <span className="chip">
                      {m.category === "Classical" ? "알고리즘 기반" : "딥러닝 모델"}
                    </span>
                  </td>
                  {METRIC_DEFS.map((def) => {
                    const isBest = best[def.key] === m.id;
                    const val =
                      r?.metrics[def.key as keyof JobResult["metrics"]] ?? 0;
                    return (
                      <td
                        key={def.key}
                        style={{ ...tdStyle, textAlign: "right" }}
                        className="num"
                      >
                        <span
                          style={{
                            fontWeight: isBest ? 600 : 500,
                            color: isBest ? "var(--success)" : "var(--text)",
                          }}
                        >
                          {def.key === "ssim" ? val.toFixed(3) : val.toFixed(2)}
                          {isBest && <span style={{ marginLeft: 4 }}>★</span>}
                        </span>
                      </td>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
