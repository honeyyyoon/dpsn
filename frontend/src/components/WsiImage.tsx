import type { CSSProperties, ReactNode } from 'react';

interface TissueSvgProps {
  seed?: number;
  tint?: string | null;
  intensity?: number;
  style?: CSSProperties;
  mode?: 'h-e' | 'norm' | 'dim' | 'pale' | 'blue';
}

export function TissueSvg({ seed = 1, tint = null, intensity = 1, style, mode = 'h-e' }: TissueSvgProps) {
  const rand = (i: number) => {
    const x = Math.sin(seed * 9301 + i * 49297) * 233280;
    return x - Math.floor(x);
  };

  const cells = [];
  const N = 90;
  for (let i = 0; i < N; i++) {
    cells.push({
      cx: rand(i) * 400,
      cy: rand(i + 1000) * 400,
      r: 6 + rand(i + 2000) * 10,
    });
  }

  const palettes = {
    'h-e':  { bg: '#e8c4d4', stroma: '#d89ab4', nuclei: '#5b2766' },
    'norm': { bg: '#f2d4dc', stroma: '#e8a8bb', nuclei: '#6b2d70' },
    'dim':  { bg: '#b894a5', stroma: '#9c6f89', nuclei: '#3d1a47' },
    'pale': { bg: '#f5e0e6', stroma: '#edc5d3', nuclei: '#8c4a94' },
    'blue': { bg: '#c9d4e4', stroma: '#9fb4ce', nuclei: '#1e3a8a' },
  };
  const pal = palettes[mode] || palettes['h-e'];

  return (
    <svg viewBox="0 0 400 400" preserveAspectRatio="xMidYMid slice"
         style={{ width: '100%', height: '100%', display: 'block', ...style }}>
      <defs>
        <radialGradient id={`bg-${seed}-${mode}`} cx="50%" cy="40%" r="70%">
          <stop offset="0%" stopColor={pal.bg} stopOpacity="1"/>
          <stop offset="100%" stopColor={pal.stroma} stopOpacity="1"/>
        </radialGradient>
        <filter id={`blur-${seed}`}>
          <feGaussianBlur stdDeviation="0.8"/>
        </filter>
      </defs>
      <rect width="400" height="400" fill={`url(#bg-${seed}-${mode})`}/>
      {Array.from({ length: 18 }).map((_, i) => (
        <line key={`f${i}`}
          x1={rand(i + 5000) * 400} y1={rand(i + 6000) * 400}
          x2={rand(i + 5000) * 400 + (rand(i + 7000) - 0.5) * 120}
          y2={rand(i + 6000) * 400 + (rand(i + 8000) - 0.5) * 120}
          stroke={pal.stroma} strokeOpacity={0.4} strokeWidth={2.5}/>
      ))}
      <g filter={`url(#blur-${seed})`}>
        {cells.map((c, i) => (
          <g key={i}>
            <circle cx={c.cx} cy={c.cy} r={c.r} fill={pal.stroma} opacity="0.75"/>
            <circle cx={c.cx + (rand(i + 9000) - 0.5) * 3} cy={c.cy + (rand(i + 9500) - 0.5) * 3}
                    r={c.r * 0.45} fill={pal.nuclei} opacity="0.85"/>
          </g>
        ))}
      </g>
      {tint && (
        <rect width="400" height="400" fill={tint} opacity={0.12 * intensity}
              style={{ mixBlendMode: 'color' }}/>
      )}
    </svg>
  );
}

interface WsiViewProps {
  label?: string;
  sublabel?: string;
  seed: number;
  src?: string;
  mode?: TissueSvgProps['mode'];
  tint?: string;
  intensity?: number;
  showGrid?: boolean;
  chip?: string;
  chipColor?: string;
  style?: CSSProperties;
  children?: ReactNode;
}

export function WsiView({ label, sublabel, seed, src, mode = 'h-e', tint, intensity = 1,
                          showGrid = false, chip, chipColor, style, children }: WsiViewProps) {
  return (
    <div style={{
      position: 'relative',
      background: '#0f1629',
      borderRadius: 'var(--r-md)',
      overflow: 'hidden',
      aspectRatio: '1 / 1',
      minHeight: 0,
      ...style,
    }}>
      {src
        ? <img src={src} alt="" style={{ width: '100%', height: '100%', objectFit: 'contain', display: 'block' }}/>
        : <TissueSvg seed={seed} mode={mode} tint={tint} intensity={intensity}/>
      }
      {showGrid && (
        <svg viewBox="0 0 400 400" preserveAspectRatio="none"
             style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none' }}>
          {Array.from({ length: 8 }).map((_, i) =>
            <line key={`v${i}`} x1={i * 50} y1={0} x2={i * 50} y2={400}
                  stroke="rgba(255,255,255,0.12)" strokeWidth="1"/>)}
          {Array.from({ length: 8 }).map((_, i) =>
            <line key={`h${i}`} x1={0} y1={i * 50} x2={400} y2={i * 50}
                  stroke="rgba(255,255,255,0.12)" strokeWidth="1"/>)}
        </svg>
      )}
      {chip && (
        <div style={{
          position: 'absolute', top: 10, left: 10,
          display: 'inline-flex', alignItems: 'center', gap: 6,
          padding: '4px 10px', borderRadius: 999,
          fontSize: 11, fontWeight: 600,
          color: '#fff',
          background: chipColor || 'rgba(15,22,41,0.72)',
          backdropFilter: 'blur(8px)',
          letterSpacing: '0.02em',
        }}>{chip}</div>
      )}
      {label && (
        <div style={{
          position: 'absolute', bottom: 10, left: 10, right: 10,
          color: '#fff', fontSize: 12,
          display: 'flex', alignItems: 'flex-end', justifyContent: 'space-between',
          pointerEvents: 'none',
        }}>
          <div>
            <div style={{ fontWeight: 600, letterSpacing: '-0.01em' }}>{label}</div>
            {sublabel && <div style={{ opacity: 0.7, fontSize: 11, marginTop: 2, fontFamily: 'var(--font-mono)' }}>{sublabel}</div>}
          </div>
        </div>
      )}
      {children}
    </div>
  );
}
