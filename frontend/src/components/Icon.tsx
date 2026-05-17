import type { CSSProperties } from 'react';

interface IconProps {
  name: string;
  size?: number;
  color?: string;
  strokeWidth?: number;
  style?: CSSProperties;
}

export default function Icon({ name, size = 18, color = 'currentColor', strokeWidth = 1.6, style }: IconProps) {
  const common = {
    width: size, height: size, viewBox: '0 0 24 24',
    fill: 'none' as const, stroke: color, strokeWidth,
    strokeLinecap: 'round' as const, strokeLinejoin: 'round' as const, style,
  };
  switch (name) {
    case 'upload':
      return <svg {...common}><path d="M12 3v12M6 9l6-6 6 6M4 17v2a2 2 0 002 2h12a2 2 0 002-2v-2"/></svg>;
    case 'image':
      return <svg {...common}><rect x="3" y="4" width="18" height="16" rx="2"/><circle cx="9" cy="10" r="2"/><path d="M21 16l-5-5-9 9"/></svg>;
    case 'layers':
      return <svg {...common}><path d="M12 3l9 5-9 5-9-5 9-5z"/><path d="M3 13l9 5 9-5"/><path d="M3 18l9 5 9-5"/></svg>;
    case 'play':
      return <svg {...common} fill={color} stroke="none"><path d="M7 4v16l14-8z"/></svg>;
    case 'grid':
      return <svg {...common}><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/></svg>;
    case 'history':
      return <svg {...common}><path d="M3 12a9 9 0 109-9 9.75 9.75 0 00-6.74 2.74L3 8"/><path d="M3 3v5h5"/><path d="M12 7v5l3 2"/></svg>;
    case 'chevron-right':
      return <svg {...common}><path d="M9 6l6 6-6 6"/></svg>;
    case 'chevron-left':
      return <svg {...common}><path d="M15 6l-6 6 6 6"/></svg>;
    case 'chevron-down':
      return <svg {...common}><path d="M6 9l6 6 6-6"/></svg>;
    case 'check':
      return <svg {...common}><path d="M5 12l5 5 9-11"/></svg>;
    case 'plus':
      return <svg {...common}><path d="M12 5v14M5 12h14"/></svg>;
    case 'x':
      return <svg {...common}><path d="M6 6l12 12M18 6L6 18"/></svg>;
    case 'download':
      return <svg {...common}><path d="M12 3v12M6 11l6 6 6-6M4 21h16"/></svg>;
    case 'zoom-in':
      return <svg {...common}><circle cx="11" cy="11" r="7"/><path d="M21 21l-4.3-4.3M8 11h6M11 8v6"/></svg>;
    case 'zoom-out':
      return <svg {...common}><circle cx="11" cy="11" r="7"/><path d="M21 21l-4.3-4.3M8 11h6"/></svg>;
    case 'maximize':
      return <svg {...common}><path d="M3 9V4h5M21 9V4h-5M3 15v5h5M21 15v5h-5"/></svg>;
    case 'cpu':
      return <svg {...common}><rect x="5" y="5" width="14" height="14" rx="2"/><rect x="9" y="9" width="6" height="6"/><path d="M9 1v3M15 1v3M9 20v3M15 20v3M1 9h3M1 15h3M20 9h3M20 15h3"/></svg>;
    case 'sparkle':
      return <svg {...common}><path d="M12 3l2 6 6 2-6 2-2 6-2-6-6-2 6-2z"/></svg>;
    case 'eye':
      return <svg {...common}><path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7-10-7-10-7z"/><circle cx="12" cy="12" r="3"/></svg>;
    case 'trash':
      return <svg {...common}><path d="M3 6h18M8 6V4h8v2M19 6l-1 14H6L5 6"/><path d="M10 11v6M14 11v6"/></svg>;
    case 'stop':
      return <svg {...common} fill={color} stroke="none"><rect x="5" y="5" width="14" height="14" rx="2"/></svg>;
    case 'dot-menu':
      return <svg {...common}><circle cx="12" cy="5" r="1" fill={color}/><circle cx="12" cy="12" r="1" fill={color}/><circle cx="12" cy="19" r="1" fill={color}/></svg>;
    case 'swap':
      return <svg {...common}><path d="M7 16H3l4-4M17 8h4l-4 4M21 16h-4l4-4M3 8h4L3 12"/></svg>;
    default:
      return <svg {...common}><circle cx="12" cy="12" r="9"/></svg>;
  }
}
