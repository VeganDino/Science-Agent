// Minimal inline SVG icons. Stroke-based, 1.5px width, match ChatGPT's style.
import type { SVGProps } from "react";

type Props = SVGProps<SVGSVGElement>;

const base = (props: Props) => ({
  width: 20,
  height: 20,
  viewBox: "0 0 24 24",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 1.8,
  strokeLinecap: "round" as const,
  strokeLinejoin: "round" as const,
  ...props,
});

export const IconNewChat = (p: Props) => (
  <svg {...base(p)}>
    <path d="M15.67 4.33a2.12 2.12 0 0 1 3 3L8 18H5v-3Z" />
    <path d="M13.5 6.5l3 3" />
  </svg>
);

export const IconSearch = (p: Props) => (
  <svg {...base(p)}>
    <circle cx="11" cy="11" r="7" />
    <path d="m20 20-3.5-3.5" />
  </svg>
);

export const IconCodex = (p: Props) => (
  <svg {...base(p)}>
    <path d="M12 2 3 7v10l9 5 9-5V7z" />
    <path d="M3 7l9 5 9-5" />
    <path d="M12 22V12" />
  </svg>
);

export const IconDots = (p: Props) => (
  <svg {...base(p)}>
    <circle cx="5" cy="12" r="1.2" fill="currentColor" />
    <circle cx="12" cy="12" r="1.2" fill="currentColor" />
    <circle cx="19" cy="12" r="1.2" fill="currentColor" />
  </svg>
);

export const IconChat = (p: Props) => (
  <svg {...base(p)}>
    <path d="M21 12a8 8 0 0 1-11.8 7L4 20l1-5A8 8 0 1 1 21 12z" />
  </svg>
);

export const IconCube = (p: Props) => (
  <svg {...base(p)}>
    <path d="M12 2 3 7v10l9 5 9-5V7z" />
    <path d="M3 7l9 5 9-5" />
  </svg>
);

export const IconFolder = (p: Props) => (
  <svg {...base(p)}>
    <path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" />
  </svg>
);

export const IconFolderPlus = (p: Props) => (
  <svg {...base(p)}>
    <path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" />
    <path d="M12 11v6" />
    <path d="M9 14h6" />
  </svg>
);

export const IconChevronDown = (p: Props) => (
  <svg {...base(p)}>
    <path d="m6 9 6 6 6-6" />
  </svg>
);

export const IconSidebar = (p: Props) => (
  <svg {...base(p)}>
    <rect x="3" y="4" width="18" height="16" rx="2" />
    <path d="M9 4v16" />
  </svg>
);

export const IconShare = (p: Props) => (
  <svg {...base(p)}>
    <circle cx="18" cy="5" r="2.5" />
    <circle cx="6" cy="12" r="2.5" />
    <circle cx="18" cy="19" r="2.5" />
    <path d="m8.2 10.8 7.6-4.6" />
    <path d="m8.2 13.2 7.6 4.6" />
  </svg>
);

export const IconSettings2 = (p: Props) => (
  <svg {...base(p)}>
    <circle cx="12" cy="12" r="3" />
    <path d="M19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.8-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.8 1.7 1.7 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.7 1.7 0 0 0 1.5-1 1.7 1.7 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.8.3h0a1.7 1.7 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.8v0a1.7 1.7 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1z" />
  </svg>
);

export const IconPlus = (p: Props) => (
  <svg {...base(p)}>
    <path d="M12 5v14" />
    <path d="M5 12h14" />
  </svg>
);

export const IconPaperPlane = (p: Props) => (
  <svg {...base(p)}>
    <path d="M22 2 11 13" />
    <path d="M22 2 15 22l-4-9-9-4Z" />
  </svg>
);

export const IconX = (p: Props) => (
  <svg {...base(p)}>
    <path d="M18 6 6 18" />
    <path d="m6 6 12 12" />
  </svg>
);

export const IconTrash = (p: Props) => (
  <svg {...base(p)}>
    <path d="M3 6h18" />
    <path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
    <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6" />
    <path d="M10 11v6" />
    <path d="M14 11v6" />
  </svg>
);

export const IconFile = (p: Props) => (
  <svg {...base(p)}>
    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
    <path d="M14 2v6h6" />
  </svg>
);

export const IconFilePDF = (p: Props) => (
  <svg {...base(p)}>
    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
    <path d="M14 2v6h6" />
    <text
      x="12"
      y="18"
      fontSize="5.2"
      fontWeight="700"
      textAnchor="middle"
      fill="currentColor"
      stroke="none"
    >
      PDF
    </text>
  </svg>
);

export const IconSpreadsheet = (p: Props) => (
  <svg {...base(p)}>
    <rect x="3" y="4" width="18" height="16" rx="2" />
    <path d="M3 10h18" />
    <path d="M3 15h18" />
    <path d="M9 4v16" />
    <path d="M15 4v16" />
  </svg>
);

export const IconImageFile = (p: Props) => (
  <svg {...base(p)}>
    <rect x="3" y="4" width="18" height="16" rx="2" />
    <circle cx="9" cy="10" r="1.8" />
    <path d="m21 17-4.5-4.5L9 20" />
  </svg>
);

export const IconSparkleLogo = (p: Props) => (
  <svg {...base(p)} viewBox="0 0 41 41">
    <path
      d="M37.5 17.4a10.2 10.2 0 0 0-.9-8.4 10.3 10.3 0 0 0-11-4.9A10.3 10.3 0 0 0 3.5 8.2a10.2 10.2 0 0 0 1.3 12.3 10.3 10.3 0 0 0 .9 8.4 10.3 10.3 0 0 0 11 4.9 10.3 10.3 0 0 0 22-4.1 10.2 10.2 0 0 0-1.2-12.3Zm-15.3 21.4a7.6 7.6 0 0 1-4.9-1.8l.2-.2 8.2-4.7.4-.3v-9.3l3.5 2v9.4a7.6 7.6 0 0 1-7.4 4.9ZM6.3 30.8a7.6 7.6 0 0 1-.9-5.1l.2.2 8.2 4.7.4.2 8-4.6v4l-8.1 4.7a7.6 7.6 0 0 1-7.8-.1v-.1Zm-2-17.7a7.6 7.6 0 0 1 4-3.3v9.6l.4.2 8 4.6-3.5 2-8.1-4.7a7.6 7.6 0 0 1-.8-8.4Zm28 6.5-8-4.6L22 13l8.1-4.7a7.6 7.6 0 0 1 11.3 7.9l-.2-.2-8.2-4.7Zm1.8-2.7v-9.4L26 2.8a7.6 7.6 0 0 1 12.2 9.1l-.1.2a6 6 0 0 1-.2-.1Zm-17.4 3.7-3.5-2v-9.4a7.6 7.6 0 0 1 12.5-5.8l-.2.2L16 8.3l-.4.3Zm2-4.2 3.6-2.1 3.6 2.1v4.2l-3.6 2-3.6-2Z"
      fill="currentColor"
      stroke="none"
    />
  </svg>
);
