import type { FC, SVGProps } from "react";

import {
  IconFile,
  IconFilePDF,
  IconImageFile,
  IconSpreadsheet,
} from "@/components/icons";

type IconComp = FC<SVGProps<SVGSVGElement>>;

export interface FileVisual {
  Icon: IconComp;
  /** Tailwind text color class for the icon */
  color: string;
  /** Unicode emoji for plain-text rendering (user bubble) */
  emoji: string;
  /** Short human label (e.g. "PDF", "Spreadsheet") */
  label: string;
}

const EXT_MAP: Record<string, FileVisual> = {
  pdf: { Icon: IconFilePDF, color: "text-red-400", emoji: "📄", label: "PDF" },

  csv: { Icon: IconSpreadsheet, color: "text-emerald-400", emoji: "📊", label: "CSV" },
  tsv: { Icon: IconSpreadsheet, color: "text-emerald-400", emoji: "📊", label: "TSV" },
  xls: { Icon: IconSpreadsheet, color: "text-emerald-400", emoji: "📊", label: "Excel" },
  xlsx: { Icon: IconSpreadsheet, color: "text-emerald-400", emoji: "📊", label: "Excel" },

  png: { Icon: IconImageFile, color: "text-sky-400", emoji: "🖼️", label: "Image" },
  jpg: { Icon: IconImageFile, color: "text-sky-400", emoji: "🖼️", label: "Image" },
  jpeg: { Icon: IconImageFile, color: "text-sky-400", emoji: "🖼️", label: "Image" },
  gif: { Icon: IconImageFile, color: "text-sky-400", emoji: "🖼️", label: "Image" },
  webp: { Icon: IconImageFile, color: "text-sky-400", emoji: "🖼️", label: "Image" },
  bmp: { Icon: IconImageFile, color: "text-sky-400", emoji: "🖼️", label: "Image" },
  svg: { Icon: IconImageFile, color: "text-sky-400", emoji: "🖼️", label: "Image" },
  tif: { Icon: IconImageFile, color: "text-sky-400", emoji: "🖼️", label: "Image" },
  tiff: { Icon: IconImageFile, color: "text-sky-400", emoji: "🖼️", label: "Image" },
};

const DEFAULT: FileVisual = {
  Icon: IconFile,
  color: "text-text-secondary",
  emoji: "📎",
  label: "File",
};

export function getFileVisual(filename: string): FileVisual {
  const i = filename.lastIndexOf(".");
  if (i < 0) return DEFAULT;
  const ext = filename.slice(i + 1).toLowerCase();
  return EXT_MAP[ext] ?? DEFAULT;
}
