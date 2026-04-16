export type Topic = {
  id: string
  title: string
  /** Short hint under the title */
  hint?: string
  /** MATLAB source; omitted for special panels */
  code?: string
  /** Default: MATLAB code block */
  kind?: 'matlab' | 'stealth-gemini'
}
