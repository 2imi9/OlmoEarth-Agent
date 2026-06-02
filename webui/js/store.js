/* Shared cross-module state. BRIDGE flips to live once /api/health answers
   (see main's detectBridge); while false the page behaves as the static demo.
   It is an object, so every importing module sees the same live mutations. */
export const BRIDGE = { live: false, checked: false, llmLocalUp: true, claudeAvailable: true };
