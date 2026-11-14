import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@12/dist/mermaid.esm.min.mjs";
import elkLayouts from "https://cdn.jsdelivr.net/npm/@mermaid-js/layout-elk@1.0.2/dist/mermaid-layout-elk.esm.min.mjs";

mermaid.registerLayoutLoaders(elkLayouts);
mermaid.initialize({ startOnLoad: false });

const sources = new WeakMap();
let activeTheme = null;
let rendering = false;
let queued = false;

function themeName() {
  const scheme = document.body.getAttribute("data-md-color-scheme");
  return scheme === "slate" ? "dark" : "default";
}

function diagramSource(node) {
  if (!sources.has(node)) {
    sources.set(node, node.textContent);
  }
  return sources.get(node);
}

async function renderDiagrams() {
  if (rendering) {
    queued = true;
    return;
  }
  rendering = true;
  try {
    do {
      queued = false;
      await renderOnce();
    } while (queued);
  } finally {
    rendering = false;
  }
}

async function renderOnce() {
  const nodes = [...document.querySelectorAll(".mermaid")];
  if (nodes.length === 0) {
    return;
  }
  const theme = themeName();
  const themeChanged = activeTheme !== theme;
  activeTheme = theme;
  mermaid.initialize({
    startOnLoad: false,
    theme,
    securityLevel: "strict",
    fontFamily: getComputedStyle(document.body).fontFamily,
    sequence: { useMaxWidth: false },
  });

  const pending = [];
  for (const node of nodes) {
    const source = diagramSource(node);
    const processed = node.getAttribute("data-processed") === "true";
    if (processed && !themeChanged) {
      continue;
    }
    node.removeAttribute("data-processed");
    node.textContent = source;
    pending.push(node);
  }
  if (pending.length === 0) {
    return;
  }
  await mermaid.run({ nodes: pending });
}

const diagrams = window.document$;
if (diagrams && typeof diagrams.subscribe === "function") {
  diagrams.subscribe(() => {
    void renderDiagrams();
  });
} else {
  document.addEventListener("DOMContentLoaded", () => {
    void renderDiagrams();
  });
}

new MutationObserver(() => {
  void renderDiagrams();
}).observe(document.body, {
  attributes: true,
  attributeFilter: ["data-md-color-scheme"],
});
