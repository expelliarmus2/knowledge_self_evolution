/* 知识库自演进 · 前端逻辑：vis-network 渲染检索子图 / 全图，演进操作面板 */
"use strict";

const statusEl = document.getElementById("status-bar");
const detailEl = document.getElementById("detail-content");
let network = null;
let currentResult = null;

function setStatus(msg, isError) {
  statusEl.textContent = msg;
  statusEl.className = isError ? "error" : "";
}

// ---------- 环境自检：JS 引擎 / vis 库 / DOM 就绪 ----------
(function selfCheck() {
  const errs = [];
  if (typeof window.fetch !== "function") errs.push("浏览器不支持 fetch（请换 Chrome/Edge 新版）");
  if (typeof window.vis === "undefined" || typeof vis.Network !== "function") {
    errs.push("vis-network 库未加载成功（检查 web/lib/vis-network.min.js）");
  }
  const canvas = document.getElementById("graph-canvas");
  if (canvas && canvas.clientHeight < 50) errs.push("图区高度为 " + canvas.clientHeight + "px（布局异常）");
  if (errs.length) {
    setStatus("初始化失败：" + errs.join("；"), true);
    return;
  }
  setStatus("就绪。点击「检索子图」或「全图浏览」。（JS 引擎正常）");
})();

// 全局异常捕获：任何 JS 错误直接显示在状态栏（便于远程诊断）
window.onerror = function (msg, src, line, col) {
  setStatus("脚本错误：" + msg + "（" + (src || "") + ":" + line + "）", true);
  return false;
};
window.addEventListener("unhandledrejection", function (e) {
  setStatus("异步错误：" + (e.reason && e.reason.message ? e.reason.message : e.reason), true);
});

function destroyNetwork() {
  if (network) { network.destroy(); network = null; }
}

function toVisData(subgraph) {
  const nodes = new vis.DataSet();
  const edges = new vis.DataSet();
  const nodeIds = new Set();
  (subgraph.nodes || []).forEach(n => {
    let color, shape = "dot", size = 12, border = "#2c3a55";
    if (n.type === "entity") {
      color = { background: n.matched ? "#4a90e2" : "#2f5b96", border: "#7fb0ff" };
      size = n.matched ? 15 : 10;
    } else { // chunk
      const isSeed = n.seed, isHop = (n.hop || 0) > 0;
      if (isSeed) color = { background: n.source === "both" ? "#16a085" : (n.source === "triple" ? "#e67e22" : "#2ecc71"), border: "#d8f5e2" };
      else if (isHop) color = { background: "#e67e22", border: "#f5c99b" };
      else color = { background: "#8e44ad", border: "#c39bd3" };
      shape = "square";
      size = 10;
      if (n.rank && n.rank <= 10) { border = "#f5c542"; size = 14; }
    }
    nodes.add({
      id: n.id, label: n.label, shape, size,
      color, title: n.type === "chunk" ? (n.text || n.id) : (n.label + (n.score ? `\nscore=${n.score}` : "")),
      font: { color: "#c8d4e8", size: 11 },
      _raw: n,
    });
    nodeIds.add(n.id);
  });
  let eIdx = 0;
  (subgraph.edges || []).forEach(e => {
    if (!nodeIds.has(e.source) || !nodeIds.has(e.target)) return;
    let opts = { color: { color: "#55617a", highlight: "#7fb0ff" }, dashes: false, width: 1 };
    if (e.type === "triple") {
      opts = { color: { color: e.filtered ? "#f5c542" : "#b03060", highlight: "#ffd97a" }, width: e.filtered ? 2.5 : 1.5 };
    } else if (e.type === "chunk_hop") {
      opts = { color: { color: "#e67e22", highlight: "#f5c99b" }, width: 1.2, dashes: true };
    } else if (e.type === "text_conclude") {
      opts = { color: { color: "#44546f", highlight: "#7fb0ff" }, width: 1, dashes: [2, 4] };
    }
    edges.add({
      id: `e${eIdx++}`, from: e.source, to: e.target,
      label: e.type === "triple" ? e.relation : (e.type === "chunk_hop" ? `hop${e.hop || 1}` : ""),
      title: e.type === "triple"
        ? `${e.source} —${e.relation}→ ${e.target}${e.score ? `\nscore=${e.score}` : ""}${e.expanded_hop ? `\n扩展第${e.expanded_hop}跳` : ""}`
        : (e.type === "chunk_hop" ? `共享实体: ${(e.shared_entities || []).join(", ")}` : "实体出现于该 chunk"),
      font: { color: "#93a3c0", size: 9, strokeWidth: 0 },
      ...opts, _raw: e,
    });
  });
  return { nodes, edges };
}

function renderSubgraph(subgraph, msg) {
  destroyNetwork();
  const fallback = document.getElementById("canvas-fallback");
  if (fallback) fallback.parentNode.removeChild(fallback); // 首次渲染成功后移除提示
  const { nodes, edges } = toVisData(subgraph);
  const container = document.getElementById("graph-canvas");
  const data = { nodes, edges };
  const big = nodes.length > 200;
  const options = {
    physics: {
      solver: "forceAtlas2Based", forceAtlas2Based: { gravitationalConstant: -55 },
      stabilization: { iterations: big ? 120 : 250 },   // 大图降迭代，加快首帧
    },
    interaction: { hover: true, tooltipDelay: 120 },
  };
  network = new vis.Network(container, data, options);
  network.on("click", params => {
    if (params.nodes.length > 0) {
      const raw = nodes.get(params.nodes[0])._raw;
      detailEl.textContent = JSON.stringify(raw, null, 2);
      switchTab("detail");
    } else if (params.edges.length > 0) {
      const raw = edges.get(params.edges[0])._raw;
      detailEl.textContent = JSON.stringify(raw, null, 2);
      switchTab("detail");
    }
  });
  setStatus(`${msg}（节点 ${nodes.length} / 边 ${edges.length}）`);
}

// ---------- 检索 ----------
async function doSearch() {
  const query = document.getElementById("query-input").value.trim();
  if (!query) { setStatus("请输入查询。"); return; }
  const topK = parseInt(document.getElementById("topk-input").value, 10) || 10;
  const hops = parseInt(document.getElementById("hops-input").value, 10);
  const mode = document.getElementById("subgraph-mode").value;
  const traceEnt = parseInt(document.getElementById("trace-entities").value, 10) || 30;
  const t0 = performance.now();
  setStatus("检索中…");
  try {
    const resp = await fetch("/api/search", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, top_k: topK, hops, subgraph_mode: mode,
                             trace_max_entities: mode === "trace" ? traceEnt : undefined }),
    });
    const data = await resp.json();
    if (!resp.ok) { setStatus(`检索失败: ${data.error || resp.status}`); return; }
    currentResult = data;
    renderSearchResult(data);
    const sg = data.subgraph || { nodes: [], edges: [] };
    const ms = (performance.now() - t0).toFixed(0);
    renderSubgraph(sg, `「${query}」${sg.mode === "trace" ? "溯源" : "PPR"}子图：` +
      `${sg.nodes.length}节点/${sg.edges.length}边 · 后端${(data.timing && data.timing.total) || "?"}s · 端到端${ms}ms`);
  } catch (err) { setStatus(`请求异常: ${err}`); }
}

function renderSearchResult(data) {
  document.getElementById("filter-mode").textContent = (data.filter_info && data.filter_info.mode) || "";
  const ft = document.getElementById("filtered-triples");
  ft.innerHTML = "";
  if ((data.filtered_triples || []).length === 0) ft.innerHTML = '<div class="empty">无相关三元组</div>';
  (data.filtered_triples || []).forEach(t => {
    const div = document.createElement("div");
    div.className = "triple-item filtered";
    div.innerHTML = `<b>${esc(t[0])}</b> <span class="rel">—${esc(t[1])}→</span> <b>${esc(t[2])}</b>`;
    ft.appendChild(div);
  });
  const mt = document.getElementById("matched-triples");
  mt.innerHTML = "";
  const uniqTriples = dedupeTriples(data.triples || []);
  const total = data.triples_total != null ? data.triples_total : uniqTriples.length;
  if (uniqTriples.length === 0) mt.innerHTML = '<div class="empty">无</div>';
  uniqTriples.slice(0, 30).forEach(t => {
    const div = document.createElement("div");
    div.className = "triple-item" + (isFiltered(data, t) ? " filtered" : "");
    div.innerHTML = `<b>${esc(t.head)}</b> <span class="rel">—${esc(t.relation)}→</span> <b>${esc(t.tail)}</b>` +
      (t.score != null ? `<span class="score">${t.score.toFixed(3)}${t.expanded_hop ? ` · 扩展${t.expanded_hop}跳` : ""}</span>` : "");
    mt.appendChild(div);
  });
  if (total > 30) {
    const div = document.createElement("div");
    div.className = "empty";
    div.textContent = `…候选共 ${total} 条，仅显示前 30 条（完整候选已在后端剪枝）`;
    mt.appendChild(div);
  }
  const rc = document.getElementById("ranked-chunks");
  rc.innerHTML = "";
  if ((data.chunks || []).length === 0) rc.innerHTML = '<div class="empty">无</div>';
  (data.chunks || []).forEach(c => {
    const div = document.createElement("div");
    div.className = "chunk-item";
    div.innerHTML = `<b>#${c.rank}</b> <span class="meta">[${c.source}|hop${c.hop}] score=${c.score}</span>` +
      `<div class="text">${esc((c.text || "").slice(0, 110))}${(c.text || "").length > 110 ? "…" : ""}</div>`;
    div.onclick = () => { detailEl.textContent = JSON.stringify(c, null, 2); switchTab("detail"); };
    rc.appendChild(div);
  });
}

function dedupeTriples(triples) {
  const seen = new Set(); const out = [];
  triples.forEach(t => {
    const k = `${t.head}\t${t.relation}\t${t.tail}`;
    if (!seen.has(k)) { seen.add(k); out.push(t); }
  });
  return out;
}
function isFiltered(data, t) {
  return (data.filtered_triples || []).some(f => f[0] === t.head && f[1] === t.relation && f[2] === t.tail);
}
function esc(s) { return String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])); }

// ---------- 全图 ----------
async function loadFullGraph() {
  setStatus("加载全图…");
  try {
    const resp = await fetch("/api/graph");
    const data = await resp.json();
    if (!resp.ok) { setStatus(`加载失败: ${data.error || resp.status}`); return; }
    renderSubgraph(data, `全图（v${data.version}）`);
  } catch (err) { setStatus(`请求异常: ${err}`); }
}

// ---------- 演进 ----------
async function doEvolve() {
  const raw = document.getElementById("evolve-ops").value.trim();
  const out = document.getElementById("evolve-result");
  if (!raw) { out.textContent = "请输入图操作 JSON。"; return; }
  let ops;
  try { ops = JSON.parse(raw); } catch (e) { out.textContent = `JSON 解析错误: ${e}`; return; }
  setStatus("应用演进操作…");
  try {
    const resp = await fetch("/api/evolve", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ops }),
    });
    const data = await resp.json();
    out.textContent = JSON.stringify(data, null, 2);
    if (resp.ok) {
      setStatus(`演进完成，图版本 → v${data.version}。检索将自动感知新结构。`);
      loadFullGraph();
    } else setStatus(`演进失败: ${data.error || resp.status}`);
  } catch (err) { out.textContent = `请求异常: ${err}`; }
}

// ---------- Tabs ----------
function switchTab(name) {
  document.querySelectorAll(".tab").forEach(t => t.classList.toggle("active", t.dataset.tab === name));
  document.querySelectorAll(".tab-body").forEach(b => { b.style.display = b.id === `tab-${name}` ? "" : "none"; });
}
document.querySelectorAll(".tab").forEach(t => t.onclick = () => switchTab(t.dataset.tab));

document.getElementById("btn-search").onclick = doSearch;
document.getElementById("btn-full-graph").onclick = loadFullGraph;
document.getElementById("btn-evolve").onclick = doEvolve;
document.getElementById("query-input").addEventListener("keydown", e => { if (e.key === "Enter") doSearch(); });

// 初始化：自动尝试加载全图
loadFullGraph();
