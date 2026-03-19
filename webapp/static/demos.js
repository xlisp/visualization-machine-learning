/* ====================================================================
   Visualization Machine Learning - Interactive Demos
   ==================================================================== */

const C = {
  bg: '#0a0e17', surface: '#151c2c', border: '#243049',
  text: '#e4e8f1', dim: '#8892a8',
  blue: '#4f8cff', purple: '#6c5ce7', green: '#00d68f',
  red: '#ff6b6b', orange: '#ffa94d', yellow: '#ffd43b', cyan: '#22d3ee',
  white: '#ffffff',
};

function getCtx(id) {
  const canvas = document.getElementById(id);
  if (!canvas) return null;
  const ctx = canvas.getContext('2d');
  return { canvas, ctx, w: canvas.width, h: canvas.height };
}

function clear(ctx, w, h) {
  ctx.fillStyle = C.bg;
  ctx.fillRect(0, 0, w, h);
}

function drawAxes(ctx, w, h, pad = 40) {
  ctx.strokeStyle = C.border;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad, pad); ctx.lineTo(pad, h - pad);
  ctx.lineTo(w - pad, h - pad);
  ctx.stroke();
}

function lerp(a, b, t) { return a + (b - a) * t; }

/* ===================== DEMOS ===================== */
const demos = {};

/* ---------- 1. LEAST SQUARES ---------- */
(function() {
  const d = getCtx('canvas-least-squares');
  if (!d) return;
  const { canvas, ctx, w, h } = d;
  const pad = 50;
  let points = [];
  let a = 0, b = 0;

  function fit() {
    if (points.length < 2) { a = 0; b = 0; return; }
    let sx = 0, sy = 0, sxx = 0, sxy = 0, n = points.length;
    points.forEach(p => { sx += p.x; sy += p.y; sxx += p.x * p.x; sxy += p.x * p.y; });
    const det = n * sxx - sx * sx;
    if (Math.abs(det) < 1e-10) return;
    a = (n * sxy - sx * sy) / det;
    b = (sy * sxx - sx * sxy) / det;
  }

  function draw() {
    clear(ctx, w, h);
    drawAxes(ctx, w, h, pad);

    // Grid
    ctx.strokeStyle = 'rgba(36,48,73,0.5)';
    ctx.lineWidth = 0.5;
    for (let i = 1; i <= 10; i++) {
      const x = pad + (w - 2 * pad) * i / 10;
      const y = pad + (h - 2 * pad) * i / 10;
      ctx.beginPath(); ctx.moveTo(x, pad); ctx.lineTo(x, h - pad); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(pad, y); ctx.lineTo(w - pad, y); ctx.stroke();
    }

    // Fit line
    if (points.length >= 2) {
      fit();
      ctx.strokeStyle = C.blue;
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      const x0 = 0, x1 = 1;
      const px0 = pad + x0 * (w - 2 * pad), py0 = h - pad - (a * x0 + b) * (h - 2 * pad);
      const px1 = pad + x1 * (w - 2 * pad), py1 = h - pad - (a * x1 + b) * (h - 2 * pad);
      ctx.moveTo(px0, py0); ctx.lineTo(px1, py1);
      ctx.stroke();

      // Residuals
      ctx.strokeStyle = 'rgba(255,107,107,0.4)';
      ctx.lineWidth = 1;
      points.forEach(p => {
        const px = pad + p.x * (w - 2 * pad);
        const py = h - pad - p.y * (h - 2 * pad);
        const pyFit = h - pad - (a * p.x + b) * (h - 2 * pad);
        ctx.beginPath(); ctx.moveTo(px, py); ctx.lineTo(px, pyFit); ctx.stroke();
      });

      // Equation
      ctx.fillStyle = C.cyan;
      ctx.font = '14px monospace';
      ctx.fillText(`y = ${a.toFixed(3)}x + ${b.toFixed(3)}`, pad + 10, pad + 20);
      const mse = points.reduce((s, p) => s + (p.y - a * p.x - b) ** 2, 0) / points.length;
      ctx.fillText(`MSE = ${mse.toFixed(5)}  n = ${points.length}`, pad + 10, pad + 40);
    }

    // Points
    points.forEach(p => {
      const px = pad + p.x * (w - 2 * pad);
      const py = h - pad - p.y * (h - 2 * pad);
      ctx.beginPath();
      ctx.arc(px, py, 5, 0, Math.PI * 2);
      ctx.fillStyle = C.orange;
      ctx.fill();
      ctx.strokeStyle = C.white;
      ctx.lineWidth = 1;
      ctx.stroke();
    });

    // Labels
    ctx.fillStyle = C.dim;
    ctx.font = '11px sans-serif';
    ctx.fillText('x', w - pad + 10, h - pad + 5);
    ctx.fillText('y', pad - 5, pad - 10);
  }

  canvas.addEventListener('click', e => {
    const rect = canvas.getBoundingClientRect();
    const scaleX = w / rect.width;
    const scaleY = h / rect.height;
    const mx = (e.clientX - rect.left) * scaleX;
    const my = (e.clientY - rect.top) * scaleY;
    const x = (mx - pad) / (w - 2 * pad);
    const y = (h - pad - my) / (h - 2 * pad);
    if (x >= 0 && x <= 1 && y >= 0 && y <= 1) {
      points.push({ x, y });
      draw();
    }
  });

  demos.leastSquares = {
    reset() { points = []; a = 0; b = 0; draw(); }
  };
  draw();
})();

/* ---------- 2. GRADIENT DESCENT (Convex Opt) ---------- */
(function() {
  const d = getCtx('canvas-gradient-descent');
  if (!d) return;
  const { canvas, ctx, w, h } = d;

  let lr = 0.01;
  let path = [];
  let animId = null;

  // f(x,y) = x^2 + 3*y^2 + 0.5*x*y - 2*x - y + 5
  function f(x, y) { return x * x + 3 * y * y + 0.5 * x * y - 2 * x - y + 5; }
  function grad(x, y) { return [2 * x + 0.5 * y - 2, 6 * y + 0.5 * x - 1]; }

  // Map world [-3,3] x [-3,3] to canvas
  const range = 3;
  function toCanvas(x, y) {
    return [(x + range) / (2 * range) * w, (range - y) / (2 * range) * h];
  }
  function toWorld(px, py) {
    return [px / w * (2 * range) - range, range - py / h * (2 * range)];
  }

  function drawContours() {
    const imageData = ctx.createImageData(w, h);
    let maxVal = 0, minVal = Infinity;
    const vals = new Float32Array(w * h);
    for (let py = 0; py < h; py++) {
      for (let px = 0; px < w; px++) {
        const [wx, wy] = toWorld(px, py);
        const v = f(wx, wy);
        vals[py * w + px] = v;
        maxVal = Math.max(maxVal, v);
        minVal = Math.min(minVal, v);
      }
    }
    for (let i = 0; i < vals.length; i++) {
      const t = (vals[i] - minVal) / (maxVal - minVal);
      const idx = i * 4;
      // Blue-purple gradient
      imageData.data[idx] = Math.floor(10 + t * 40);
      imageData.data[idx + 1] = Math.floor(14 + (1 - t) * 60);
      imageData.data[idx + 2] = Math.floor(23 + (1 - t) * 120);
      imageData.data[idx + 3] = 255;
    }
    ctx.putImageData(imageData, 0, 0);

    // Contour lines
    const levels = 15;
    ctx.strokeStyle = 'rgba(79,140,255,0.2)';
    ctx.lineWidth = 0.5;
    for (let l = 0; l < levels; l++) {
      const threshold = minVal + (maxVal - minVal) * (l / levels);
      for (let py = 1; py < h - 1; py++) {
        for (let px = 1; px < w - 1; px++) {
          const v = vals[py * w + px];
          const vr = vals[py * w + px + 1];
          const vd = vals[(py + 1) * w + px];
          if ((v - threshold) * (vr - threshold) < 0 || (v - threshold) * (vd - threshold) < 0) {
            ctx.fillStyle = 'rgba(79,140,255,0.3)';
            ctx.fillRect(px, py, 1, 1);
          }
        }
      }
    }
  }

  function draw() {
    drawContours();

    // Path
    if (path.length > 1) {
      ctx.strokeStyle = C.red;
      ctx.lineWidth = 2;
      ctx.beginPath();
      const [sx, sy] = toCanvas(path[0].x, path[0].y);
      ctx.moveTo(sx, sy);
      for (let i = 1; i < path.length; i++) {
        const [px, py] = toCanvas(path[i].x, path[i].y);
        ctx.lineTo(px, py);
      }
      ctx.stroke();
    }

    // Points on path
    path.forEach((p, i) => {
      const [px, py] = toCanvas(p.x, p.y);
      ctx.beginPath();
      ctx.arc(px, py, i === path.length - 1 ? 6 : 3, 0, Math.PI * 2);
      ctx.fillStyle = i === path.length - 1 ? C.red : C.orange;
      ctx.fill();
    });

    // Info
    if (path.length > 0) {
      const last = path[path.length - 1];
      ctx.fillStyle = C.cyan;
      ctx.font = '13px monospace';
      ctx.fillText(`pos=(${last.x.toFixed(3)}, ${last.y.toFixed(3)})  f=${f(last.x, last.y).toFixed(4)}  steps=${path.length - 1}`, 10, 20);
    }
  }

  function animate() {
    if (path.length === 0) return;
    const last = path[path.length - 1];
    const [gx, gy] = grad(last.x, last.y);
    const nx = last.x - lr * gx;
    const ny = last.y - lr * gy;
    path.push({ x: nx, y: ny });
    draw();
    if (Math.abs(gx) + Math.abs(gy) > 0.001 && path.length < 500) {
      animId = requestAnimationFrame(animate);
    }
  }

  canvas.addEventListener('click', e => {
    const rect = canvas.getBoundingClientRect();
    const px = (e.clientX - rect.left) * (w / rect.width);
    const py = (e.clientY - rect.top) * (h / rect.height);
    const [wx, wy] = toWorld(px, py);
    if (animId) cancelAnimationFrame(animId);
    path = [{ x: wx, y: wy }];
    animate();
  });

  demos.gradientDescent = {
    reset() { if (animId) cancelAnimationFrame(animId); path = []; draw(); },
    setLR(v) {
      lr = v;
      document.getElementById('gd-lr-val').textContent = v.toFixed(3);
    }
  };
  draw();
})();

/* ---------- 3. LINEAR PROGRAMMING ---------- */
(function() {
  const d = getCtx('canvas-linear-prog');
  if (!d) return;
  const { ctx, w, h } = d;
  const pad = 50;

  function toCanvas(x, y) {
    return [pad + x / 10 * (w - 2 * pad), h - pad - y / 10 * (h - 2 * pad)];
  }

  function draw() {
    clear(ctx, w, h);
    drawAxes(ctx, w, h, pad);

    // Feasible region: x+y<=8, 2x+y<=10, x>=0, y>=0
    // Vertices: (0,0), (5,0), (2,6), (0,8)
    const vertices = [[0, 0], [5, 0], [2, 6], [0, 8]];
    ctx.fillStyle = 'rgba(79,140,255,0.15)';
    ctx.strokeStyle = C.blue;
    ctx.lineWidth = 2;
    ctx.beginPath();
    vertices.forEach(([x, y], i) => {
      const [px, py] = toCanvas(x, y);
      i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
    });
    ctx.closePath();
    ctx.fill();
    ctx.stroke();

    // Constraint lines
    ctx.setLineDash([5, 5]);
    ctx.strokeStyle = 'rgba(255,169,77,0.5)';
    ctx.lineWidth = 1;
    // x + y = 8
    let [x0, y0] = toCanvas(0, 8);
    let [x1, y1] = toCanvas(8, 0);
    ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
    // 2x + y = 10
    [x0, y0] = toCanvas(0, 10);
    [x1, y1] = toCanvas(5, 0);
    ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
    ctx.setLineDash([]);

    // Objective: maximize 3x + 2y
    // Iso-profit lines
    for (let z = 5; z <= 25; z += 5) {
      ctx.strokeStyle = `rgba(0,214,143,${z === 20 ? 0.6 : 0.15})`;
      ctx.lineWidth = z === 20 ? 2 : 1;
      const [ax, ay] = toCanvas(0, z / 2);
      const [bx, by] = toCanvas(z / 3, 0);
      ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke();
    }

    // Optimal point (2, 6) -> 3*2 + 2*6 = 18
    const [ox, oy] = toCanvas(2, 6);
    ctx.beginPath();
    ctx.arc(ox, oy, 8, 0, Math.PI * 2);
    ctx.fillStyle = C.red;
    ctx.fill();
    ctx.strokeStyle = C.white;
    ctx.lineWidth = 2;
    ctx.stroke();

    // Labels
    ctx.fillStyle = C.cyan;
    ctx.font = '13px monospace';
    ctx.fillText('Maximize: 3x + 2y', pad + 10, pad + 20);
    ctx.fillText('Optimal: (2, 6) = 18', pad + 10, pad + 40);

    ctx.fillStyle = C.orange;
    ctx.font = '11px sans-serif';
    const [l1x, l1y] = toCanvas(3, 5.5);
    ctx.fillText('x + y ≤ 8', l1x, l1y);
    const [l2x, l2y] = toCanvas(3.5, 3.5);
    ctx.fillText('2x + y ≤ 10', l2x, l2y);

    // Vertex labels
    ctx.fillStyle = C.dim;
    vertices.forEach(([x, y]) => {
      const [px, py] = toCanvas(x, y);
      ctx.fillText(`(${x},${y})`, px + 8, py - 5);
    });

    // Axis numbers
    ctx.fillStyle = C.dim;
    ctx.font = '10px sans-serif';
    for (let i = 0; i <= 10; i += 2) {
      const [px, py] = toCanvas(i, 0);
      ctx.fillText(i, px - 4, py + 15);
      const [qx, qy] = toCanvas(0, i);
      ctx.fillText(i, qx - 20, qy + 4);
    }
  }
  draw();
})();

/* ---------- 4. SVM ---------- */
(function() {
  const d = getCtx('canvas-svm');
  if (!d) return;
  const { canvas, ctx, w, h } = d;
  const pad = 30;
  let points = [];
  let currentClass = 0;

  function draw() {
    clear(ctx, w, h);

    // Decision boundary (simple linear if we have enough points)
    if (points.filter(p => p.c === 0).length >= 2 && points.filter(p => p.c === 1).length >= 2) {
      // Find centroid of each class
      const c0 = points.filter(p => p.c === 0);
      const c1 = points.filter(p => p.c === 1);
      const m0 = { x: c0.reduce((s, p) => s + p.x, 0) / c0.length, y: c0.reduce((s, p) => s + p.y, 0) / c0.length };
      const m1 = { x: c1.reduce((s, p) => s + p.x, 0) / c1.length, y: c1.reduce((s, p) => s + p.y, 0) / c1.length };

      // Perpendicular bisector
      const mx = (m0.x + m1.x) / 2, my = (m0.y + m1.y) / 2;
      const dx = m1.x - m0.x, dy = m1.y - m0.y;
      const len = Math.sqrt(dx * dx + dy * dy);
      if (len > 0.01) {
        const nx = -dy / len, ny = dx / len;

        // Draw decision regions
        const imageData = ctx.createImageData(w, h);
        for (let py = 0; py < h; py++) {
          for (let px = 0; px < w; px++) {
            const x = px / w, y = py / h;
            const side = (x - mx) * dx + (y - my) * dy;
            const idx = (py * w + px) * 4;
            if (side < 0) {
              imageData.data[idx] = 20; imageData.data[idx + 1] = 25; imageData.data[idx + 2] = 60;
            } else {
              imageData.data[idx] = 40; imageData.data[idx + 1] = 15; imageData.data[idx + 2] = 25;
            }
            imageData.data[idx + 3] = 255;
          }
        }
        ctx.putImageData(imageData, 0, 0);

        // Decision line
        ctx.strokeStyle = C.yellow;
        ctx.lineWidth = 2;
        ctx.setLineDash([]);
        ctx.beginPath();
        ctx.moveTo((mx + nx * 2) * w, (my + ny * 2) * h);
        ctx.lineTo((mx - nx * 2) * w, (my - ny * 2) * h);
        ctx.stroke();

        // Margin lines
        const margin = len * 0.3;
        ctx.strokeStyle = 'rgba(255,212,59,0.3)';
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo((mx + dx / len * margin / 2 + nx * 2) * w, (my + dy / len * margin / 2 + ny * 2) * h);
        ctx.lineTo((mx + dx / len * margin / 2 - nx * 2) * w, (my + dy / len * margin / 2 - ny * 2) * h);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo((mx - dx / len * margin / 2 + nx * 2) * w, (my - dy / len * margin / 2 + ny * 2) * h);
        ctx.lineTo((mx - dx / len * margin / 2 - nx * 2) * w, (my - dy / len * margin / 2 - ny * 2) * h);
        ctx.stroke();
        ctx.setLineDash([]);
      }
    }

    // Points
    points.forEach(p => {
      ctx.beginPath();
      ctx.arc(p.x * w, p.y * h, 7, 0, Math.PI * 2);
      ctx.fillStyle = p.c === 0 ? C.blue : C.red;
      ctx.fill();
      ctx.strokeStyle = C.white;
      ctx.lineWidth = 1.5;
      ctx.stroke();
    });

    ctx.fillStyle = C.dim;
    ctx.font = '12px sans-serif';
    ctx.fillText(`A: ${points.filter(p => p.c === 0).length}  B: ${points.filter(p => p.c === 1).length}`, 10, 20);
  }

  canvas.addEventListener('click', e => {
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width;
    const y = (e.clientY - rect.top) / rect.height;
    points.push({ x, y, c: currentClass });
    draw();
  });

  demos.svm = {
    reset() { points = []; draw(); },
    setClass(c) {
      currentClass = c;
      document.getElementById('svm-c0').className = c === 0 ? 'active-btn' : '';
      document.getElementById('svm-c1').className = c === 1 ? 'active-btn' : '';
    }
  };
  draw();
})();

/* ---------- 5. DECISION TREE ---------- */
(function() {
  const d = getCtx('canvas-decision-tree');
  if (!d) return;
  const { canvas, ctx, w, h } = d;
  let points = [];
  let currentClass = 0;
  let splits = [];

  function findBestSplit(pts) {
    if (pts.length < 2) return null;
    let bestGini = Infinity, bestAxis = 0, bestVal = 0;
    for (let axis = 0; axis < 2; axis++) {
      const vals = [...new Set(pts.map(p => axis === 0 ? p.x : p.y))].sort((a, b) => a - b);
      for (let i = 0; i < vals.length - 1; i++) {
        const thresh = (vals[i] + vals[i + 1]) / 2;
        const left = pts.filter(p => (axis === 0 ? p.x : p.y) <= thresh);
        const right = pts.filter(p => (axis === 0 ? p.x : p.y) > thresh);
        if (left.length === 0 || right.length === 0) continue;
        const giniL = 1 - (left.filter(p => p.c === 0).length / left.length) ** 2 - (left.filter(p => p.c === 1).length / left.length) ** 2;
        const giniR = 1 - (right.filter(p => p.c === 0).length / right.length) ** 2 - (right.filter(p => p.c === 1).length / right.length) ** 2;
        const gini = (left.length * giniL + right.length * giniR) / pts.length;
        if (gini < bestGini) { bestGini = gini; bestAxis = axis; bestVal = thresh; }
      }
    }
    return bestGini < Infinity ? { axis: bestAxis, val: bestVal } : null;
  }

  function draw() {
    clear(ctx, w, h);

    // Draw split regions
    if (splits.length > 0) {
      const imageData = ctx.createImageData(w, h);
      for (let py = 0; py < h; py++) {
        for (let px = 0; px < w; px++) {
          const x = px / w, y = py / h;
          let region = 0;
          splits.forEach(s => {
            const v = s.axis === 0 ? x : y;
            region = region * 2 + (v > s.val ? 1 : 0);
          });
          const idx = (py * w + px) * 4;
          const colors = [[15, 20, 50], [40, 15, 20], [15, 40, 25], [35, 25, 15], [20, 15, 40], [15, 35, 35]];
          const c = colors[region % colors.length];
          imageData.data[idx] = c[0]; imageData.data[idx + 1] = c[1]; imageData.data[idx + 2] = c[2];
          imageData.data[idx + 3] = 255;
        }
      }
      ctx.putImageData(imageData, 0, 0);
    }

    // Split lines
    splits.forEach(s => {
      ctx.strokeStyle = C.yellow;
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 4]);
      ctx.beginPath();
      if (s.axis === 0) {
        ctx.moveTo(s.val * w, 0); ctx.lineTo(s.val * w, h);
      } else {
        ctx.moveTo(0, s.val * h); ctx.lineTo(w, s.val * h);
      }
      ctx.stroke();
      ctx.setLineDash([]);
    });

    // Points
    points.forEach(p => {
      ctx.beginPath();
      ctx.arc(p.x * w, p.y * h, 6, 0, Math.PI * 2);
      ctx.fillStyle = p.c === 0 ? C.blue : C.red;
      ctx.fill();
      ctx.strokeStyle = C.white;
      ctx.lineWidth = 1;
      ctx.stroke();
    });

    ctx.fillStyle = C.dim;
    ctx.font = '12px sans-serif';
    ctx.fillText(`Splits: ${splits.length}`, 10, 20);
  }

  canvas.addEventListener('click', e => {
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width;
    const y = (e.clientY - rect.top) / rect.height;
    points.push({ x, y, c: currentClass });
    draw();
  });

  demos.decisionTree = {
    reset() { points = []; splits = []; draw(); },
    setClass(c) {
      currentClass = c;
      document.getElementById('dt-c0').className = c === 0 ? 'active-btn' : '';
      document.getElementById('dt-c1').className = c === 1 ? 'active-btn' : '';
    },
    split() {
      if (points.length < 4) return;
      const s = findBestSplit(points);
      if (s) { splits.push(s); draw(); }
    }
  };
  draw();
})();

/* ---------- 6. K-MEANS ---------- */
(function() {
  const d = getCtx('canvas-kmeans');
  if (!d) return;
  const { canvas, ctx, w, h } = d;
  let points = [], centroids = [], assignments = [];
  let k = 3, iteration = 0, autoId = null;
  const colors = [C.blue, C.red, C.green, C.orange, C.purple, C.cyan, C.yellow, '#ff69b4'];

  function generateData() {
    k = parseInt(document.getElementById('kmeans-k').value) || 3;
    points = [];
    for (let c = 0; c < k; c++) {
      const cx = 0.15 + Math.random() * 0.7;
      const cy = 0.15 + Math.random() * 0.7;
      for (let i = 0; i < 20 + Math.random() * 20; i++) {
        points.push({ x: cx + (Math.random() - 0.5) * 0.2, y: cy + (Math.random() - 0.5) * 0.2 });
      }
    }
    // Init centroids randomly
    centroids = [];
    for (let i = 0; i < k; i++) {
      centroids.push({ x: 0.1 + Math.random() * 0.8, y: 0.1 + Math.random() * 0.8 });
    }
    assignments = new Array(points.length).fill(-1);
    iteration = 0;
  }

  function step() {
    if (points.length === 0 || centroids.length === 0) return;
    // Assign
    let changed = false;
    points.forEach((p, i) => {
      let minDist = Infinity, minC = 0;
      centroids.forEach((c, ci) => {
        const dist = (p.x - c.x) ** 2 + (p.y - c.y) ** 2;
        if (dist < minDist) { minDist = dist; minC = ci; }
      });
      if (assignments[i] !== minC) changed = true;
      assignments[i] = minC;
    });
    // Update centroids
    centroids.forEach((c, ci) => {
      const assigned = points.filter((_, i) => assignments[i] === ci);
      if (assigned.length > 0) {
        c.x = assigned.reduce((s, p) => s + p.x, 0) / assigned.length;
        c.y = assigned.reduce((s, p) => s + p.y, 0) / assigned.length;
      }
    });
    iteration++;
    draw();
    return changed;
  }

  function draw() {
    clear(ctx, w, h);

    // Voronoi-like background
    if (centroids.length > 0 && assignments.some(a => a >= 0)) {
      const imageData = ctx.createImageData(w, h);
      for (let py = 0; py < h; py += 3) {
        for (let px = 0; px < w; px += 3) {
          const x = px / w, y = py / h;
          let minDist = Infinity, minC = 0;
          centroids.forEach((c, ci) => {
            const dist = (x - c.x) ** 2 + (y - c.y) ** 2;
            if (dist < minDist) { minDist = dist; minC = ci; }
          });
          const color = colors[minC % colors.length];
          const r = parseInt(color.slice(1, 3), 16);
          const g = parseInt(color.slice(3, 5), 16);
          const b = parseInt(color.slice(5, 7), 16);
          for (let dy = 0; dy < 3 && py + dy < h; dy++) {
            for (let dx = 0; dx < 3 && px + dx < w; dx++) {
              const idx = ((py + dy) * w + (px + dx)) * 4;
              imageData.data[idx] = r * 0.12;
              imageData.data[idx + 1] = g * 0.12;
              imageData.data[idx + 2] = b * 0.12;
              imageData.data[idx + 3] = 255;
            }
          }
        }
      }
      ctx.putImageData(imageData, 0, 0);
    }

    // Points
    points.forEach((p, i) => {
      ctx.beginPath();
      ctx.arc(p.x * w, p.y * h, 4, 0, Math.PI * 2);
      ctx.fillStyle = assignments[i] >= 0 ? colors[assignments[i] % colors.length] : C.dim;
      ctx.fill();
    });

    // Centroids
    centroids.forEach((c, ci) => {
      ctx.beginPath();
      ctx.arc(c.x * w, c.y * h, 10, 0, Math.PI * 2);
      ctx.fillStyle = colors[ci % colors.length];
      ctx.fill();
      ctx.strokeStyle = C.white;
      ctx.lineWidth = 3;
      ctx.stroke();
      // Cross
      ctx.strokeStyle = C.white;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(c.x * w - 5, c.y * h); ctx.lineTo(c.x * w + 5, c.y * h);
      ctx.moveTo(c.x * w, c.y * h - 5); ctx.lineTo(c.x * w, c.y * h + 5);
      ctx.stroke();
    });

    ctx.fillStyle = C.cyan;
    ctx.font = '13px monospace';
    ctx.fillText(`K=${k}  Iteration: ${iteration}  Points: ${points.length}`, 10, 20);
  }

  demos.kmeans = {
    reset() { if (autoId) clearInterval(autoId); points = []; centroids = []; assignments = []; iteration = 0; draw(); },
    generate() { if (autoId) clearInterval(autoId); generateData(); draw(); },
    step() { step(); },
    autoRun() {
      if (autoId) { clearInterval(autoId); autoId = null; return; }
      autoId = setInterval(() => {
        const changed = step();
        if (!changed) { clearInterval(autoId); autoId = null; }
      }, 400);
    }
  };
  draw();
})();

/* ---------- 7. NEURAL NETWORK (MLP) ---------- */
(function() {
  const d = getCtx('canvas-neural-net');
  if (!d) return;
  const { canvas, ctx, w, h } = d;

  const layers = [3, 5, 4, 2];
  let weights = [];
  let activations = [];
  let gradients = [];
  let phase = 'idle'; // idle, forward, backward
  let animStep = 0;

  function initWeights() {
    weights = [];
    for (let l = 0; l < layers.length - 1; l++) {
      const lw = [];
      for (let i = 0; i < layers[l]; i++) {
        const nw = [];
        for (let j = 0; j < layers[l + 1]; j++) {
          nw.push((Math.random() - 0.5) * 2);
        }
        lw.push(nw);
      }
      weights.push(lw);
    }
    activations = layers.map(n => new Array(n).fill(0));
    gradients = layers.map(n => new Array(n).fill(0));
    activations[0] = activations[0].map(() => Math.random());
  }

  function getNodePos(l, n) {
    const layerX = 80 + l * (w - 160) / (layers.length - 1);
    const layerH = layers[l] * 50;
    const startY = (h - layerH) / 2 + 25;
    return { x: layerX, y: startY + n * 50 };
  }

  function draw() {
    clear(ctx, w, h);

    // Connections
    for (let l = 0; l < layers.length - 1; l++) {
      for (let i = 0; i < layers[l]; i++) {
        for (let j = 0; j < layers[l + 1]; j++) {
          const p1 = getNodePos(l, i);
          const p2 = getNodePos(l + 1, j);
          const wt = weights[l] ? weights[l][i][j] : 0;
          const alpha = Math.min(Math.abs(wt) * 0.5, 0.8);
          ctx.strokeStyle = wt > 0 ? `rgba(79,140,255,${alpha})` : `rgba(255,107,107,${alpha})`;
          ctx.lineWidth = Math.abs(wt) * 1.5 + 0.5;
          ctx.beginPath();
          ctx.moveTo(p1.x, p1.y);
          ctx.lineTo(p2.x, p2.y);
          ctx.stroke();
        }
      }
    }

    // Signal animation
    if (phase === 'forward' && animStep < layers.length) {
      for (let l = 0; l < animStep; l++) {
        for (let i = 0; i < layers[l]; i++) {
          for (let j = 0; j < layers[l + 1]; j++) {
            const p1 = getNodePos(l, i);
            const p2 = getNodePos(l + 1, j);
            const t = (animStep - l) / layers.length;
            const sx = lerp(p1.x, p2.x, t);
            const sy = lerp(p1.y, p2.y, t);
            ctx.beginPath();
            ctx.arc(sx, sy, 3, 0, Math.PI * 2);
            ctx.fillStyle = C.cyan;
            ctx.fill();
          }
        }
      }
    }

    if (phase === 'backward' && animStep < layers.length) {
      for (let l = layers.length - 1; l > layers.length - 1 - animStep && l > 0; l--) {
        for (let i = 0; i < layers[l]; i++) {
          for (let j = 0; j < layers[l - 1]; j++) {
            const p1 = getNodePos(l, i);
            const p2 = getNodePos(l - 1, j);
            const t = (animStep - (layers.length - 1 - l)) / layers.length;
            const sx = lerp(p1.x, p2.x, t);
            const sy = lerp(p1.y, p2.y, t);
            ctx.beginPath();
            ctx.arc(sx, sy, 3, 0, Math.PI * 2);
            ctx.fillStyle = C.red;
            ctx.fill();
          }
        }
      }
    }

    // Nodes
    for (let l = 0; l < layers.length; l++) {
      for (let n = 0; n < layers[l]; n++) {
        const { x, y } = getNodePos(l, n);
        const act = activations[l][n];
        const grad = gradients[l][n];

        // Glow
        if (Math.abs(act) > 0.1 || Math.abs(grad) > 0.1) {
          const gradient = ctx.createRadialGradient(x, y, 0, x, y, 25);
          const color = phase === 'backward' && grad !== 0 ? C.red : C.blue;
          gradient.addColorStop(0, color.replace(')', ',0.3)').replace('rgb', 'rgba'));
          gradient.addColorStop(1, 'transparent');
          ctx.fillStyle = gradient;
          ctx.fillRect(x - 25, y - 25, 50, 50);
        }

        ctx.beginPath();
        ctx.arc(x, y, 16, 0, Math.PI * 2);
        const brightness = Math.abs(act) * 0.6 + 0.2;
        ctx.fillStyle = `rgba(79,140,255,${brightness})`;
        ctx.fill();
        ctx.strokeStyle = C.border;
        ctx.lineWidth = 2;
        ctx.stroke();

        ctx.fillStyle = C.white;
        ctx.font = '10px monospace';
        ctx.textAlign = 'center';
        ctx.fillText(act.toFixed(1), x, y + 4);
      }
    }
    ctx.textAlign = 'left';

    // Layer labels
    const labels = ['Input', 'Hidden 1', 'Hidden 2', 'Output'];
    ctx.fillStyle = C.dim;
    ctx.font = '11px sans-serif';
    labels.forEach((label, l) => {
      const pos = getNodePos(l, 0);
      ctx.textAlign = 'center';
      ctx.fillText(label, pos.x, h - 15);
    });
    ctx.textAlign = 'left';

    ctx.fillStyle = C.cyan;
    ctx.font = '13px monospace';
    const phaseText = phase === 'idle' ? 'Ready' : phase === 'forward' ? 'Forward Pass →' : '← Backward Pass';
    ctx.fillText(phaseText, 10, 20);
  }

  let animTimer = null;
  function animate() {
    animStep++;
    draw();
    if (animStep < layers.length * 2) {
      animTimer = setTimeout(animate, 200);
    } else {
      // Compute activations
      if (phase === 'forward') {
        for (let l = 1; l < layers.length; l++) {
          for (let j = 0; j < layers[l]; j++) {
            let sum = 0;
            for (let i = 0; i < layers[l - 1]; i++) {
              sum += activations[l - 1][i] * weights[l - 1][i][j];
            }
            activations[l][j] = 1 / (1 + Math.exp(-sum)); // sigmoid
          }
        }
      } else if (phase === 'backward') {
        // Fake gradients for visualization
        for (let l = layers.length - 1; l >= 0; l--) {
          for (let n = 0; n < layers[l]; n++) {
            gradients[l][n] = (Math.random() - 0.5) * 2;
          }
        }
      }
      phase = 'idle';
      draw();
    }
  }

  initWeights();

  demos.neuralNet = {
    reset() {
      if (animTimer) clearTimeout(animTimer);
      initWeights();
      phase = 'idle';
      draw();
    },
    forward() {
      if (animTimer) clearTimeout(animTimer);
      phase = 'forward';
      animStep = 0;
      animate();
    },
    backward() {
      if (animTimer) clearTimeout(animTimer);
      phase = 'backward';
      animStep = 0;
      animate();
    },
    train() {
      // Quick forward + backward
      for (let l = 1; l < layers.length; l++) {
        for (let j = 0; j < layers[l]; j++) {
          let sum = 0;
          for (let i = 0; i < layers[l - 1]; i++) {
            sum += activations[l - 1][i] * weights[l - 1][i][j];
          }
          activations[l][j] = 1 / (1 + Math.exp(-sum));
        }
      }
      // Update weights randomly (demo purposes)
      for (let l = 0; l < weights.length; l++) {
        for (let i = 0; i < weights[l].length; i++) {
          for (let j = 0; j < weights[l][i].length; j++) {
            weights[l][i][j] += (Math.random() - 0.5) * 0.1;
          }
        }
      }
      draw();
    }
  };
  draw();
})();

/* ---------- 8. ACTIVATION FUNCTIONS ---------- */
(function() {
  const d = getCtx('canvas-activation');
  if (!d) return;
  const { canvas, ctx, w, h } = d;
  let currentFunc = 'relu';
  let mouseX = null;

  const funcs = {
    relu: { f: x => Math.max(0, x), df: x => x > 0 ? 1 : 0, range: [-3, 3], name: 'ReLU' },
    sigmoid: { f: x => 1 / (1 + Math.exp(-x)), df: x => { const s = 1 / (1 + Math.exp(-x)); return s * (1 - s); }, range: [-6, 6], name: 'Sigmoid' },
    tanh: { f: x => Math.tanh(x), df: x => 1 - Math.tanh(x) ** 2, range: [-4, 4], name: 'Tanh' },
    softmax: { f: x => 1 / (1 + Math.exp(-x)), df: x => { const s = 1 / (1 + Math.exp(-x)); return s * (1 - s); }, range: [-6, 6], name: 'Softmax (2-class equiv.)' },
  };

  function draw() {
    clear(ctx, w, h);
    const fn = funcs[currentFunc];
    const pad = 50;
    const [xmin, xmax] = fn.range;
    const ymin = -1.5, ymax = 2;

    function toCanvas(x, y) {
      return [pad + (x - xmin) / (xmax - xmin) * (w - 2 * pad),
              h - pad - (y - ymin) / (ymax - ymin) * (h - 2 * pad)];
    }

    // Grid
    ctx.strokeStyle = 'rgba(36,48,73,0.5)';
    ctx.lineWidth = 0.5;
    for (let x = Math.ceil(xmin); x <= xmax; x++) {
      const [px] = toCanvas(x, 0);
      ctx.beginPath(); ctx.moveTo(px, pad); ctx.lineTo(px, h - pad); ctx.stroke();
      ctx.fillStyle = C.dim;
      ctx.font = '10px sans-serif';
      ctx.fillText(x, px - 4, h - pad + 15);
    }
    for (let y = Math.ceil(ymin); y <= ymax; y += 0.5) {
      const [, py] = toCanvas(0, y);
      ctx.beginPath(); ctx.moveTo(pad, py); ctx.lineTo(w - pad, py); ctx.stroke();
    }

    // Axes
    const [ax0] = toCanvas(0, 0);
    const [, ay0] = toCanvas(0, 0);
    ctx.strokeStyle = C.dim;
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(pad, ay0); ctx.lineTo(w - pad, ay0); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(ax0, pad); ctx.lineTo(ax0, h - pad); ctx.stroke();

    // Function curve
    ctx.strokeStyle = C.blue;
    ctx.lineWidth = 3;
    ctx.beginPath();
    for (let px = pad; px <= w - pad; px++) {
      const x = xmin + (px - pad) / (w - 2 * pad) * (xmax - xmin);
      const y = fn.f(x);
      const [cx, cy] = toCanvas(x, y);
      px === pad ? ctx.moveTo(cx, cy) : ctx.lineTo(cx, cy);
    }
    ctx.stroke();

    // Derivative curve
    ctx.strokeStyle = C.orange;
    ctx.lineWidth = 2;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    for (let px = pad; px <= w - pad; px++) {
      const x = xmin + (px - pad) / (w - 2 * pad) * (xmax - xmin);
      const y = fn.df(x);
      const [cx, cy] = toCanvas(x, y);
      px === pad ? ctx.moveTo(cx, cy) : ctx.lineTo(cx, cy);
    }
    ctx.stroke();
    ctx.setLineDash([]);

    // Mouse hover line
    if (mouseX !== null && mouseX >= pad && mouseX <= w - pad) {
      const x = xmin + (mouseX - pad) / (w - 2 * pad) * (xmax - xmin);
      const y = fn.f(x);
      const dy = fn.df(x);
      ctx.strokeStyle = 'rgba(255,255,255,0.3)';
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(mouseX, pad); ctx.lineTo(mouseX, h - pad); ctx.stroke();

      const [px, py] = toCanvas(x, y);
      ctx.beginPath(); ctx.arc(px, py, 6, 0, Math.PI * 2);
      ctx.fillStyle = C.blue; ctx.fill();

      const [, dpy] = toCanvas(x, dy);
      ctx.beginPath(); ctx.arc(mouseX, dpy, 5, 0, Math.PI * 2);
      ctx.fillStyle = C.orange; ctx.fill();

      ctx.fillStyle = C.white;
      ctx.font = '12px monospace';
      ctx.fillText(`x=${x.toFixed(2)}  f(x)=${y.toFixed(3)}  f'(x)=${dy.toFixed(3)}`, pad + 10, pad + 20);
    }

    // Legend
    ctx.fillStyle = C.blue;
    ctx.font = 'bold 14px sans-serif';
    ctx.fillText(`${fn.name}`, w - 200, pad + 20);
    ctx.fillStyle = C.blue;
    ctx.fillRect(w - 200, pad + 30, 20, 3);
    ctx.fillStyle = C.text;
    ctx.font = '11px sans-serif';
    ctx.fillText('f(x)', w - 170, pad + 35);
    ctx.fillStyle = C.orange;
    ctx.fillRect(w - 200, pad + 48, 20, 3);
    ctx.fillStyle = C.text;
    ctx.fillText("f'(x)", w - 170, pad + 53);
  }

  canvas.addEventListener('mousemove', e => {
    const rect = canvas.getBoundingClientRect();
    mouseX = (e.clientX - rect.left) * (w / rect.width);
    draw();
  });
  canvas.addEventListener('mouseleave', () => { mouseX = null; draw(); });

  demos.activation = {
    show(name) {
      currentFunc = name;
      ['relu', 'sigmoid', 'tanh', 'softmax'].forEach(n => {
        const el = document.getElementById('act-' + n);
        if (el) el.className = n === name ? 'active-btn' : '';
      });
      draw();
    }
  };
  draw();
})();

/* ---------- 9. SGD ---------- */
(function() {
  const d = getCtx('canvas-sgd');
  if (!d) return;
  const { canvas, ctx, w, h } = d;

  // Generate noisy linear data
  let trueA = 2, trueB = 0.5;
  let data = [];
  let paramA = 0, paramB = 0;
  let losses = [];
  let batchSize = 16;
  let autoId = null;
  let stepCount = 0;

  function generateData() {
    data = [];
    for (let i = 0; i < 100; i++) {
      const x = Math.random();
      const y = trueA * x + trueB + (Math.random() - 0.5) * 0.5;
      data.push({ x, y });
    }
  }

  function sgdStep() {
    const lr = 0.05;
    const batch = batchSize === 'all' ? data : data.sort(() => Math.random() - 0.5).slice(0, batchSize);
    let gradA = 0, gradB = 0;
    batch.forEach(d => {
      const pred = paramA * d.x + paramB;
      const err = pred - d.y;
      gradA += 2 * err * d.x / batch.length;
      gradB += 2 * err / batch.length;
    });
    paramA -= lr * gradA;
    paramB -= lr * gradB;
    const loss = data.reduce((s, d) => s + (paramA * d.x + paramB - d.y) ** 2, 0) / data.length;
    losses.push(loss);
    stepCount++;
  }

  function draw() {
    clear(ctx, w, h);
    const pad = 45;
    const plotW = (w - pad * 2 - 30) / 2;

    // Left: scatter + fit line
    // Border
    ctx.strokeStyle = C.border;
    ctx.strokeRect(pad, pad, plotW, h - 2 * pad);

    // Data points
    data.forEach(d => {
      const px = pad + d.x * plotW;
      const py = h - pad - d.y / 3 * (h - 2 * pad);
      ctx.beginPath();
      ctx.arc(px, py, 3, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(79,140,255,0.5)';
      ctx.fill();
    });

    // Fit line
    ctx.strokeStyle = C.red;
    ctx.lineWidth = 2;
    ctx.beginPath();
    const y0 = paramB;
    const y1 = paramA + paramB;
    ctx.moveTo(pad, h - pad - y0 / 3 * (h - 2 * pad));
    ctx.lineTo(pad + plotW, h - pad - y1 / 3 * (h - 2 * pad));
    ctx.stroke();

    // True line
    ctx.strokeStyle = C.green;
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    const ty0 = trueB;
    const ty1 = trueA + trueB;
    ctx.moveTo(pad, h - pad - ty0 / 3 * (h - 2 * pad));
    ctx.lineTo(pad + plotW, h - pad - ty1 / 3 * (h - 2 * pad));
    ctx.stroke();
    ctx.setLineDash([]);

    // Right: loss curve
    const rightX = pad + plotW + 30;
    ctx.strokeStyle = C.border;
    ctx.strokeRect(rightX, pad, plotW, h - 2 * pad);

    if (losses.length > 1) {
      const maxLoss = Math.max(...losses.slice(0, 10), 1);
      ctx.strokeStyle = C.orange;
      ctx.lineWidth = 2;
      ctx.beginPath();
      losses.forEach((l, i) => {
        const px = rightX + i / Math.max(losses.length - 1, 1) * plotW;
        const py = h - pad - Math.min(l / maxLoss, 1) * (h - 2 * pad);
        i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
      });
      ctx.stroke();
    }

    // Labels
    ctx.fillStyle = C.dim;
    ctx.font = '11px sans-serif';
    ctx.fillText('Data & Fit', pad + plotW / 2 - 25, pad - 8);
    ctx.fillText('Loss Curve', rightX + plotW / 2 - 25, pad - 8);

    ctx.fillStyle = C.cyan;
    ctx.font = '12px monospace';
    ctx.fillText(`a=${paramA.toFixed(3)} b=${paramB.toFixed(3)}  step=${stepCount}`, pad, 20);
    const curLoss = losses.length > 0 ? losses[losses.length - 1] : '-';
    ctx.fillText(`loss=${typeof curLoss === 'number' ? curLoss.toFixed(5) : curLoss}  batch=${batchSize}`, pad, 36);

    // Legend
    ctx.fillStyle = C.red; ctx.fillRect(pad + 5, h - 25, 15, 2);
    ctx.fillStyle = C.dim; ctx.font = '10px sans-serif'; ctx.fillText('fit', pad + 25, h - 22);
    ctx.fillStyle = C.green; ctx.fillRect(pad + 55, h - 25, 15, 2);
    ctx.fillText('true', pad + 75, h - 22);
  }

  generateData();

  demos.sgd = {
    reset() { if (autoId) { clearInterval(autoId); autoId = null; } paramA = 0; paramB = 0; losses = []; stepCount = 0; generateData(); draw(); },
    step() { sgdStep(); draw(); },
    autoRun() {
      if (autoId) { clearInterval(autoId); autoId = null; return; }
      autoId = setInterval(() => { sgdStep(); draw(); if (stepCount > 200) { clearInterval(autoId); autoId = null; } }, 50);
    },
    setBatch(v) { batchSize = v === 'all' ? 'all' : parseInt(v); }
  };
  draw();
})();

/* ---------- 10. OPTIMIZERS COMPARISON ---------- */
(function() {
  const d = getCtx('canvas-optimizers');
  if (!d) return;
  const { canvas, ctx, w, h } = d;

  // Rosenbrock-like function
  function f(x, y) { return (1 - x) ** 2 + 10 * (y - x * x) ** 2; }
  function grad(x, y) {
    return [-2 * (1 - x) - 40 * x * (y - x * x), 20 * (y - x * x)];
  }

  const range = 2.5;
  function toCanvas(x, y) { return [(x + range) / (2 * range) * w, (range - y) / (2 * range) * h]; }

  let paths = { sgd: [], momentum: [], adam: [] };
  let animId = null;
  let step = 0;

  function drawSurface() {
    const imageData = ctx.createImageData(w, h);
    let maxVal = 0;
    const vals = new Float32Array(w * h);
    for (let py = 0; py < h; py++) {
      for (let px = 0; px < w; px++) {
        const x = px / w * 2 * range - range;
        const y = range - py / h * 2 * range;
        const v = Math.log(f(x, y) + 1);
        vals[py * w + px] = v;
        maxVal = Math.max(maxVal, v);
      }
    }
    for (let i = 0; i < vals.length; i++) {
      const t = vals[i] / maxVal;
      const idx = i * 4;
      imageData.data[idx] = Math.floor(10 + t * 30);
      imageData.data[idx + 1] = Math.floor(14 + (1 - t) * 50);
      imageData.data[idx + 2] = Math.floor(23 + (1 - t) * 80);
      imageData.data[idx + 3] = 255;
    }
    ctx.putImageData(imageData, 0, 0);
  }

  function drawPath(path, color) {
    if (path.length < 2) return;
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    path.forEach((p, i) => {
      const [px, py] = toCanvas(p.x, p.y);
      i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
    });
    ctx.stroke();
    // Current point
    const last = path[path.length - 1];
    const [px, py] = toCanvas(last.x, last.y);
    ctx.beginPath(); ctx.arc(px, py, 5, 0, Math.PI * 2);
    ctx.fillStyle = color; ctx.fill();
    ctx.strokeStyle = C.white; ctx.lineWidth = 1; ctx.stroke();
  }

  function draw() {
    drawSurface();

    // Optimal point (1,1)
    const [ox, oy] = toCanvas(1, 1);
    ctx.beginPath(); ctx.arc(ox, oy, 8, 0, Math.PI * 2);
    ctx.strokeStyle = C.yellow; ctx.lineWidth = 2; ctx.stroke();
    ctx.fillStyle = C.yellow; ctx.font = '11px sans-serif';
    ctx.fillText('min(1,1)', ox + 10, oy);

    drawPath(paths.sgd, C.red);
    drawPath(paths.momentum, C.blue);
    drawPath(paths.adam, C.green);

    // Legend
    ctx.fillStyle = C.bg;
    ctx.globalAlpha = 0.7;
    ctx.fillRect(5, 5, 180, 70);
    ctx.globalAlpha = 1;
    ctx.font = '12px monospace';
    ctx.fillStyle = C.red; ctx.fillText('● SGD (lr=0.001)', 15, 22);
    ctx.fillStyle = C.blue; ctx.fillText('● Momentum (β=0.9)', 15, 40);
    ctx.fillStyle = C.green; ctx.fillText('● Adam (β1=0.9,β2=0.999)', 15, 58);
  }

  function reset() {
    if (animId) cancelAnimationFrame(animId);
    const start = { x: -1.5, y: -1 };
    paths = {
      sgd: [{ ...start }],
      momentum: [{ ...start, vx: 0, vy: 0 }],
      adam: [{ ...start, mx: 0, my: 0, vx: 0, vy: 0, t: 0 }]
    };
    step = 0;
    draw();
  }

  function optimize() {
    step++;
    const lr_sgd = 0.001, lr_mom = 0.001, lr_adam = 0.001;

    // SGD
    const sgdLast = paths.sgd[paths.sgd.length - 1];
    const [gx1, gy1] = grad(sgdLast.x, sgdLast.y);
    paths.sgd.push({ x: sgdLast.x - lr_sgd * gx1, y: sgdLast.y - lr_sgd * gy1 });

    // Momentum
    const momLast = paths.momentum[paths.momentum.length - 1];
    const [gx2, gy2] = grad(momLast.x, momLast.y);
    const beta = 0.9;
    const nvx = beta * momLast.vx + lr_mom * gx2;
    const nvy = beta * momLast.vy + lr_mom * gy2;
    paths.momentum.push({ x: momLast.x - nvx, y: momLast.y - nvy, vx: nvx, vy: nvy });

    // Adam
    const adamLast = paths.adam[paths.adam.length - 1];
    const [gx3, gy3] = grad(adamLast.x, adamLast.y);
    const b1 = 0.9, b2 = 0.999, eps = 1e-8;
    const t = adamLast.t + 1;
    const nmx = b1 * adamLast.mx + (1 - b1) * gx3;
    const nmy = b1 * adamLast.my + (1 - b1) * gy3;
    const nvx2 = b2 * adamLast.vx + (1 - b2) * gx3 * gx3;
    const nvy2 = b2 * adamLast.vy + (1 - b2) * gy3 * gy3;
    const mxh = nmx / (1 - b1 ** t), myh = nmy / (1 - b1 ** t);
    const vxh = nvx2 / (1 - b2 ** t), vyh = nvy2 / (1 - b2 ** t);
    paths.adam.push({
      x: adamLast.x - lr_adam * mxh / (Math.sqrt(vxh) + eps),
      y: adamLast.y - lr_adam * myh / (Math.sqrt(vyh) + eps),
      mx: nmx, my: nmy, vx: nvx2, vy: nvy2, t
    });

    draw();
    if (step < 3000) animId = requestAnimationFrame(optimize);
  }

  demos.optimizers = {
    reset,
    start() { reset(); animId = requestAnimationFrame(optimize); }
  };
  reset();
})();

/* ---------- 11. CNN Drawing Demo ---------- */
(function() {
  const drawCanvas = document.getElementById('canvas-draw');
  const cnnCanvas = document.getElementById('canvas-cnn');
  if (!drawCanvas || !cnnCanvas) return;
  const dCtx = drawCanvas.getContext('2d');
  const cCtx = cnnCanvas.getContext('2d');
  let drawing = false;

  dCtx.fillStyle = '#000';
  dCtx.fillRect(0, 0, 200, 200);
  dCtx.strokeStyle = '#fff';
  dCtx.lineWidth = 12;
  dCtx.lineCap = 'round';

  drawCanvas.addEventListener('mousedown', e => { drawing = true; dCtx.beginPath(); });
  drawCanvas.addEventListener('mouseup', () => drawing = false);
  drawCanvas.addEventListener('mouseleave', () => drawing = false);
  drawCanvas.addEventListener('mousemove', e => {
    if (!drawing) return;
    const rect = drawCanvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) * (200 / rect.width);
    const y = (e.clientY - rect.top) * (200 / rect.height);
    dCtx.lineTo(x, y);
    dCtx.stroke();
    dCtx.beginPath();
    dCtx.moveTo(x, y);
  });

  function drawCNNVisualization() {
    const cw = cnnCanvas.width, ch = cnnCanvas.height;
    cCtx.fillStyle = C.bg;
    cCtx.fillRect(0, 0, cw, ch);

    // Get image data and downsample to 28x28
    const imgData = dCtx.getImageData(0, 0, 200, 200);
    const grid = [];
    for (let y = 0; y < 28; y++) {
      const row = [];
      for (let x = 0; x < 28; x++) {
        const sx = Math.floor(x * 200 / 28);
        const sy = Math.floor(y * 200 / 28);
        const idx = (sy * 200 + sx) * 4;
        row.push(imgData.data[idx] / 255);
      }
      grid.push(row);
    }

    // Draw 28x28 grid
    const cellSize = 4;
    const offsetX = 10, offsetY = 10;
    for (let y = 0; y < 28; y++) {
      for (let x = 0; x < 28; x++) {
        const v = grid[y][x];
        cCtx.fillStyle = `rgb(${Math.floor(v * 255)},${Math.floor(v * 255)},${Math.floor(v * 255)})`;
        cCtx.fillRect(offsetX + x * cellSize, offsetY + y * cellSize, cellSize, cellSize);
      }
    }

    // Simulate conv filter responses
    const filters = [
      [[-1, -1, -1], [0, 0, 0], [1, 1, 1]], // horizontal
      [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], // vertical
      [[-1, -1, 0], [-1, 0, 1], [0, 1, 1]], // diagonal
    ];

    const featureMaps = filters.map(filter => {
      const map = [];
      for (let y = 1; y < 27; y++) {
        const row = [];
        for (let x = 1; x < 27; x++) {
          let sum = 0;
          for (let fy = -1; fy <= 1; fy++) {
            for (let fx = -1; fx <= 1; fx++) {
              sum += grid[y + fy][x + fx] * filter[fy + 1][fx + 1];
            }
          }
          row.push(Math.max(0, sum)); // ReLU
        }
        map.push(row);
      }
      return map;
    });

    // Draw feature maps
    const fmSize = 3;
    const fmLabels = ['Edge H', 'Edge V', 'Edge D'];
    featureMaps.forEach((fm, fi) => {
      const fmX = 140;
      const fmY = 10 + fi * 100;
      let maxVal = 0;
      fm.forEach(row => row.forEach(v => maxVal = Math.max(maxVal, v)));
      for (let y = 0; y < fm.length; y++) {
        for (let x = 0; x < fm[0].length; x++) {
          const v = maxVal > 0 ? fm[y][x] / maxVal : 0;
          const r = Math.floor(v * 79);
          const g = Math.floor(v * 140);
          const b = Math.floor(v * 255);
          cCtx.fillStyle = `rgb(${r},${g},${b})`;
          cCtx.fillRect(fmX + x * fmSize, fmY + y * fmSize, fmSize, fmSize);
        }
      }
      cCtx.fillStyle = C.dim;
      cCtx.font = '10px sans-serif';
      cCtx.fillText(fmLabels[fi], fmX, fmY + 26 * fmSize + 12);
    });

    // Arrow
    cCtx.strokeStyle = C.dim;
    cCtx.lineWidth = 1;
    cCtx.beginPath();
    cCtx.moveTo(125, ch / 2);
    cCtx.lineTo(135, ch / 2);
    cCtx.stroke();

    cCtx.fillStyle = C.cyan;
    cCtx.font = '12px monospace';
    cCtx.fillText('Input 28x28', offsetX, offsetY + 28 * cellSize + 15);
    cCtx.fillText('Conv2D(3x3) + ReLU', 140, ch - 15);
  }

  demos.cnn = {
    clear() {
      dCtx.fillStyle = '#000';
      dCtx.fillRect(0, 0, 200, 200);
      cCtx.fillStyle = C.bg;
      cCtx.fillRect(0, 0, cnnCanvas.width, cnnCanvas.height);
    },
    recognize() { drawCNNVisualization(); }
  };
})();

/* ---------- 12. RNN/LSTM ---------- */
(function() {
  const d = getCtx('canvas-rnn');
  if (!d) return;
  const { canvas, ctx, w, h } = d;
  let timeStep = 0;
  let autoId = null;
  const seqLen = 6;
  let cellState = new Array(seqLen).fill(0);
  let hiddenState = new Array(seqLen).fill(0);
  let inputs = [];

  function reset() {
    timeStep = 0;
    inputs = Array.from({ length: seqLen }, () => Math.random());
    cellState = new Array(seqLen).fill(0);
    hiddenState = new Array(seqLen).fill(0);
  }

  function step() {
    if (timeStep >= seqLen) return;
    const t = timeStep;
    const x = inputs[t];
    const hPrev = t > 0 ? hiddenState[t - 1] : 0;
    const cPrev = t > 0 ? cellState[t - 1] : 0;

    // Simplified LSTM gates
    const forget = 1 / (1 + Math.exp(-(0.5 * x + 0.3 * hPrev - 0.2)));
    const input = 1 / (1 + Math.exp(-(0.8 * x + 0.2 * hPrev)));
    const candidate = Math.tanh(0.6 * x + 0.4 * hPrev);
    cellState[t] = forget * cPrev + input * candidate;
    const output = 1 / (1 + Math.exp(-(0.3 * x + 0.7 * hPrev + 0.1)));
    hiddenState[t] = output * Math.tanh(cellState[t]);
    timeStep++;
  }

  function draw() {
    clear(ctx, w, h);
    const cellW = 70, cellH = 50;
    const startX = 40, startY = h / 2 - cellH / 2;
    const gap = (w - 2 * startX - cellW) / (seqLen - 1);

    // Title
    ctx.fillStyle = C.cyan;
    ctx.font = '13px monospace';
    ctx.fillText(`LSTM  Time Step: ${timeStep}/${seqLen}`, 10, 20);

    // Cell state line (top)
    ctx.strokeStyle = C.green;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(startX, startY - 30);
    ctx.lineTo(startX + (seqLen - 1) * gap + cellW, startY - 30);
    ctx.stroke();
    ctx.fillStyle = C.green;
    ctx.font = '11px sans-serif';
    ctx.fillText('Cell State (C)', startX, startY - 35);

    for (let t = 0; t < seqLen; t++) {
      const x = startX + t * gap;
      const active = t < timeStep;
      const current = t === timeStep - 1;

      // LSTM cell box
      ctx.fillStyle = current ? 'rgba(79,140,255,0.2)' : 'rgba(21,28,44,0.8)';
      ctx.fillRect(x, startY, cellW, cellH);
      ctx.strokeStyle = active ? C.blue : C.border;
      ctx.lineWidth = current ? 2 : 1;
      ctx.strokeRect(x, startY, cellW, cellH);

      // Cell label
      ctx.fillStyle = active ? C.white : C.dim;
      ctx.font = '10px monospace';
      ctx.textAlign = 'center';
      ctx.fillText(`h=${hiddenState[t].toFixed(2)}`, x + cellW / 2, startY + cellH / 2 + 4);

      // Input arrow
      ctx.strokeStyle = active ? C.orange : C.border;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x + cellW / 2, startY + cellH + 30);
      ctx.lineTo(x + cellW / 2, startY + cellH);
      ctx.stroke();
      // Input value
      ctx.fillStyle = active ? C.orange : C.dim;
      ctx.fillText(`x=${inputs[t].toFixed(2)}`, x + cellW / 2, startY + cellH + 45);

      // Cell state values
      if (active) {
        ctx.fillStyle = C.green;
        ctx.fillText(`c=${cellState[t].toFixed(2)}`, x + cellW / 2, startY - 15);
      }

      // Connection to next
      if (t < seqLen - 1) {
        ctx.strokeStyle = active ? C.blue : C.border;
        ctx.lineWidth = active ? 2 : 1;
        ctx.beginPath();
        ctx.moveTo(x + cellW, startY + cellH / 2);
        ctx.lineTo(x + gap, startY + cellH / 2);
        ctx.stroke();
        // Arrow
        ctx.beginPath();
        ctx.moveTo(x + gap - 6, startY + cellH / 2 - 4);
        ctx.lineTo(x + gap, startY + cellH / 2);
        ctx.lineTo(x + gap - 6, startY + cellH / 2 + 4);
        ctx.stroke();
      }

      // Gate indicators
      if (active) {
        const gateY = startY + cellH + 60;
        const gateNames = ['f', 'i', 'o'];
        const gateColors = [C.red, C.green, C.purple];
        gateNames.forEach((g, gi) => {
          ctx.fillStyle = gateColors[gi];
          ctx.font = '9px monospace';
          ctx.fillText(g, x + cellW / 2 - 15 + gi * 15, gateY);
        });
      }

      ctx.textAlign = 'left';
    }

    // Output
    const lastX = startX + (seqLen - 1) * gap;
    if (timeStep > 0) {
      ctx.fillStyle = C.cyan;
      ctx.font = '12px monospace';
      ctx.fillText(`Output: ${hiddenState[timeStep - 1].toFixed(4)}`, lastX + cellW + 10, startY + cellH / 2 + 4);
    }
  }

  reset();

  demos.rnn = {
    reset() { if (autoId) { clearInterval(autoId); autoId = null; } reset(); draw(); },
    step() { step(); draw(); },
    autoRun() {
      if (autoId) { clearInterval(autoId); autoId = null; return; }
      autoId = setInterval(() => {
        if (timeStep >= seqLen) { clearInterval(autoId); autoId = null; return; }
        step(); draw();
      }, 600);
    }
  };
  draw();
})();

/* ---------- 13. SEQ2SEQ ---------- */
(function() {
  const d = getCtx('canvas-seq2seq');
  if (!d) return;
  const { canvas, ctx, w, h } = d;

  const numWords = {
    '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
    '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'
  };

  let encTokens = [], decTokens = [];
  let animStep = 0, animId = null;

  function draw() {
    clear(ctx, w, h);
    const encY = 120, decY = 260;
    const boxW = 55, boxH = 40;

    // Title
    ctx.fillStyle = C.cyan;
    ctx.font = '13px monospace';
    ctx.fillText('Seq2Seq: Encoder → Context → Decoder', 10, 25);

    // Encoder
    ctx.fillStyle = C.dim;
    ctx.font = '12px sans-serif';
    ctx.fillText('Encoder', 10, encY - 15);

    const encStartX = (w - encTokens.length * (boxW + 10)) / 2;
    encTokens.forEach((tok, i) => {
      const x = encStartX + i * (boxW + 10);
      const active = animStep > i;
      ctx.fillStyle = active ? 'rgba(79,140,255,0.2)' : C.surface;
      ctx.fillRect(x, encY, boxW, boxH);
      ctx.strokeStyle = active ? C.blue : C.border;
      ctx.lineWidth = active ? 2 : 1;
      ctx.strokeRect(x, encY, boxW, boxH);
      ctx.fillStyle = active ? C.white : C.dim;
      ctx.font = '16px monospace';
      ctx.textAlign = 'center';
      ctx.fillText(tok, x + boxW / 2, encY + boxH / 2 + 6);

      // Connection
      if (i < encTokens.length - 1) {
        ctx.strokeStyle = active ? C.blue : C.border;
        ctx.beginPath();
        ctx.moveTo(x + boxW, encY + boxH / 2);
        ctx.lineTo(x + boxW + 10, encY + boxH / 2);
        ctx.stroke();
      }
    });

    // Context vector
    const ctxY = (encY + boxH + decY) / 2 - 15;
    const ctxActive = animStep >= encTokens.length;
    ctx.fillStyle = ctxActive ? 'rgba(108,92,231,0.3)' : C.surface;
    const ctxX = w / 2 - 50;
    ctx.fillRect(ctxX, ctxY, 100, 30);
    ctx.strokeStyle = ctxActive ? C.purple : C.border;
    ctx.lineWidth = ctxActive ? 2 : 1;
    ctx.strokeRect(ctxX, ctxY, 100, 30);
    ctx.fillStyle = ctxActive ? C.white : C.dim;
    ctx.font = '11px monospace';
    ctx.textAlign = 'center';
    ctx.fillText('Context Vector', w / 2, ctxY + 20);

    // Arrow from encoder to context
    ctx.strokeStyle = ctxActive ? C.purple : C.border;
    ctx.beginPath();
    ctx.moveTo(w / 2, encY + boxH);
    ctx.lineTo(w / 2, ctxY);
    ctx.stroke();

    // Arrow from context to decoder
    ctx.beginPath();
    ctx.moveTo(w / 2, ctxY + 30);
    ctx.lineTo(w / 2, decY);
    ctx.stroke();

    // Decoder
    ctx.fillStyle = C.dim;
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText('Decoder', 10, decY - 15);

    const decStartX = (w - decTokens.length * (boxW + 10)) / 2;
    decTokens.forEach((tok, i) => {
      const x = decStartX + i * (boxW + 10);
      const active = animStep > encTokens.length + i;
      ctx.fillStyle = active ? 'rgba(0,214,143,0.2)' : C.surface;
      ctx.fillRect(x, decY, boxW, boxH);
      ctx.strokeStyle = active ? C.green : C.border;
      ctx.lineWidth = active ? 2 : 1;
      ctx.strokeRect(x, decY, boxW, boxH);
      ctx.fillStyle = active ? C.white : C.dim;
      ctx.font = '13px monospace';
      ctx.textAlign = 'center';
      ctx.fillText(active ? tok : '?', x + boxW / 2, decY + boxH / 2 + 5);

      if (i < decTokens.length - 1) {
        ctx.strokeStyle = active ? C.green : C.border;
        ctx.beginPath();
        ctx.moveTo(x + boxW, decY + boxH / 2);
        ctx.lineTo(x + boxW + 10, decY + boxH / 2);
        ctx.stroke();
      }
    });

    ctx.textAlign = 'left';

    // Output
    if (animStep >= encTokens.length + decTokens.length) {
      ctx.fillStyle = C.yellow;
      ctx.font = '16px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(`"${decTokens.join(' ')}"`, w / 2, decY + boxH + 40);
      ctx.textAlign = 'left';
    }
  }

  demos.seq2seq = {
    translate() {
      const input = document.getElementById('seq2seq-input').value.trim();
      if (!input) return;
      encTokens = input.split('');
      decTokens = encTokens.map(c => numWords[c] || c);
      animStep = 0;
      if (animId) clearInterval(animId);
      animId = setInterval(() => {
        animStep++;
        draw();
        if (animStep >= encTokens.length + decTokens.length + 1) {
          clearInterval(animId); animId = null;
        }
      }, 300);
    }
  };

  encTokens = ['1', '2', '3'];
  decTokens = ['one', 'two', 'three'];
  draw();
})();

/* ---------- 14. ATTENTION HEATMAP ---------- */
(function() {
  const d = getCtx('canvas-attention');
  if (!d) return;
  const { canvas, ctx, w, h } = d;

  let tokens = ['The', 'cat', 'sat', 'on', 'the', 'mat'];
  let weights = [];

  function randomWeights() {
    const n = tokens.length;
    weights = [];
    for (let i = 0; i < n; i++) {
      const row = [];
      let sum = 0;
      for (let j = 0; j < n; j++) {
        const v = Math.random();
        row.push(v);
        sum += v;
      }
      weights.push(row.map(v => v / sum)); // softmax-like normalization
    }
    // Make diagonal slightly stronger
    weights.forEach((row, i) => {
      row[i] += 0.3;
      const sum = row.reduce((a, b) => a + b, 0);
      row.forEach((_, j) => row[j] /= sum);
    });
  }

  function draw() {
    clear(ctx, w, h);
    const n = tokens.length;
    const pad = 80;
    const cellW = (w - 2 * pad) / n;
    const cellH = (h - 2 * pad) / n;

    // Heatmap
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const v = weights[i] ? weights[i][j] : 0;
        const r = Math.floor(79 * v * 2);
        const g = Math.floor(140 * v * 2);
        const b = Math.floor(255 * v * 2);
        ctx.fillStyle = `rgb(${Math.min(r, 255)},${Math.min(g, 255)},${Math.min(b, 255)})`;
        ctx.fillRect(pad + j * cellW, pad + i * cellH, cellW - 1, cellH - 1);

        // Value text
        ctx.fillStyle = v > 0.3 ? C.white : C.dim;
        ctx.font = '11px monospace';
        ctx.textAlign = 'center';
        ctx.fillText(v.toFixed(2), pad + j * cellW + cellW / 2, pad + i * cellH + cellH / 2 + 4);
      }
    }

    // Labels
    ctx.fillStyle = C.cyan;
    ctx.font = '12px sans-serif';
    tokens.forEach((tok, i) => {
      // Top (Key)
      ctx.textAlign = 'center';
      ctx.fillText(tok, pad + i * cellW + cellW / 2, pad - 10);
      // Left (Query)
      ctx.textAlign = 'right';
      ctx.fillText(tok, pad - 10, pad + i * cellH + cellH / 2 + 4);
    });

    ctx.fillStyle = C.dim;
    ctx.textAlign = 'center';
    ctx.font = '11px sans-serif';
    ctx.fillText('Key →', w / 2, pad - 30);
    ctx.textAlign = 'left';
    ctx.save();
    ctx.translate(pad - 40, h / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = 'center';
    ctx.fillText('Query →', 0, 0);
    ctx.restore();

    ctx.textAlign = 'left';
    ctx.fillStyle = C.cyan;
    ctx.font = '13px monospace';
    ctx.fillText('Attention Weights (Softmax)', 10, 20);
  }

  randomWeights();
  demos.attention = {
    randomize() {
      tokens = ['The', 'cat', 'sat', 'on', 'the', 'mat'];
      randomWeights();
      draw();
    }
  };
  draw();
})();

/* ---------- 15. SELF-ATTENTION (QKV) ---------- */
(function() {
  const d = getCtx('canvas-self-attention');
  if (!d) return;
  const { canvas, ctx, w, h } = d;

  function draw(tokens, weights) {
    clear(ctx, w, h);
    const n = tokens.length;
    if (n === 0) return;

    const padL = 30, padR = 30, padT = 60, padB = 40;
    const tokenW = Math.min(70, (w - padL - padR) / n);
    const startX = (w - n * tokenW) / 2;
    const topY = padT;
    const botY = h - padB - 30;

    // Title
    ctx.fillStyle = C.cyan;
    ctx.font = '13px monospace';
    ctx.fillText('Self-Attention: Q·K^T / √d_k → Softmax → V', 10, 25);

    // Draw connections (attention arcs)
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        if (!weights[i]) continue;
        const v = weights[i][j];
        if (v < 0.05) continue;
        const x1 = startX + i * tokenW + tokenW / 2;
        const x2 = startX + j * tokenW + tokenW / 2;
        const arcH = Math.abs(i - j) * 20 + 20;

        ctx.strokeStyle = `rgba(79,140,255,${Math.min(v * 2, 0.8)})`;
        ctx.lineWidth = v * 5 + 0.5;
        ctx.beginPath();
        ctx.moveTo(x1, topY + 35);
        const cpY = topY + 35 - arcH;
        ctx.quadraticCurveTo((x1 + x2) / 2, cpY, x2, topY + 35);
        ctx.stroke();
      }
    }

    // Tokens (top)
    tokens.forEach((tok, i) => {
      const x = startX + i * tokenW;

      // Token box
      ctx.fillStyle = 'rgba(79,140,255,0.15)';
      ctx.fillRect(x + 2, topY, tokenW - 4, 30);
      ctx.strokeStyle = C.blue;
      ctx.lineWidth = 1;
      ctx.strokeRect(x + 2, topY, tokenW - 4, 30);
      ctx.fillStyle = C.white;
      ctx.font = '12px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(tok, x + tokenW / 2, topY + 20);
    });

    // QKV visualization
    const qkvY = h / 2 + 20;
    const labels = ['Q (Query)', 'K (Key)', 'V (Value)'];
    const qkvColors = [C.blue, C.green, C.orange];
    const barW = (w - 60) / 3;

    labels.forEach((label, li) => {
      const bx = 30 + li * barW;
      ctx.fillStyle = qkvColors[li];
      ctx.font = '11px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(label, bx + barW / 2, qkvY - 8);

      // Random vector visualization
      for (let i = 0; i < n && i < 8; i++) {
        const barH = 15;
        const val = Math.random();
        const bw = val * (barW - 20);
        ctx.fillStyle = `rgba(${parseInt(qkvColors[li].slice(1, 3), 16)},${parseInt(qkvColors[li].slice(3, 5), 16)},${parseInt(qkvColors[li].slice(5, 7), 16)},0.4)`;
        ctx.fillRect(bx + 10, qkvY + i * (barH + 2), bw, barH);
        ctx.strokeStyle = qkvColors[li];
        ctx.lineWidth = 0.5;
        ctx.strokeRect(bx + 10, qkvY + i * (barH + 2), bw, barH);
      }
    });

    ctx.textAlign = 'left';

    // Formula
    ctx.fillStyle = C.dim;
    ctx.font = '12px monospace';
    ctx.fillText('Attention(Q,K,V) = softmax(QK^T/√d_k)V', 10, h - 15);
  }

  demos.selfAttention = {
    compute() {
      const input = document.getElementById('sa-input').value.trim();
      const tokens = input.split(/\s+/).slice(0, 8);
      const n = tokens.length;
      const weights = [];
      for (let i = 0; i < n; i++) {
        const row = [];
        let sum = 0;
        for (let j = 0; j < n; j++) {
          // Simulate: similar words attend to each other
          let v = Math.random() * 0.5;
          if (i === j) v += 0.5;
          if (Math.abs(i - j) <= 1) v += 0.2;
          row.push(v);
          sum += v;
        }
        weights.push(row.map(v => v / sum));
      }
      draw(tokens, weights);
    }
  };

  // Default
  const defaultTokens = ['The', 'cat', 'sat', 'on', 'the', 'mat'];
  const defaultWeights = [];
  for (let i = 0; i < 6; i++) {
    const row = [];
    let sum = 0;
    for (let j = 0; j < 6; j++) {
      let v = Math.random() * 0.3;
      if (i === j) v += 0.6;
      if (Math.abs(i - j) <= 1) v += 0.15;
      row.push(v); sum += v;
    }
    defaultWeights.push(row.map(v => v / sum));
  }
  draw(defaultTokens, defaultWeights);
})();

/* ---------- 16. TRANSFORMER ARCHITECTURE ---------- */
(function() {
  const d = getCtx('canvas-transformer');
  if (!d) return;
  const { canvas, ctx, w, h } = d;

  function drawBlock(x, y, bw, bh, label, color, sublabel) {
    ctx.fillStyle = color.replace(')', ',0.15)').replace('rgb', 'rgba');
    ctx.fillRect(x, y, bw, bh);
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.strokeRect(x, y, bw, bh);
    ctx.fillStyle = C.white;
    ctx.font = 'bold 11px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(label, x + bw / 2, y + bh / 2 + (sublabel ? -2 : 4));
    if (sublabel) {
      ctx.fillStyle = C.dim;
      ctx.font = '9px sans-serif';
      ctx.fillText(sublabel, x + bw / 2, y + bh / 2 + 12);
    }
    ctx.textAlign = 'left';
  }

  function drawArrow(x1, y1, x2, y2) {
    ctx.strokeStyle = C.dim;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x1, y1); ctx.lineTo(x2, y2);
    ctx.stroke();
    const angle = Math.atan2(y2 - y1, x2 - x1);
    ctx.beginPath();
    ctx.moveTo(x2, y2);
    ctx.lineTo(x2 - 6 * Math.cos(angle - 0.3), y2 - 6 * Math.sin(angle - 0.3));
    ctx.lineTo(x2 - 6 * Math.cos(angle + 0.3), y2 - 6 * Math.sin(angle + 0.3));
    ctx.fillStyle = C.dim;
    ctx.fill();
  }

  function draw() {
    clear(ctx, w, h);

    const cx = w / 2;
    const bw = 160, bh = 32, gap = 8;

    // Title
    ctx.fillStyle = C.cyan;
    ctx.font = 'bold 14px monospace';
    ctx.textAlign = 'center';
    ctx.fillText('Transformer Architecture', cx, 25);
    ctx.textAlign = 'left';

    let y = 45;

    // Input Embedding
    drawBlock(cx - bw / 2, y, bw, bh, 'Input Embedding', C.dim, '+ Positional Encoding');
    y += bh + gap;
    drawArrow(cx, y - gap, cx, y);

    // Nx block
    ctx.strokeStyle = C.border;
    ctx.setLineDash([4, 4]);
    ctx.strokeRect(cx - bw / 2 - 20, y - 5, bw + 40, bh * 4 + gap * 5 + 10);
    ctx.setLineDash([]);
    ctx.fillStyle = C.dim;
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'right';
    ctx.fillText('× N', cx + bw / 2 + 15, y + 10);
    ctx.textAlign = 'left';

    y += 5;
    // Multi-Head Attention
    drawBlock(cx - bw / 2, y, bw, bh, 'Multi-Head Attention', C.blue, 'Q  K  V');
    y += bh + gap;
    drawArrow(cx, y - gap, cx, y);

    // Add & Norm
    drawBlock(cx - bw / 2, y, bw, bh * 0.7, 'Add & Norm', C.green);
    y += bh * 0.7 + gap;
    drawArrow(cx, y - gap, cx, y);

    // Feed Forward
    drawBlock(cx - bw / 2, y, bw, bh, 'Feed Forward', C.purple, 'FFN(x) = ReLU(xW₁+b₁)W₂+b₂');
    y += bh + gap;
    drawArrow(cx, y - gap, cx, y);

    // Add & Norm
    drawBlock(cx - bw / 2, y, bw, bh * 0.7, 'Add & Norm', C.green);
    y += bh * 0.7 + gap + 15;
    drawArrow(cx, y - 15, cx, y);

    // Residual connection arrows (left side)
    ctx.strokeStyle = 'rgba(0,214,143,0.3)';
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    // First residual
    ctx.beginPath();
    ctx.moveTo(cx - bw / 2 - 10, 45 + bh + gap + 5);
    ctx.lineTo(cx - bw / 2 - 10, 45 + bh + gap + 5 + bh + gap);
    ctx.lineTo(cx - bw / 2, 45 + bh + gap + 5 + bh + gap);
    ctx.stroke();
    ctx.setLineDash([]);

    // Linear
    drawBlock(cx - bw / 2, y, bw, bh, 'Linear', C.orange);
    y += bh + gap;
    drawArrow(cx, y - gap, cx, y);

    // Softmax
    drawBlock(cx - bw / 2, y, bw, bh, 'Softmax', C.red);
    y += bh + gap;
    drawArrow(cx, y - gap, cx, y);

    // Output
    drawBlock(cx - bw / 2, y, bw, bh, 'Output Probabilities', C.yellow);

    // Side annotations
    ctx.fillStyle = C.dim;
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'left';

    const annotX = cx + bw / 2 + 30;
    ctx.fillText('d_model = 512', annotX, 100);
    ctx.fillText('h = 8 heads', annotX, 120);
    ctx.fillText('d_k = d_v = 64', annotX, 140);
    ctx.fillText('d_ff = 2048', annotX, 160);
    ctx.fillText('N = 6 layers', annotX, 180);
    ctx.fillText('dropout = 0.1', annotX, 200);
  }

  draw();
})();

/* ---------- 17. GRID WORLD (Q-Learning) ---------- */
(function() {
  const d = getCtx('canvas-gridworld');
  if (!d) return;
  const { canvas, ctx, w, h } = d;

  const gridSize = 6;
  const cellSize = Math.min((w - 40) / gridSize, (h - 60) / gridSize);
  const offsetX = (w - gridSize * cellSize) / 2;
  const offsetY = 30;

  let agent = { x: 0, y: 0 };
  const goal = { x: 5, y: 5 };
  const walls = [{ x: 2, y: 1 }, { x: 2, y: 2 }, { x: 2, y: 3 }, { x: 4, y: 3 }, { x: 4, y: 4 }];
  let Q = {};
  let epsilon = 0.3;
  let totalReward = 0;
  let episodes = 0;
  const actions = [[0, -1], [0, 1], [-1, 0], [1, 0]]; // up, down, left, right
  const actionNames = ['↑', '↓', '←', '→'];
  let showPolicyMode = false;

  function key(x, y) { return `${x},${y}`; }
  function getQ(x, y, a) { return (Q[key(x, y)] || [0, 0, 0, 0])[a]; }
  function setQ(x, y, a, v) { if (!Q[key(x, y)]) Q[key(x, y)] = [0, 0, 0, 0]; Q[key(x, y)][a] = v; }
  function isWall(x, y) { return walls.some(w => w.x === x && w.y === y); }
  function isValid(x, y) { return x >= 0 && x < gridSize && y >= 0 && y < gridSize && !isWall(x, y); }

  function bestAction(x, y) {
    let bestA = 0, bestV = -Infinity;
    for (let a = 0; a < 4; a++) {
      const v = getQ(x, y, a);
      if (v > bestV) { bestV = v; bestA = a; }
    }
    return bestA;
  }

  function step() {
    let a;
    if (Math.random() < epsilon) {
      a = Math.floor(Math.random() * 4);
    } else {
      a = bestAction(agent.x, agent.y);
    }
    const nx = agent.x + actions[a][0];
    const ny = agent.y + actions[a][1];

    let reward = -0.1;
    let nextX = agent.x, nextY = agent.y;
    if (isValid(nx, ny)) { nextX = nx; nextY = ny; }
    if (nextX === goal.x && nextY === goal.y) {
      reward = 10;
    } else if (!isValid(nx, ny)) {
      reward = -1;
    }

    // Q-learning update
    const lr = 0.1, gamma = 0.95;
    const maxNextQ = Math.max(...(Q[key(nextX, nextY)] || [0, 0, 0, 0]));
    const oldQ = getQ(agent.x, agent.y, a);
    setQ(agent.x, agent.y, a, oldQ + lr * (reward + gamma * maxNextQ - oldQ));

    totalReward += reward;
    agent.x = nextX;
    agent.y = nextY;

    if (agent.x === goal.x && agent.y === goal.y) {
      episodes++;
      agent = { x: 0, y: 0 };
    }
  }

  function draw() {
    clear(ctx, w, h);

    for (let y = 0; y < gridSize; y++) {
      for (let x = 0; x < gridSize; x++) {
        const px = offsetX + x * cellSize;
        const py = offsetY + y * cellSize;

        // Cell background
        if (isWall(x, y)) {
          ctx.fillStyle = 'rgba(255,107,107,0.3)';
        } else if (x === goal.x && y === goal.y) {
          ctx.fillStyle = 'rgba(0,214,143,0.3)';
        } else {
          // Q-value heatmap
          const maxQ = Math.max(...(Q[key(x, y)] || [0, 0, 0, 0]));
          const intensity = Math.min(Math.max(maxQ / 10, 0), 1);
          ctx.fillStyle = `rgba(79,140,255,${intensity * 0.3})`;
        }
        ctx.fillRect(px, py, cellSize, cellSize);
        ctx.strokeStyle = C.border;
        ctx.lineWidth = 1;
        ctx.strokeRect(px, py, cellSize, cellSize);

        // Wall X
        if (isWall(x, y)) {
          ctx.fillStyle = C.red;
          ctx.font = `${cellSize * 0.5}px sans-serif`;
          ctx.textAlign = 'center';
          ctx.fillText('X', px + cellSize / 2, py + cellSize / 2 + cellSize * 0.15);
        }

        // Goal star
        if (x === goal.x && y === goal.y) {
          ctx.fillStyle = C.green;
          ctx.font = `${cellSize * 0.5}px sans-serif`;
          ctx.textAlign = 'center';
          ctx.fillText('G', px + cellSize / 2, py + cellSize / 2 + cellSize * 0.15);
        }

        // Policy arrows
        if (showPolicyMode && !isWall(x, y) && !(x === goal.x && y === goal.y)) {
          const a = bestAction(x, y);
          ctx.fillStyle = C.yellow;
          ctx.font = `${cellSize * 0.3}px sans-serif`;
          ctx.textAlign = 'center';
          ctx.fillText(actionNames[a], px + cellSize / 2, py + cellSize / 2 + cellSize * 0.1);
        }
      }
    }

    // Agent
    if (!isWall(agent.x, agent.y)) {
      const ax = offsetX + agent.x * cellSize + cellSize / 2;
      const ay = offsetY + agent.y * cellSize + cellSize / 2;
      ctx.beginPath();
      ctx.arc(ax, ay, cellSize * 0.3, 0, Math.PI * 2);
      ctx.fillStyle = C.blue;
      ctx.fill();
      ctx.strokeStyle = C.white;
      ctx.lineWidth = 2;
      ctx.stroke();
    }

    // Info
    ctx.textAlign = 'left';
    ctx.fillStyle = C.cyan;
    ctx.font = '12px monospace';
    const infoY = offsetY + gridSize * cellSize + 20;
    ctx.fillText(`Episodes: ${episodes}  Reward: ${totalReward.toFixed(1)}  ε: ${epsilon.toFixed(2)}`, offsetX, infoY);

    // Q-value table for current cell
    const qvals = Q[key(agent.x, agent.y)] || [0, 0, 0, 0];
    ctx.fillStyle = C.dim;
    ctx.font = '11px monospace';
    ctx.fillText(`Q(${agent.x},${agent.y}): ↑${qvals[0].toFixed(2)} ↓${qvals[1].toFixed(2)} ←${qvals[2].toFixed(2)} →${qvals[3].toFixed(2)}`, offsetX, infoY + 18);
  }

  demos.gridWorld = {
    reset() {
      Q = {}; agent = { x: 0, y: 0 }; totalReward = 0; episodes = 0; showPolicyMode = false; draw();
    },
    step() { step(); draw(); },
    train() {
      for (let i = 0; i < 100; i++) step();
      draw();
    },
    showPolicy() { showPolicyMode = !showPolicyMode; draw(); },
    setEpsilon(v) { epsilon = v; }
  };
  draw();
})();

/* ===================== UTILITY FUNCTIONS ===================== */

// Source code viewer
async function showSource(path) {
  try {
    const resp = await fetch(`/api/source/${path}`);
    const data = await resp.json();
    if (data.code) {
      document.getElementById('modal-title').textContent = path;
      document.getElementById('modal-code').textContent = data.code;
      document.getElementById('source-modal').classList.add('show');
    }
  } catch (e) {
    console.error('Failed to load source:', e);
  }
}

function closeSource() {
  document.getElementById('source-modal').classList.remove('show');
}

// Keyboard shortcut to close modal
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') closeSource();
});

// Navigation path nodes
document.querySelectorAll('.path-node').forEach(node => {
  node.addEventListener('click', () => {
    const target = node.dataset.target;
    if (target) document.getElementById(target)?.scrollIntoView({ behavior: 'smooth' });
  });
});

// Active nav highlight on scroll
const sections = document.querySelectorAll('.section, .hero');
const navLinks = document.querySelectorAll('.nav-links a');

window.addEventListener('scroll', () => {
  let current = '';
  sections.forEach(section => {
    const top = section.offsetTop - 100;
    if (window.scrollY >= top) current = section.id;
  });
  navLinks.forEach(link => {
    link.classList.remove('active');
    if (link.getAttribute('href') === '#' + current) link.classList.add('active');
  });
});

console.log('Visualization Machine Learning - All demos loaded.');
