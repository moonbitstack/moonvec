// moonvec site: theme, custom scroll behaviour, syntax colour, and the two live demos.
// Everything numeric on the page comes from the compiled library through `globalThis.moonvecDemo`.

(() => {
  const root = document.documentElement;
  if (!window.moonvecDemo) document.body.classList.add("no-engine");

  /* ---------- theme ---------- */

  const media = window.matchMedia("(prefers-color-scheme: dark)");
  const effective = () => root.dataset.theme || (media.matches ? "dark" : "light");
  const syncThemeButton = () => {
    const button = document.getElementById("theme-toggle");
    if (!button) return;
    button.setAttribute("aria-label", effective() === "dark" ? "Switch to the light theme" : "Switch to the dark theme");
  };
  const applyTheme = (theme) => {
    if (theme) {
      root.dataset.theme = theme;
      try {
        sessionStorage.setItem("moonvec-theme", theme);
      } catch {}
    }
    syncThemeButton();
    window.dispatchEvent(new CustomEvent("moonvec:theme"));
  };
  document.getElementById("theme-toggle")?.addEventListener("click", () => {
    applyTheme(effective() === "dark" ? "light" : "dark");
  });
  media.addEventListener("change", syncThemeButton);
  syncThemeButton();

  /* ---------- page scrollbar: visible once the page has moved ---------- */

  const markTop = () => root.classList.toggle("at-top", window.scrollY === 0);
  window.addEventListener("scroll", markTop, { passive: true });
  markTop();

  /* ---------- copy buttons ---------- */

  const copyText = async (text) => {
    try {
      await navigator.clipboard.writeText(text);
      return true;
    } catch {
      const area = document.createElement("textarea");
      area.value = text;
      area.setAttribute("readonly", "");
      area.style.position = "fixed";
      area.style.opacity = "0";
      document.body.appendChild(area);
      area.select();
      const ok = document.execCommand("copy");
      area.remove();
      return ok;
    }
  };

  document.querySelectorAll(".copy").forEach((button) => {
    button.addEventListener("click", async () => {
      const figure = button.closest(".code-figure");
      const install = button.closest(".install");
      const text = figure
        ? figure.querySelector("pre")?.textContent ?? ""
        : install
          ? install.querySelector(".cmd")?.textContent ?? ""
          : "";
      const done = await copyText(text);
      if (!done) return;
      const label = button.textContent;
      button.textContent = "Copied";
      window.setTimeout(() => {
        button.textContent = label;
      }, 1400);
    });
  });

  /* ---------- MoonBit syntax colour ---------- */

  const KEYWORDS = new Set([
    "pub", "priv", "fn", "let", "mut", "match", "for", "in", "if", "else", "raise", "test",
    "struct", "enum", "impl", "trait", "true", "false", "type", "typealias", "extern", "break",
    "continue", "while", "return", "guard", "async", "with", "as", "is", "suberror", "open",
  ]);
  const TOKEN = /(\/\/[^\n]*)|("(?:[^"\\\n]|\\.)*")|(@[a-z_][\w]*)|(\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b)|([A-Za-z_][\w]*)/g;
  const escape = (text) => text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

  const colourise = (code) => {
    let out = "";
    let last = 0;
    for (const match of code.matchAll(TOKEN)) {
      out += escape(code.slice(last, match.index));
      const [whole, comment, string, attribute, number, word] = match;
      if (comment) out += `<span class="tok-com">${escape(whole)}</span>`;
      else if (string) out += `<span class="tok-str">${escape(whole)}</span>`;
      else if (attribute) out += `<span class="tok-type">${escape(whole)}</span>`;
      else if (number) out += `<span class="tok-num">${escape(whole)}</span>`;
      else if (word) {
        if (KEYWORDS.has(word)) out += `<span class="tok-key">${escape(word)}</span>`;
        else if (/^[A-Z]/.test(word)) out += `<span class="tok-type">${escape(word)}</span>`;
        else if (/^\s*\(/.test(code.slice(match.index + word.length))) {
          out += `<span class="tok-fn">${escape(word)}</span>`;
        } else out += escape(word);
      } else out += escape(whole);
      last = match.index + whole.length;
    }
    out += escape(code.slice(last));
    return out;
  };

  document.querySelectorAll('pre[data-lang="mbt"] > code, pre[data-lang="moonbit"] > code').forEach((code) => {
    code.innerHTML = colourise(code.textContent);
  });

  /* ---------- the array panel ---------- */

  const parseBroadcast = (reply) => {
    const parts = reply.split("|");
    if (parts[0] === "err") return { error: parts.slice(1).join("|") };
    const shape = parts[1].split(",").map(Number);
    const values = parts[2].split(" ").filter(Boolean).map(Number);
    return { shape, values };
  };

  const indexInto = (container, count) => {
    if (!container) return;
    container.innerHTML = "";
    for (let i = 0; i < count; i += 1) {
      const mark = document.createElement("span");
      mark.textContent = String(i);
      container.appendChild(mark);
    }
  };

  const frameAxes = (frame, shape) => {
    if (!frame) return;
    const rows = shape.length === 1 ? 1 : shape[0];
    const cols = shape.length === 1 ? shape[0] : shape[shape.length - 1];
    frame.style.setProperty("--cols", String(cols));
    frame.style.setProperty("--rows", String(rows));
    indexInto(frame.querySelector(".axis-cols"), cols);
    indexInto(frame.querySelector(".axis-rows"), rows);
  };

  const clearAxes = (frame) => {
    if (!frame) return;
    frame.querySelector(".axis-cols").innerHTML = "";
    frame.querySelector(".axis-rows").innerHTML = "";
  };

  const fillGrid = (grid, shape, values, small) => {
    grid.innerHTML = "";
    const cols = shape.length === 1 ? shape[0] : shape[shape.length - 1];
    grid.style.setProperty("--cols", String(cols));
    frameAxes(grid.closest(".array-frame"), shape);
    const low = Math.min(...values);
    const high = Math.max(...values);
    const span = high - low || 1;
    values.forEach((value, index) => {
      const cell = document.createElement("div");
      cell.className = "cell";
      cell.textContent = Number.isInteger(value) ? String(value) : String(Math.round(value * 100) / 100);
      cell.style.setProperty("--t", String((value - low) / span));
      cell.style.setProperty("--i", String(index));
      grid.appendChild(cell);
    });
    grid.classList.toggle("small", Boolean(small));
  };

  const heroGrid = document.getElementById("hero-grid");
  if (heroGrid) {
    const status = document.getElementById("hero-status");
    const demo = window.moonvecDemo;
    if (demo) {
      try {
        const reply = demo.broadcast("3,1|1 2 3|1,4|10 20 30 40");
        const parsed = parseBroadcast(reply);
        if (parsed.error) throw new Error(parsed.error);
        fillGrid(heroGrid, parsed.shape, parsed.values);
        if (status) status.textContent = "computed in this browser";
      } catch (error) {
        heroGrid.innerHTML = "";
        if (status) status.textContent = "the library refused this shape";
      }
    } else if (status) {
      status.textContent = "engine not loaded";
    }
  }

  /* ---------- playground: broadcasting ---------- */

  /* The select is hidden and the list is drawn here, so the open menu uses the page's own paper,
     rule and teal instead of the operating system's menu. */
  document.querySelectorAll("[data-picker]").forEach((root) => {
    const select = document.getElementById(root.dataset.picker);
    if (!select) return;
    const button = root.querySelector(".picker-button");
    const menu = root.querySelector(".picker-menu");
    const value = root.querySelector(".picker-value");
    const options = [...select.options].map((option, index) => {
      const item = document.createElement("li");
      item.className = "picker-option";
      item.id = `${select.id}-option-${index}`;
      item.tabIndex = -1;
      item.setAttribute("role", "option");
      item.dataset.value = option.value;
      item.textContent = option.textContent.trim();
      menu.appendChild(item);
      return item;
    });

    const sync = () => {
      const selected = options.find((item) => item.dataset.value === select.value) ?? options[0];
      value.textContent = selected.textContent;
      for (const item of options) {
        item.setAttribute("aria-selected", String(item === selected));
      }
    };

    const isOpen = () => button.getAttribute("aria-expanded") === "true";
    const close = (focusButton) => {
      if (!isOpen()) return;
      button.setAttribute("aria-expanded", "false");
      menu.hidden = true;
      root.classList.remove("open");
      if (focusButton) button.focus();
    };
    const open = (focusSelected) => {
      button.setAttribute("aria-expanded", "true");
      menu.hidden = false;
      root.classList.add("open");
      if (focusSelected) {
        const selected = options.find((item) => item.getAttribute("aria-selected") === "true");
        (selected ?? options[0])?.focus();
      }
    };
    const choose = (item) => {
      select.value = item.dataset.value;
      sync();
      close(true);
      select.dispatchEvent(new Event("change"));
    };

    button.addEventListener("click", () => {
      if (isOpen()) close(true);
      else open(false);
    });
    button.addEventListener("keydown", (event) => {
      if (event.key === "ArrowDown" || event.key === "ArrowUp") {
        event.preventDefault();
        open(true);
      } else if (event.key === "Escape") {
        close(true);
      }
    });
    menu.addEventListener("keydown", (event) => {
      const at = options.indexOf(document.activeElement);
      if (event.key === "ArrowDown") {
        event.preventDefault();
        options[Math.min(options.length - 1, at + 1)].focus();
      } else if (event.key === "ArrowUp") {
        event.preventDefault();
        options[Math.max(0, at - 1)].focus();
      } else if (event.key === "Home") {
        event.preventDefault();
        options[0].focus();
      } else if (event.key === "End") {
        event.preventDefault();
        options[options.length - 1].focus();
      } else if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        if (at >= 0) choose(options[at]);
      } else if (event.key === "Escape") {
        event.preventDefault();
        close(true);
      } else if (event.key === "Tab") {
        close(false);
      }
    });
    menu.addEventListener("click", (event) => {
      const item = event.target.closest(".picker-option");
      if (item) choose(item);
    });
    document.addEventListener("pointerdown", (event) => {
      if (!root.contains(event.target)) close(false);
    });
    select.addEventListener("change", sync);
    sync();
  });

  const OPERANDS = {
    "2x3": { shape: [2, 3], values: [1, 2, 3, 4, 5, 6] },
    "3x1": { shape: [3, 1], values: [1, 2, 3] },
    "1x4": { shape: [1, 4], values: [10, 20, 30, 40] },
    "4": { shape: [4], values: [4, 3, 2, 1] },
    "2x2": { shape: [2, 2], values: [1, 2, 3, 4] },
  };

  const shapeText = (shape) => `[${shape.join(", ")}]`;

  const renderPlain = (grid, shape, values, tint) => {
    grid.innerHTML = "";
    const cols = shape.length === 1 ? shape[0] : shape[shape.length - 1];
    grid.style.setProperty("--cols", String(cols));
    frameAxes(grid.closest(".grid-frame"), shape);
    const low = Math.min(...values);
    const high = Math.max(...values);
    const span = high - low || 1;
    values.forEach((value) => {
      const cell = document.createElement("span");
      cell.textContent = String(Math.round(value * 100) / 100);
      if (tint) cell.style.setProperty("--t", String((value - low) / span));
      grid.appendChild(cell);
    });
  };

  const broadcastPage = document.getElementById("broadcast-demo");
  if (broadcastPage) {
    const leftSelect = document.getElementById("operand-a");
    const rightSelect = document.getElementById("operand-b");
    const leftGrid = document.getElementById("grid-a");
    const rightGrid = document.getElementById("grid-b");
    const resultGrid = document.getElementById("grid-sum");
    const leftShape = document.getElementById("shape-a");
    const rightShape = document.getElementById("shape-b");
    const resultShape = document.getElementById("shape-sum");
    const message = document.getElementById("broadcast-message");

    const run = () => {
      const left = OPERANDS[leftSelect.value];
      const right = OPERANDS[rightSelect.value];
      leftShape.textContent = shapeText(left.shape);
      rightShape.textContent = shapeText(right.shape);
      renderPlain(leftGrid, left.shape, left.values, false);
      renderPlain(rightGrid, right.shape, right.values, false);
      resultGrid.classList.add("result");
      const demo = window.moonvecDemo;
      if (!demo) {
        resultGrid.innerHTML = "";
        resultGrid.classList.add("is-empty");
        resultGrid.closest(".grid-frame")?.classList.add("is-empty");
        clearAxes(resultGrid.closest(".grid-frame"));
        resultShape.textContent = "—";
        message.innerHTML = '<span class="err">the compiled engine is not loaded</span>';
        return;
      }
      const request = [
        left.shape.join(","),
        left.values.join(" "),
        right.shape.join(","),
        right.values.join(" "),
      ].join("|");
      const parsed = parseBroadcast(demo.broadcast(request));
      if (parsed.error) {
        resultGrid.innerHTML = "";
        resultGrid.classList.add("is-empty");
        resultGrid.closest(".grid-frame")?.classList.add("is-empty");
        clearAxes(resultGrid.closest(".grid-frame"));
        resultShape.textContent = "—";
        message.innerHTML = "";
        const line = document.createElement("span");
        line.className = "err";
        line.textContent = parsed.error;
        message.appendChild(line);
        return;
      }
      resultGrid.classList.remove("is-empty");
      resultGrid.closest(".grid-frame")?.classList.remove("is-empty");
      resultShape.textContent = shapeText(parsed.shape);
      renderPlain(resultGrid, parsed.shape, parsed.values, true);
      message.textContent = `${shapeText(left.shape)} + ${shapeText(right.shape)} → ${shapeText(parsed.shape)}, one element at a time.`;
    };

    leftSelect.addEventListener("change", run);
    rightSelect.addEventListener("change", run);
    run();
  }

  /* ---------- playground: least squares ---------- */

  const canvas = document.getElementById("fit-canvas");
  if (canvas) {
    const context = canvas.getContext("2d");
    const readoutSlope = document.getElementById("fit-slope");
    const readoutIntercept = document.getElementById("fit-intercept");
    const readoutResidual = document.getElementById("fit-residual");
    const readoutFormula = document.getElementById("fit-formula");
    const reset = document.getElementById("fit-reset");
    const X_MAX = 10;
    const Y_MAX = 12;
    const PAD = { left: 44, right: 16, top: 16, bottom: 32 };
    let points = [
      { x: 1.2, y: 2.4 },
      { x: 2.1, y: 3.2 },
      { x: 3.4, y: 5.1 },
      { x: 4.2, y: 5.4 },
      { x: 5.1, y: 7.2 },
      { x: 6.3, y: 8.1 },
      { x: 7.2, y: 9.6 },
      { x: 8.1, y: 9.9 },
    ];
    let line = null;
    let dragging = -1;

    const style = () => {
      const computed = getComputedStyle(root);
      const read = (name) => computed.getPropertyValue(name).trim();
      return {
        ink: read("--ink"),
        ink2: read("--ink-2"),
        ink3: read("--ink-3"),
        rule: read("--rule"),
        teal: read("--teal"),
        blue: read("--blue"),
        paper: read("--paper"),
        surface: read("--surface"),
      };
    };

    const measure = () => {
      const ratio = window.devicePixelRatio || 1;
      const box = canvas.getBoundingClientRect();
      canvas.width = Math.round(box.width * ratio);
      canvas.height = Math.round(box.height * ratio);
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
      return { width: box.width, height: box.height };
    };

    const plot = (box) => ({
      left: PAD.left,
      top: PAD.top,
      width: box.width - PAD.left - PAD.right,
      height: box.height - PAD.top - PAD.bottom,
    });

    const toScreen = (point, area) => ({
      x: area.left + (point.x / X_MAX) * area.width,
      y: area.top + area.height - (point.y / Y_MAX) * area.height,
    });

    const toData = (clientX, clientY, area) => {
      const box = canvas.getBoundingClientRect();
      const x = ((clientX - box.left - area.left) / area.width) * X_MAX;
      const y = ((area.top + area.height - (clientY - box.top)) / area.height) * Y_MAX;
      return {
        x: Math.min(X_MAX, Math.max(0, x)),
        y: Math.min(Y_MAX, Math.max(0, y)),
      };
    };

    const draw = () => {
      const theme = style();
      const box = measure();
      const area = plot(box);
      context.clearRect(0, 0, box.width, box.height);
      context.fillStyle = theme.paper;
      context.fillRect(0, 0, box.width, box.height);

      context.strokeStyle = theme.rule;
      context.lineWidth = 1;
      context.font = "10px 'Source Code Pro', monospace";
      context.fillStyle = theme.ink3;
      context.textAlign = "center";
      context.textBaseline = "top";
      for (let x = 0; x <= X_MAX; x += 1) {
        const at = area.left + (x / X_MAX) * area.width;
        context.beginPath();
        context.moveTo(Math.round(at) + 0.5, area.top);
        context.lineTo(Math.round(at) + 0.5, area.top + area.height);
        context.stroke();
        if (x % 2 === 0) context.fillText(String(x), at, area.top + area.height + 8);
      }
      context.textAlign = "right";
      context.textBaseline = "middle";
      for (let y = 0; y <= Y_MAX; y += 2) {
        const at = area.top + area.height - (y / Y_MAX) * area.height;
        context.beginPath();
        context.moveTo(area.left, Math.round(at) + 0.5);
        context.lineTo(area.left + area.width, Math.round(at) + 0.5);
        context.stroke();
        context.fillText(String(y), area.left - 8, at);
      }
      context.strokeStyle = theme.ink3;
      context.beginPath();
      context.moveTo(area.left + 0.5, area.top);
      context.lineTo(area.left + 0.5, area.top + area.height + 0.5);
      context.lineTo(area.left + area.width, area.top + area.height + 0.5);
      context.stroke();

      if (line) {
        const start = toScreen({ x: 0, y: line.intercept }, area);
        const end = toScreen({ x: X_MAX, y: line.intercept + line.slope * X_MAX }, area);
        context.strokeStyle = theme.blue;
        context.lineWidth = 1.6;
        context.beginPath();
        context.moveTo(start.x, start.y);
        context.lineTo(end.x, end.y);
        context.stroke();

        context.strokeStyle = `${theme.teal}99`;
        context.lineWidth = 1;
        context.setLineDash([3, 3]);
        for (const point of points) {
          const on = toScreen(point, area);
          const below = toScreen({ x: point.x, y: line.intercept + line.slope * point.x }, area);
          context.beginPath();
          context.moveTo(on.x, on.y);
          context.lineTo(below.x, below.y);
          context.stroke();
        }
        context.setLineDash([]);
      }

      for (const point of points) {
        const at = toScreen(point, area);
        context.beginPath();
        context.arc(at.x, at.y, 4.5, 0, Math.PI * 2);
        context.fillStyle = theme.surface;
        context.fill();
        context.strokeStyle = theme.ink;
        context.lineWidth = 1.4;
        context.stroke();
      }

      context.save();
      context.fillStyle = theme.ink3;
      context.font = "9px 'Source Code Pro', monospace";
      context.textAlign = "right";
      context.textBaseline = "bottom";
      context.fillText("axis 1", area.left + area.width, area.top + area.height + 27);
      context.translate(10, area.top + area.height);
      context.rotate(-Math.PI / 2);
      context.textAlign = "left";
      context.textBaseline = "top";
      context.fillText("axis 0", 0, 0);
      context.restore();
    };

    const format = (value) => {
      if (!Number.isFinite(value)) return "—";
      const rounded = Math.abs(value) < 1e-9 ? 0 : value;
      return String(Number(rounded.toFixed(4)));
    };

    const solve = () => {
      const demo = window.moonvecDemo;
      if (!demo) {
        line = null;
        if (readoutFormula) readoutFormula.textContent = "the compiled engine is not loaded";
        return;
      }
      const request = points.map((point) => `${point.x.toFixed(3)},${point.y.toFixed(3)}`).join(";");
      const reply = demo.fit(request);
      const parts = reply.split("|");
      if (parts[0] === "err") {
        line = null;
        if (readoutFormula) readoutFormula.textContent = parts.slice(1).join("|");
        return;
      }
      const slope = Number(parts[1]);
      const intercept = Number(parts[2]);
      const residual = Number(parts[3]);
      line = { slope, intercept, residual };
      if (readoutSlope) readoutSlope.textContent = format(slope);
      if (readoutIntercept) readoutIntercept.textContent = format(intercept);
      if (readoutResidual) readoutResidual.textContent = format(residual);
      if (readoutFormula) {
        const sign = intercept < 0 ? "−" : "+";
        readoutFormula.textContent = `y = ${format(slope)} x ${sign} ${format(Math.abs(intercept))}`;
      }
    };

    const refresh = () => {
      solve();
      draw();
    };

    canvas.addEventListener("pointerdown", (event) => {
      const box = canvas.getBoundingClientRect();
      const area = plot({ width: box.width, height: box.height });
      let best = -1;
      let distance = 16;
      points.forEach((point, index) => {
        const at = toScreen(point, area);
        const dx = event.clientX - box.left - at.x;
        const dy = event.clientY - box.top - at.y;
        const reach = Math.hypot(dx, dy);
        if (reach < distance) {
          distance = reach;
          best = index;
        }
      });
      if (best >= 0) {
        dragging = best;
        canvas.setPointerCapture(event.pointerId);
        canvas.style.cursor = "grabbing";
      }
    });

    canvas.addEventListener("pointermove", (event) => {
      const box = canvas.getBoundingClientRect();
      const area = plot({ width: box.width, height: box.height });
      if (dragging >= 0) {
        const next = toData(event.clientX, event.clientY, area);
        points[dragging] = next;
        refresh();
      } else {
        const at = toData(event.clientX, event.clientY, area);
        const near = points.some((point) => Math.hypot(point.x - at.x, point.y - at.y) < 0.55);
        canvas.style.cursor = near ? "grab" : "crosshair";
      }
    });

    const release = () => {
      dragging = -1;
      canvas.style.cursor = "crosshair";
    };
    canvas.addEventListener("pointerup", release);
    canvas.addEventListener("pointercancel", release);

    reset?.addEventListener("click", () => {
      points = [
        { x: 1.2, y: 2.4 },
        { x: 2.1, y: 3.2 },
        { x: 3.4, y: 5.1 },
        { x: 4.2, y: 5.4 },
        { x: 5.1, y: 7.2 },
        { x: 6.3, y: 8.1 },
        { x: 7.2, y: 9.6 },
        { x: 8.1, y: 9.9 },
      ];
      refresh();
    });

    new ResizeObserver(() => {
      draw();
    }).observe(canvas);
    window.addEventListener("moonvec:theme", draw);
    refresh();
  }

  /* ---------- api filter ---------- */

  const filter = document.getElementById("api-filter");
  if (filter) {
    const apply = () => {
      const needle = filter.value.trim().toLowerCase();
      const entries = [...document.querySelectorAll(".entry")];
      let shown = 0;
      for (const entry of entries) {
        const hit = needle === "" || entry.textContent.toLowerCase().includes(needle);
        entry.classList.toggle("match", hit);
        if (hit) shown += 1;
      }
      for (const group of document.querySelectorAll(".api-group")) {
        group.classList.toggle("has-match", Boolean(group.querySelector(".entry.match")));
      }
      for (const section of document.querySelectorAll(".api-package")) {
        const has = Boolean(section.querySelector(".entry.match"));
        section.style.display = needle && !has ? "none" : "";
      }
      document.body.classList.toggle("api-filtered", needle !== "");
      const empty = document.getElementById("api-empty");
      if (empty) empty.style.display = needle && shown === 0 ? "block" : "none";
    };
    filter.addEventListener("input", apply);
    apply();
  }

  /* ---------- guide scroll spy ---------- */

  const steps = [...document.querySelectorAll(".step")];
  const links = new Map(
    [...document.querySelectorAll(".guide-nav a")].map((link) => [link.getAttribute("href"), link]),
  );
  if (steps.length && links.size) {
    let current = "";
    const mark = () => {
      const line = window.scrollY + 120;
      let found = steps[0].id;
      for (const step of steps) {
        if (step.offsetTop <= line) found = step.id;
      }
      if (found === current) return;
      current = found;
      for (const link of links.values()) link.classList.remove("active");
      links.get(`#${current}`)?.classList.add("active");
    };
    window.addEventListener("scroll", mark, { passive: true });
    mark();
  }
})();
