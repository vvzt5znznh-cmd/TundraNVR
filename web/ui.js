/* Tundra shared chrome, copy, Live + Events pages. */
(function (global) {
  const COPY = {
    brand: "Tundra",
    pages: { live: "Live", events: "Events" },
    seats: { edge: "Edge", node: "Detect", hub: "Verify", operator: "Review" },
    questions: {
      edge: "Unusual for this camera?",
      node: "What is it, and how long?",
      hub: "Alert, or suppress?",
      operator: "Incident, or normal?",
    },
    cascadeIdle: "Edge → Detect → Verify → Review",
    target: "Target",
    running: "This process",
    tripRan: "this trip",
    tripIdle: "idle",
    gap: "not the target",
    baselineReady: "Baseline ready",
    baselineLearn: (pct) => "Learning " + pct + "%",
    polReady: (n) => "Baseline for this camera is ready (" + n + " cells with motion).",
    polLearn: (n, need) =>
      "Learning this camera’s usual footprint — " + n + " / " + need + " cells with repeated motion. Detect still runs; Review is not paged.",
    noAuth: "NO AUTH",
    verifyOffline: (hhmm) => "Verify offline since " + hhmm + " — unusual traffic goes to the Unverified shelf.",
    namerSub: "Detect names objects. It does not filter in recall mode.",
    paged: {
      learning: "Learning this camera",
      sample: "Sample clip — not a live alert",
      unusual: "Unusual for this camera",
      named_object: "Named object",
      rule: "Rule (unattended bag)",
      verify_unavailable: "Unverified — Verify offline",
      verified: "Verify alert",
      audit: "Audit sample",
    },
    emptyUnverified: "No unverified events.",
    unverified: "Unverified",
    setSource: "Set source",
    sourcePh: "RTSP URL, file path, or camera index (0)",
    switching: "Switching…",
    tokenCamera: "API token required to change the camera.",
    tokenReview: "API token required to review.",
    loadFail: "Could not load.",
    incident: "Incident",
    normal: "Normal",
    reviewHint: "Normal teaches Edge what usual looks like. Incident does not.",
    reviewKept: "Kept as an incident.",
    reviewFolded: "Saved as usual for this camera.",
    selectClip: "Select a clip.",
    emptyAll: "No events yet. Unusual motion on this camera will appear here.",
    emptyAlerts: "No alerts waiting.",
    notAlert: "Not an operator alert.",
    alertPrefix: "Alert: ",
    all: "All",
    alerts: "Alerts",
    operatorTitle: "Review",
    details: "Details",
    live: "Live",
    waiting: "Waiting",
    sampleLoop: "Sample",
    noSignal: "No camera on this host.",
    fallbackNote: "No camera on this host — looping a sample clip.",
    error: "Error",
    connecting: "Connecting",
    quiet: "Quiet",
    unusual: "Unusual",
    usual: "Usual",
    learning: "Learning",
    sentDetect: "Sent to Detect",
    quietSub: "Kept on Edge. No detector.",
    unusualSub: "Upload to Detect.",
    usualSub: "Kept on Edge. No detector.",
    learnBit: (n, need) => " Learning " + n + "/" + need + " cells.",
    nothingFrom: "Nothing from Edge",
    nothingSub: "Detector idle — usual motion never leaves Edge.",
    closedSub: "Closed here. Verify is not asked.",
    sendHub: "Named — Verify decides.",
    named: "Named",
    unsure: "Unnamed",
    hubIdle: "Idle",
    hubIdleSub: "Detect did not escalate.",
    hubNeed: "Needs review.",
    hubLog: "Logged without paging.",
    describing: "Verifying…",
    spotted: "Where it was spotted",
    noMark: "No marked still for this event.",
    clip: "Clip",
    sampleWhy:
      "This host is looping a short sample, so the baseline is only the first seconds of that loop — walking the rest of the clip still looks rare.",
  };

  function esc(value) {
    return String(value ?? "").replace(/[&<>"]/g, (c) => ({
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
    }[c]));
  }

  function fmt(ts) {
    const d = new Date(ts);
    return Number.isNaN(d.getTime()) ? (ts || "") : d.toLocaleString();
  }

  function mountChrome(el, { page, seat }) {
    if (!el) return;
    el.innerHTML =
      `<a class="brand" href="/">${esc(COPY.brand)}</a>` +
      `<nav class="pages">` +
      `<a class="${page === "live" ? "active" : ""}" href="/">${esc(COPY.pages.live)}</a>` +
      `<a class="${page === "events" ? "active" : ""}" href="/events">${esc(COPY.pages.events)}</a>` +
      `</nav>` +
      `<div class="seats" id="seats">` +
      ["edge", "node", "hub", "operator"]
        .map(
          (key) =>
            `<button type="button" data-seat="${key}" class="${seat === key ? "on" : ""}">${esc(COPY.seats[key])}</button>`
        )
        .join("") +
      `</div>` +
      `<div class="chrome-meta">` +
      `<span class="dot wait" id="signalDot"></span>` +
      `<span id="signalText">${esc(COPY.connecting)}</span>` +
      `<span id="visionText"></span>` +
      `<span id="versionText"></span>` +
      `</div>`;
  }

  function modelLine(entry) {
    if (!entry) return "";
    const trip = entry.active ? COPY.tripRan : COPY.tripIdle;
    const gap = entry.match ? "" : " (" + COPY.gap + ")";
    return (
      COPY.target +
      " " +
      (entry.want || "—") +
      ". " +
      COPY.running +
      " " +
      (entry.running || "—") +
      gap +
      " · " +
      trip +
      "."
    );
  }

  function fillModelTable(models, seat) {
    const body = document.querySelector("#modelTable tbody");
    if (!body) return;
    const order = [
      ["edge", COPY.seats.edge],
      ["node", COPY.seats.node],
      ["hub", COPY.seats.hub],
    ];
    body.innerHTML = order
      .map(([key, fallback]) => {
        const m = (models && models[key]) || {};
        const gap = m.match ? "" : " gap";
        return `<tr class="${m.active ? "on" : ""}${key === seat ? " current" : ""}">
          <th>${esc(m.label || fallback)}</th>
          <td>${esc(m.want || "—")}</td>
          <td class="${gap}">${esc(m.running || "—")}</td>
          <td>${m.active ? COPY.tripRan : COPY.tripIdle}</td>
        </tr>`;
      })
      .join("");
  }

  function setBar(id, pct) {
    const el = document.getElementById(id);
    if (el) el.style.width = Math.min(100, Math.max(0, pct)) + "%";
  }

  function renderHeat(el, grid, usual) {
    if (!el) return;
    const g = grid || [];
    if (!g.length) {
      el.hidden = true;
      el.innerHTML = "";
      return;
    }
    const u = usual || [];
    const rows = g.length;
    const cols = (g[0] || []).length;
    el.hidden = false;
    el.style.gridTemplateColumns = "repeat(" + cols + ", 1fr)";
    let html = "";
    for (let y = 0; y < rows; y++) {
      for (let x = 0; x < cols; x++) {
        const motion = Number((g[y] || [])[x] || 0);
        const freq = Number((u[y] || [])[x] || 0);
        let bg = "#1c2229";
        if (freq > 0.08) bg = "rgba(61,186,140,0.38)";
        if (motion > 0.12) bg = freq < 0.08 ? "rgba(240,113,103,0.78)" : "rgba(61,186,140,0.8)";
        html += `<i style="background:${bg}" title="usual ${freq.toFixed(2)} · now ${motion.toFixed(2)}"></i>`;
      }
    }
    el.innerHTML = html;
  }

  function renderMeters(el, edge) {
    if (!el) return;
    const rows = [
      ["Place", Number(edge.occupancy_novelty || 0), "Where motion sits vs this camera’s usual cells"],
      ["Look", Number(edge.visual_delta || 0), "How much the frame differs from this camera’s usual look"],
      ["Amount", Number(edge.motion_spike || 0), "How much more motion than this camera usually sees"],
    ];
    el.innerHTML = rows
      .map(([name, val, title]) => {
        const pct = Math.round(Math.min(1, Math.max(0, val)) * 100);
        const hot = val >= 0.48 ? " hot" : "";
        return `<div class="meter" title="${esc(title)}"><span>${esc(name)}</span><div class="bar"><span class="${hot.trim()}" style="width:${pct}%"></span></div><em>${val.toFixed(2)}</em></div>`;
      })
      .join("");
  }

  function coverage(pol) {
    const need = Number(pol.cover_need || pol.learn_samples || 16);
    const n = Number(pol.covered_cells != null ? pol.covered_cells : pol.samples || 0);
    const pct = Math.round((pol.progress || 0) * 100);
    return { need, n, pct, ready: Boolean(pol.confident) };
  }

  function fillLearn(data) {
    const pol = data.pol || {};
    const edge = (data.handoff || {}).edge || {};
    const { need, n, pct, ready } = coverage(pol);
    const polPill = document.getElementById("polPill");
    if (polPill) {
      polPill.textContent = ready ? COPY.baselineReady : COPY.baselineLearn(pct);
      polPill.classList.toggle("wait", !ready);
    }
    const line = ready ? COPY.polReady(n) : COPY.polLearn(n, need);
    const polLine = document.getElementById("polLine");
    if (polLine) polLine.textContent = line;
    const polLineDetails = document.getElementById("polLineDetails");
    if (polLineDetails) polLineDetails.textContent = line;
    setBar("polBar", pct);
    setBar("polBarHud", pct);
    setBar("polBarDetails", pct);
    const hud = document.getElementById("learnHud");
    if (hud) {
      hud.classList.toggle("ready", ready);
      document.getElementById("learnLabel").textContent = ready
        ? COPY.baselineReady
        : "Learning this camera’s usual";
      document.getElementById("learnPct").textContent = pct + "% · " + n + "/" + need;
    }
    const kicker = document.getElementById("learnKicker");
    if (kicker) kicker.textContent = ready ? "This camera’s usual" : "Learning this camera";
    renderHeat(document.getElementById("heat"), edge.grid, edge.usual_grid);
    renderMeters(document.getElementById("whyMeters"), edge);
  }

  function objectChips(items) {
    return (items || [])
      .map((d) => {
        const id = d.track_id != null ? "#" + d.track_id + " " : "";
        const dwell = d.dwell_s != null ? " " + d.dwell_s + "s" : "";
        const zone = d.zone ? " · " + d.zone : "";
        const conf = d.conf != null ? " " + Number(d.conf).toFixed(2) : "";
        return `<span>${esc(id + (d.cls || "object") + conf + dwell + zone)}</span>`;
      })
      .join("");
  }

  function renderCascade(el, handoff, onSeat) {
    if (!el) return;
    const steps = (handoff && handoff.steps) || [];
    if (!steps.length) {
      el.innerHTML = `<li class="idle">${esc(COPY.cascadeIdle)}</li>`;
      return;
    }
    el.innerHTML = steps
      .map((s) => {
        const stop = s.stage === (handoff.stopped_at || "");
        const compact = el.classList.contains("compact");
        if (compact) {
          return `<li><span class="step ${esc(s.decision)}"><b>${esc(s.label)}</b> ${esc(s.decision)}</span></li>`;
        }
        return `<li><button type="button" class="step ${esc(s.decision)} ${stop ? "stop" : ""}" data-stage="${esc(s.stage)}">
          <b>${esc(s.label)}</b><span>${esc(s.decision)}</span><em title="${esc(s.detail || "")}">${esc(s.detail || "")}</em>
        </button></li>`;
      })
      .join("");
    el.querySelectorAll("button[data-stage]").forEach((btn) => {
      btn.addEventListener("click", () => onSeat(btn.getAttribute("data-stage")));
    });
  }

  function live() {
    let seat = "edge";
    const params = new URLSearchParams(location.search);
    const raw = (params.get("seat") || "").toLowerCase();
    if (raw === "raspberry") seat = "edge";
    else if (["edge", "node", "hub", "detect", "verify"].includes(raw)) {
      seat = raw === "detect" ? "node" : raw === "verify" ? "hub" : raw;
    }
    document.body.dataset.seat = seat;
    const chrome = document.getElementById("chrome");
    mountChrome(chrome, { page: "live", seat });
    document.getElementById("seatPill").textContent = COPY.seats[seat];
    document.getElementById("stream").src = "/api/stream.mjpg?seat=" + seat + "&t=" + Date.now();
    let currentSource = "";

    function setSeat(next) {
      if (next === "operator") {
        location.href = "/events";
        return;
      }
      if (next === "raspberry") next = "edge";
      if (next === "detect") next = "node";
      if (next === "verify") next = "hub";
      if (!["edge", "node", "hub"].includes(next)) return;
      seat = next;
      document.body.dataset.seat = seat;
      chrome.querySelectorAll("#seats button").forEach((btn) => {
        btn.classList.toggle("on", btn.getAttribute("data-seat") === seat);
      });
      document.getElementById("seatPill").textContent = COPY.seats[seat];
      document.getElementById("stream").src = "/api/stream.mjpg?seat=" + seat + "&t=" + Date.now();
      refresh();
    }

    chrome.querySelectorAll("#seats button").forEach((btn) => {
      btn.addEventListener("click", () => setSeat(btn.getAttribute("data-seat")));
    });

    async function applySource(source) {
      const msg = document.getElementById("settingsMsg");
      msg.textContent = COPY.switching;
      try {
        const res = await fetch("/api/settings", {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ source }),
        });
        const data = await res.json().catch(() => ({}));
        if (res.status === 401) throw new Error(COPY.tokenCamera);
        if (!res.ok) throw new Error(data.detail || COPY.error);
        currentSource = data.source || source;
        document.getElementById("sourceInput").value = currentSource;
        document.getElementById("stream").src = "/api/stream.mjpg?seat=" + seat + "&t=" + Date.now();
        msg.textContent = "";
        msg.className = "msg";
        refresh();
      } catch (err) {
        msg.textContent = err.message || COPY.error;
        msg.className = "msg err";
      }
    }

    function fillDetails(data) {
      fillLearn(data);
      fillModelTable(data.models, seat);
      const pol = data.pol || {};
      const edge = (data.handoff || {}).edge || {};
      const { need, n, pct } = coverage(pol);
      const tracks =
        (data.tracks || []).map((t) => "#" + t.id + " " + t.cls + " " + t.dwell_s + "s").join(", ") || "—";
      const escalate = data.escalation || {};
      const rows = [
        ["Baseline", (pol.confident ? COPY.baselineReady : COPY.baselineLearn(pct)) + " · " + n + " / " + need],
        ["Unusual score", (edge.score != null ? Number(edge.score).toFixed(2) : "—") + (edge.reason ? " · " + edge.reason : "")],
        ["Why", edge.why || edge.reason || "—"],
        ["Place / look / amount", Number(edge.occupancy_novelty || 0).toFixed(2) + " · " + Number(edge.visual_delta || 0).toFixed(2) + " · " + Number(edge.motion_spike || 0).toFixed(2)],
        ["Tracks", tracks],
        [
          "Escalation",
          (escalate.mode || "auto") +
            " → " +
            (escalate.mode_effective || "—") +
            " · Edge " +
            (escalate.raspberry_trips || 0) +
            " → Detect " +
            (escalate.node_proposals || 0) +
            " → Verify " +
            (escalate.hub_alerts || 0),
        ],
        ["Paged because", JSON.stringify(escalate.paged_because || {})],
        ["Audit", (escalate.audit_shown || 0) + " shown · " + (escalate.audit_confirmed || 0) + " confirmed"],
        ["Latency", (data.latency_ms && data.latency_ms.p50 != null ? "p50 " + data.latency_ms.p50 + " ms · p95 " + data.latency_ms.p95 + " ms" : "—") + (data.clip_drops ? " · clip drops " + data.clip_drops : "")],
        ["Camera", data.source || currentSource || "—"],
        ["Ingest", (data.fps || 0) + " fps · motion " + (data.motion_area || 0)],
      ];
      if (data.last_error) rows.push(["Error", data.last_error]);
      document.getElementById("facts").innerHTML = rows
        .map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`)
        .join("");
    }

    async function refresh() {
      try {
        const data = await (await fetch("/health")).json();
        const opened = Boolean(data.opened);
        const status = document.getElementById("status");
        status.className = "pill " + (opened ? "live" : "wait");
        status.textContent = opened ? (data.fallback ? COPY.sampleLoop : COPY.live) : COPY.waiting;
        const noauth = document.getElementById("noauthPill");
        if (noauth) {
          noauth.hidden = Boolean(data.auth_required);
          noauth.className = "pill err noauth";
          noauth.textContent = COPY.noAuth;
        }
        const verifyBanner = document.getElementById("verifyBanner");
        if (verifyBanner) {
          if (data.verify_offline_since) {
            verifyBanner.hidden = false;
            verifyBanner.textContent = COPY.verifyOffline(data.verify_offline_since);
          } else {
            verifyBanner.hidden = true;
            verifyBanner.textContent = "";
          }
        }
        const nosignal = document.getElementById("nosignal");
        if (nosignal) {
          nosignal.hidden = opened;
          nosignal.textContent = COPY.noSignal;
        }
        const dot = document.getElementById("signalDot");
        const signal = document.getElementById("signalText");
        if (dot) {
          dot.className = "dot " + (opened ? "live" : "wait");
        }
        if (signal) signal.textContent = opened ? (data.fallback ? COPY.fallbackNote : COPY.live) : COPY.waiting;
        const vis = document.getElementById("visionText");
        if (vis) {
          const name = data.vision || "local";
          vis.textContent = data.allow_cloud ? name + " · cloud" : name;
        }
        const ver = document.getElementById("versionText");
        if (ver) ver.textContent = data.display || data.version || "";
        const handoff = data.handoff || {};
        const edge = handoff.edge || {};
        const node = handoff.node || {};
        const hub = handoff.hub || {};
        const pol = data.pol || {};
        const { need, n } = coverage(pol);
        renderCascade(document.getElementById("handoff"), handoff, setSeat);
        fillDetails(data);
        document.getElementById("sceneKicker").textContent = COPY.questions[seat];
        const scene = document.getElementById("scene");
        const sub = document.getElementById("sceneSub");
        const whyEl = document.getElementById("whyDetail");
        const objs = document.getElementById("objects");
        const card = document.getElementById("sceneCard");
        const models = data.models || {};
        const seatKey = seat === "raspberry" ? "edge" : seat;
        const line = document.getElementById("modelLine");
        line.textContent = modelLine(models[seatKey]);
        line.className = "models" + (models[seatKey] && !models[seatKey].match ? " gap" : "");
        card.classList.toggle("alert", Boolean(hub.page_operator && seat === "hub"));
        const learnBit = pol.confident ? "" : COPY.learnBit(n, need);
        const whyText = [edge.why || "", data.fallback ? COPY.sampleWhy : ""].filter(Boolean).join(" ");
        if (seat === "edge") {
          if (!data.last_motion) {
            scene.textContent = COPY.quiet;
            sub.textContent = COPY.quietSub + learnBit;
            whyEl.textContent = "";
          } else if (edge.unusual) {
            scene.textContent = COPY.unusual;
            sub.textContent = (edge.reason || COPY.unusualSub) + learnBit;
            whyEl.textContent = whyText;
          } else if (!pol.confident) {
            scene.textContent = COPY.learning;
            sub.textContent = COPY.polLearn(n, need);
            whyEl.textContent = whyText;
          } else if (data.fallback && edge.upload) {
            scene.textContent = COPY.sentDetect;
            sub.textContent = COPY.sampleWhy;
            whyEl.textContent = "";
          } else {
            scene.textContent = COPY.usual;
            sub.textContent = COPY.usualSub + learnBit;
            whyEl.textContent = edge.why || "";
          }
          objs.innerHTML = "";
        } else if (seat === "node") {
          const dets = node.received ? node.classes || [] : [];
          if (!node.received) {
            scene.textContent = COPY.nothingFrom;
            sub.textContent = COPY.nothingSub;
          } else if (node.closed) {
            scene.textContent =
              node.tracks && node.tracks.length
                ? node.tracks.join(", ")
                : dets.map((d) => d.cls).join(", ") || COPY.named;
            sub.textContent = COPY.closedSub;
          } else {
            scene.textContent =
              node.tracks && node.tracks.length
                ? node.tracks.join(", ")
                : dets.map((d) => d.cls).join(", ") || COPY.unsure;
            sub.textContent = (handoff.mode_effective || (data.escalation || {}).mode_effective) === "recall"
              ? COPY.namerSub
              : COPY.sendHub;
          }
          whyEl.textContent = "";
          objs.innerHTML = objectChips(dets);
        } else if (!hub.ran) {
          scene.textContent = COPY.hubIdle;
          sub.textContent = COPY.hubIdleSub;
          whyEl.textContent = "";
          objs.innerHTML = "";
        } else {
          scene.textContent = data.last_scene || hub.detail || COPY.describing;
          sub.textContent = hub.page_operator ? COPY.hubNeed : COPY.hubLog;
          whyEl.textContent = "";
          objs.innerHTML = objectChips(node.classes || data.last_detections || []);
        }
      } catch (err) {
        document.getElementById("status").textContent = COPY.error;
        document.getElementById("status").className = "pill err";
        const dot = document.getElementById("signalDot");
        if (dot) dot.className = "dot err";
      }
    }

    document.getElementById("sourceForm").addEventListener("submit", (ev) => {
      ev.preventDefault();
      applySource(document.getElementById("sourceInput").value);
    });
    refresh();
    setInterval(refresh, 1500);
    fetch("/api/settings")
      .then((r) => r.json())
      .then((data) => {
        currentSource = data.source || "";
        document.getElementById("sourceInput").value = currentSource;
      })
      .catch(() => {
        document.getElementById("settingsMsg").textContent = COPY.loadFail;
      });
  }

  function events() {
    const chrome = document.getElementById("chrome");
    mountChrome(chrome, { page: "events", seat: "operator" });
    chrome.querySelectorAll("#seats button").forEach((btn) => {
      btn.addEventListener("click", () => {
        const next = btn.getAttribute("data-seat");
        if (next === "operator") return;
        location.href = "/?seat=" + encodeURIComponent(next);
      });
    });
    fetch("/health")
      .then((r) => r.json())
      .then((data) => {
        const opened = Boolean(data.opened);
        const dot = document.getElementById("signalDot");
        const signal = document.getElementById("signalText");
        if (dot) dot.className = "dot " + (opened ? "live" : "wait");
        if (signal) signal.textContent = opened ? COPY.live : COPY.waiting;
        const vis = document.getElementById("visionText");
        if (vis) {
          const name = data.vision || "local";
          vis.textContent = data.allow_cloud ? name + " · cloud" : name;
        }
        const ver = document.getElementById("versionText");
        if (ver) ver.textContent = data.display || data.version || "";
      })
      .catch(() => {});

    const rowsEl = document.getElementById("rows");
    let filter = "all";
    let list = [];
    let selected = null;

    function pagedLabel(event) {
      const key = event.paged_because || "";
      return event.paged_because_label || COPY.paged[key] || key || "";
    }

    function headline(event) {
      const boxes = event.boxes || [];
      if (boxes.length) {
        return boxes
          .map((b) => (b.track_id != null ? "#" + b.track_id + " " : "") + (b.cls || "object"))
          .join(", ");
      }
      if (event.summary) {
        const line = event.summary.split(/(?<=[.!?])\s/)[0];
        return line.length > 110 ? line.slice(0, 108) + "…" : line;
      }
      return (event.classes || []).join(", ") || "Motion";
    }

    function stopLabel(event) {
      if (event.operator_status) return event.operator_status;
      return event.stopped_at || "";
    }

    function whyText(event) {
      const why = event.why || {};
      const bits = [why.detail || why.reason || event.anomaly_reason || ""];
      if (why.occupancy_novelty != null) {
        bits.push(
          "Place " +
            Number(why.occupancy_novelty).toFixed(2) +
            " · look " +
            Number(why.visual_delta || 0).toFixed(2) +
            " · amount " +
            Number(why.motion_spike || 0).toFixed(2)
        );
      }
      return bits.filter(Boolean).join(" ");
    }

    function extraChips(event) {
      let html = "";
      if (event.paged_because === "audit") html += `<span class="chip audit">Audit</span>`;
      if (event.operator_status === "unverified") html += `<span class="chip">Unverified</span>`;
      if (event.provenance === "sample" || event.provenance === "fixture") {
        html += `<span class="chip">Sample</span>`;
      }
      return html;
    }

    function play(event) {
      selected = event;
      rowsEl.querySelectorAll(".log-item").forEach((btn) => btn.classList.remove("active"));
      const row = document.getElementById("evt-" + event.id);
      if (row) row.classList.add("active");
      const player = document.getElementById("player");
      if (event.clip_url) {
        player.src = event.clip_url;
        player.play().catch(() => {});
      } else {
        player.removeAttribute("src");
      }
      const marked = document.getElementById("marked");
      const spotEmpty = document.getElementById("spotEmpty");
      if (event.thumb_url) {
        marked.hidden = false;
        spotEmpty.hidden = true;
        marked.src = event.thumb_url + "?t=" + encodeURIComponent(event.ts_end || event.id);
      } else {
        marked.hidden = true;
        marked.removeAttribute("src");
        spotEmpty.hidden = false;
        spotEmpty.textContent = COPY.noMark;
      }
      const boxes = event.boxes || [];
      const chips = boxes.length
        ? boxes
        : (event.classes || []).map((cls) => ({
            cls,
            track_id: event.track_id,
            dwell_s: event.dwell_s,
            zone: event.zone,
            conf: event.score,
          }));
      const pagedEl = document.getElementById("pagedBecause");
      if (pagedEl) pagedEl.textContent = pagedLabel(event);
      document.getElementById("eventObjects").innerHTML = objectChips(chips) + extraChips(event);
      document.getElementById("whyLine").textContent = whyText(event);
      document.getElementById("detail").textContent =
        `${fmt(event.ts_start)} · track ${event.track_id ?? "—"} · dwell ${event.dwell_s ?? "—"}s · ${(event.classes || []).join(", ") || "—"}`;
      const isAudit = event.paged_because === "audit";
      document.getElementById("summary").textContent = [
        isAudit ? "" : event.anomaly ? COPY.alertPrefix + (event.anomaly_reason || "") : COPY.notAlert,
        event.summary || "",
        event.verifier_provider ? `Verifier: ${event.verifier_provider} (${event.verifier_status || "—"})` : "",
        event.novelty_score != null ? `Novelty ${Number(event.novelty_score).toFixed(2)} (ranking only).` : "",
      ]
        .filter(Boolean)
        .join(" ");
      renderCascade(document.getElementById("detailHandoff"), event.handoff || {}, () => {});
      const done = event.operator_status === "confirmed" || event.operator_status === "dismissed";
      document.getElementById("confirmBtn").disabled = done;
      document.getElementById("dismissBtn").disabled = done;
      document.getElementById("reviewMsg").textContent = done
        ? event.operator_status === "dismissed"
          ? COPY.reviewFolded
          : COPY.reviewKept
        : COPY.reviewHint;
    }

    function emptyCopy() {
      if (filter === "alerts") return COPY.emptyAlerts;
      if (filter === "unverified") return COPY.emptyUnverified;
      return COPY.emptyAll;
    }

    function render() {
      if (!list.length) {
        rowsEl.innerHTML = `<p class="empty">${emptyCopy()}</p>`;
        document.getElementById("detail").textContent = COPY.selectClip;
        document.getElementById("marked").hidden = true;
        document.getElementById("spotEmpty").hidden = false;
        document.getElementById("spotEmpty").textContent = COPY.selectClip;
        document.getElementById("eventObjects").innerHTML = "";
        document.getElementById("whyLine").textContent = "";
        const pagedEl = document.getElementById("pagedBecause");
        if (pagedEl) pagedEl.textContent = "";
        return;
      }
      rowsEl.innerHTML = list
        .map(
          (e) => `
        <button type="button" class="log-item ${e.operator_status === "unverified" ? "unverified" : e.anomaly ? "alert" : ""}" id="evt-${e.id}">
          ${e.thumb_url ? `<img class="thumb" src="${esc(e.thumb_url)}" alt="" />` : "<span></span>"}
          <div>
            <h3>${esc(headline(e))}${extraChips(e)}</h3>
            <p class="meta">${esc(pagedLabel(e))} · ${esc(fmt(e.ts_start))} · ${esc(stopLabel(e))}${e.track_id != null ? " · #" + e.track_id : ""}${e.dwell_s != null ? " · " + e.dwell_s + "s" : ""}</p>
          </div>
        </button>`
        )
        .join("");
      list.forEach((e) => document.getElementById("evt-" + e.id).addEventListener("click", () => play(e)));
      play(list[0]);
    }

    function setFilter(next) {
      filter = next;
      document.getElementById("allBtn").classList.toggle("on", filter === "all");
      document.getElementById("alertBtn").classList.toggle("on", filter === "alerts");
      const unverifiedBtn = document.getElementById("unverifiedBtn");
      if (unverifiedBtn) unverifiedBtn.classList.toggle("on", filter === "unverified");
      loadEvents();
    }

    async function loadEvents() {
      let q = "/api/events?limit=80";
      if (filter === "alerts") q += "&alerts=true";
      if (filter === "unverified") q += "&unverified=true";
      const res = await fetch(q);
      list = await res.json();
      render();
    }

    async function review(action) {
      if (!selected) return;
      const res = await fetch("/api/events/" + selected.id + "/review", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action }),
      });
      const data = await res.json().catch(() => ({}));
      const reviewMsg = document.getElementById("reviewMsg");
      if (res.status === 401) {
        reviewMsg.textContent = COPY.tokenReview;
        return;
      }
      if (!res.ok) {
        reviewMsg.textContent = data.detail || COPY.error;
        return;
      }
      list = list.map((e) => (e.id === data.id ? data : e));
      render();
    }

    document.getElementById("allBtn").addEventListener("click", () => setFilter("all"));
    document.getElementById("alertBtn").addEventListener("click", () => setFilter("alerts"));
    const unverifiedBtn = document.getElementById("unverifiedBtn");
    if (unverifiedBtn) unverifiedBtn.addEventListener("click", () => setFilter("unverified"));
    document.getElementById("confirmBtn").addEventListener("click", () => review("confirm"));
    document.getElementById("dismissBtn").addEventListener("click", () => review("dismiss"));
    loadEvents();
  }

  global.TundraUI = { COPY, esc, fmt, mountChrome, live, events };
})(window);
