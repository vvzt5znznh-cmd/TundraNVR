/* Tundra shared chrome, copy, Live + Events pages. */
(function (global) {
  const COPY = {
    brand: "Tundra",
    pages: { live: "Live", events: "Events" },
    seats: { edge: "Raspberry", node: "Node", hub: "Hub", operator: "Operator" },
    questions: {
      edge: "Unusual for this camera?",
      node: "What is it, and how long?",
      hub: "Alert, or suppress?",
      operator: "Incident, or normal?",
    },
    cascadeIdle: "Raspberry → Node → Hub → Operator",
    target: "Target",
    running: "This process",
    tripRan: "this trip",
    tripIdle: "idle",
    gap: "not the target",
    baselineReady: "Baseline ready",
    baselineLearn: (pct) => "Learning " + pct + "%",
    polReady: (n) => "Baseline for this camera is ready (" + n + " motion ticks).",
    polLearn: (n, need) =>
      "Learning this camera’s usual footprint — " + n + " / " + need + " motion ticks. Raspberry uploads until then.",
    setSource: "Set source",
    sourcePh: "RTSP URL, file path, or camera index (0)",
    switching: "Switching…",
    tokenCamera: "API token required to change the camera.",
    tokenReview: "API token required to review.",
    loadFail: "Could not load.",
    incident: "Incident",
    normal: "Normal",
    reviewHint: "Normal teaches Raspberry what usual looks like. Incident does not.",
    reviewKept: "Kept as an incident.",
    reviewFolded: "Saved as usual for this camera.",
    selectClip: "Select a clip.",
    emptyAll: "No events yet. Unusual motion on this camera will appear here.",
    emptyAlerts: "No alerts waiting.",
    notAlert: "Not an operator alert.",
    alertPrefix: "Alert: ",
    all: "All",
    alerts: "Alerts",
    operatorTitle: "Operator",
    details: "Details",
    live: "Live",
    waiting: "Waiting",
    error: "Error",
    connecting: "Connecting",
    quiet: "Quiet",
    unusual: "Unusual",
    usual: "Usual",
    quietSub: "Kept on Raspberry. No detector.",
    unusualSub: "Upload to Node.",
    usualSub: "Kept on Raspberry. No detector.",
    learnBit: (n, need) => " Learning " + n + "/" + need + " motion ticks.",
    nothingFrom: "Nothing from Raspberry",
    nothingSub: "Detector idle — usual motion never leaves Raspberry.",
    closedSub: "Closed here. Hub is not asked.",
    sendHub: "Sent to Hub to verify.",
    named: "Named",
    unsure: "Unnamed",
    hubIdle: "Idle",
    hubIdleSub: "Node did not escalate.",
    hubNeed: "Needs an operator.",
    hubLog: "Logged without paging.",
    describing: "Verifying…",
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
          <b>${esc(s.label)}</b><span>${esc(s.decision)}</span><em>${esc(s.detail || "")}</em>
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
    else if (["edge", "node", "hub"].includes(raw)) seat = raw;
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
      const pol = data.pol || {};
      const edge = (data.handoff || {}).edge || {};
      const need = pol.learn_samples || 40;
      const n = pol.samples || 0;
      const pct = Math.round((pol.progress || 0) * 100);
      const polPill = document.getElementById("polPill");
      if (pol.confident) {
        polPill.textContent = COPY.baselineReady;
        polPill.classList.remove("wait");
      } else {
        polPill.textContent = COPY.baselineLearn(pct);
        polPill.classList.add("wait");
      }
      document.getElementById("polLine").textContent = pol.confident
        ? COPY.polReady(n)
        : COPY.polLearn(n, need);
      document.getElementById("polBar").style.width = Math.min(100, pct) + "%";
      fillModelTable(data.models, seat);
      const tracks =
        (data.tracks || []).map((t) => "#" + t.id + " " + t.cls + " " + t.dwell_s + "s").join(", ") || "—";
      const escalate = data.escalation || {};
      const rows = [
        ["Baseline", (pol.confident ? COPY.baselineReady : COPY.baselineLearn(pct)) + " · " + n + " / " + need],
        ["Unusual score", (edge.score != null ? Number(edge.score).toFixed(2) : "—") + (edge.reason ? " · " + edge.reason : "")],
        ["Occupancy / look", "place " + Number(edge.occupancy_novelty || 0).toFixed(2) + " · visual " + Number(edge.visual_delta || 0).toFixed(2)],
        ["Tracks", tracks],
        [
          "Escalation",
          (escalate.mode || "recall") +
            " · R " +
            (escalate.raspberry_trips || 0) +
            " → N " +
            (escalate.node_proposals || 0) +
            " → Hub " +
            (escalate.hub_alerts || 0),
        ],
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
        status.textContent = opened ? COPY.live : COPY.waiting;
        const dot = document.getElementById("signalDot");
        const signal = document.getElementById("signalText");
        if (dot) {
          dot.className = "dot " + (opened ? "live" : "wait");
        }
        if (signal) signal.textContent = opened ? COPY.live : COPY.waiting;
        const vis = document.getElementById("visionText");
        if (vis) vis.textContent = (data.vision || "local") + (data.allow_cloud ? "" : " · local");
        const ver = document.getElementById("versionText");
        if (ver) ver.textContent = data.display || data.version || "";
        const handoff = data.handoff || {};
        const edge = handoff.edge || {};
        const node = handoff.node || {};
        const hub = handoff.hub || {};
        const pol = data.pol || {};
        const need = pol.learn_samples || 40;
        const n = pol.samples || 0;
        renderCascade(document.getElementById("handoff"), handoff, setSeat);
        fillDetails(data);
        document.getElementById("sceneKicker").textContent = COPY.questions[seat];
        const scene = document.getElementById("scene");
        const sub = document.getElementById("sceneSub");
        const objs = document.getElementById("objects");
        const card = document.getElementById("sceneCard");
        const models = data.models || {};
        const seatKey = seat === "raspberry" ? "edge" : seat;
        const line = document.getElementById("modelLine");
        line.textContent = modelLine(models[seatKey]);
        line.className = "models" + (models[seatKey] && !models[seatKey].match ? " gap" : "");
        card.classList.toggle("alert", Boolean(hub.page_operator && seat === "hub"));
        const learnBit = pol.confident ? "" : COPY.learnBit(n, need);
        if (seat === "edge") {
          if (!data.last_motion) {
            scene.textContent = COPY.quiet;
            sub.textContent = COPY.quietSub + learnBit;
          } else if (edge.upload) {
            scene.textContent = COPY.unusual;
            sub.textContent = (edge.reason || COPY.unusualSub) + learnBit;
          } else {
            scene.textContent = COPY.usual;
            sub.textContent = COPY.usualSub + learnBit;
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
            sub.textContent = COPY.sendHub;
          }
          objs.innerHTML = dets
            .map(
              (d) =>
                `<span>${d.track_id != null ? "#" + d.track_id + " " : ""}${esc(d.cls)}${d.dwell_s != null ? " " + d.dwell_s + "s" : ""}</span>`
            )
            .join("");
        } else if (!hub.ran) {
          scene.textContent = COPY.hubIdle;
          sub.textContent = COPY.hubIdleSub;
          objs.innerHTML = "";
        } else {
          scene.textContent = data.last_scene || hub.detail || COPY.describing;
          sub.textContent = hub.page_operator ? COPY.hubNeed : COPY.hubLog;
          objs.innerHTML = (node.classes || []).map((d) => `<span>${esc(d.cls)}</span>`).join("");
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
        if (vis) vis.textContent = (data.vision || "local") + (data.allow_cloud ? "" : " · local");
        const ver = document.getElementById("versionText");
        if (ver) ver.textContent = data.display || data.version || "";
      })
      .catch(() => {});

    const rowsEl = document.getElementById("rows");
    let alertsOnly = false;
    let list = [];
    let selected = null;

    function headline(event) {
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
      document.getElementById("detail").textContent =
        `${fmt(event.ts_start)} · track ${event.track_id ?? "—"} · dwell ${event.dwell_s ?? "—"}s · ${(event.classes || []).join(", ") || "—"}`;
      document.getElementById("summary").textContent = [
        event.anomaly ? COPY.alertPrefix + (event.anomaly_reason || "") : COPY.notAlert,
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

    function render() {
      if (!list.length) {
        rowsEl.innerHTML = `<p class="empty">${alertsOnly ? COPY.emptyAlerts : COPY.emptyAll}</p>`;
        document.getElementById("detail").textContent = COPY.selectClip;
        return;
      }
      rowsEl.innerHTML = list
        .map(
          (e) => `
        <button type="button" class="log-item ${e.anomaly ? "alert" : ""}" id="evt-${e.id}">
          ${e.thumb_url ? `<img class="thumb" src="${esc(e.thumb_url)}" alt="" />` : "<span></span>"}
          <div>
            <h3>${esc(headline(e))}</h3>
            <p class="meta">${esc(fmt(e.ts_start))} · ${esc(stopLabel(e))}${e.track_id != null ? " · #" + e.track_id : ""}${e.dwell_s != null ? " · " + e.dwell_s + "s" : ""}</p>
          </div>
        </button>`
        )
        .join("");
      list.forEach((e) => document.getElementById("evt-" + e.id).addEventListener("click", () => play(e)));
      play(list[0]);
    }

    async function loadEvents() {
      const res = await fetch("/api/events?limit=80" + (alertsOnly ? "&alerts=true" : ""));
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

    document.getElementById("allBtn").addEventListener("click", () => {
      alertsOnly = false;
      document.getElementById("allBtn").classList.add("on");
      document.getElementById("alertBtn").classList.remove("on");
      loadEvents();
    });
    document.getElementById("alertBtn").addEventListener("click", () => {
      alertsOnly = true;
      document.getElementById("alertBtn").classList.add("on");
      document.getElementById("allBtn").classList.remove("on");
      loadEvents();
    });
    document.getElementById("confirmBtn").addEventListener("click", () => review("confirm"));
    document.getElementById("dismissBtn").addEventListener("click", () => review("dismiss"));
    loadEvents();
  }

  global.TundraUI = { COPY, esc, fmt, mountChrome, live, events };
})(window);
