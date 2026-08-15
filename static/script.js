(function () {
  "use strict";

  var EMOTIONS = [
    { key: "sadness",  label: "Sadness",  emoji: "\uD83D\uDE22", hex: "#5b7fdb" },
    { key: "joy",      label: "Joy",      emoji: "\uD83D\uDE04", hex: "#f4c24e" },
    { key: "love",     label: "Love",     emoji: "\u2764\uFE0F", hex: "#ef6c9b" },
    { key: "anger",    label: "Anger",    emoji: "\uD83D\uDE20", hex: "#e3512b" },
    { key: "fear",     label: "Fear",     emoji: "\uD83D\uDE28", hex: "#8b6fc7" },
    { key: "surprise", label: "Surprise", emoji: "\uD83D\uDE32", hex: "#38c6c6" }
  ];

  var EXAMPLES = [
    "I just found out I got the job \u2014 I still can't believe it.",
    "I miss the way things used to be before everything changed.",
    "Watching you grow into who you are has been the best part of my life.",
    "I can't believe you went behind my back after everything.",
    "The footsteps outside got louder, then stopped right at my door."
  ];

  var textInput = document.getElementById("textInput");
  var charCount = document.getElementById("charCount");
  var analyzeBtn = document.getElementById("analyzeBtn");
  var btnLabel = analyzeBtn.querySelector(".btn-label");
  var consoleBox = document.getElementById("consoleBox");
  var examplesEl = document.getElementById("examples");
  var errorMsg = document.getElementById("errorMsg");
  var results = document.getElementById("results");
  var dominantEmoji = document.getElementById("dominantEmoji");
  var dominantLabel = document.getElementById("dominantLabel");
  var confidenceValue = document.getElementById("confidenceValue");
  var spectrumEl = document.getElementById("spectrum");
  var analyzedTextEl = document.getElementById("analyzedText");
  var statusDot = document.getElementById("statusDot");
  var statusText = document.getElementById("statusText");

  var reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  var isLoading = false;

  /* ---------- example chips ---------- */

  EXAMPLES.forEach(function (text) {
    var chip = document.createElement("button");
    chip.type = "button";
    chip.className = "chip";
    var shortText = text.length > 44 ? text.slice(0, 44).trim() + "\u2026" : text;
    chip.textContent = shortText;
    chip.setAttribute("aria-label", "Fill example: " + text);
    chip.addEventListener("click", function () {
      textInput.value = text;
      updateCharCount();
      textInput.focus();
    });
    examplesEl.appendChild(chip);
  });

  /* ---------- spectrum rows (built once, updated on each run) ---------- */

  var rowEls = {};
  EMOTIONS.forEach(function (e) {
    var row = document.createElement("div");
    row.className = "spectrum-row";
    row.setAttribute("data-emotion", e.key);

    var label = document.createElement("span");
    label.className = "spectrum-label";
    label.textContent = e.label;

    var track = document.createElement("span");
    track.className = "spectrum-track";
    var fill = document.createElement("span");
    fill.className = "spectrum-fill";
    fill.style.setProperty("--fill-color", e.hex);
    track.appendChild(fill);

    var value = document.createElement("span");
    value.className = "spectrum-value";
    value.textContent = "0%";

    row.appendChild(label);
    row.appendChild(track);
    row.appendChild(value);
    spectrumEl.appendChild(row);

    rowEls[e.key] = { row: row, fill: fill, value: value };
  });

  /* ---------- input handling ---------- */

  function updateCharCount() {
    charCount.textContent = String(textInput.value.length);
  }
  textInput.addEventListener("input", updateCharCount);

  textInput.addEventListener("keydown", function (ev) {
    if ((ev.ctrlKey || ev.metaKey) && ev.key === "Enter") {
      ev.preventDefault();
      runAnalysis();
    }
  });

  analyzeBtn.addEventListener("click", runAnalysis);

  /* ---------- analysis flow ---------- */

  function runAnalysis() {
    if (isLoading) return;
    var text = textInput.value.trim();

    hideError();

    if (!text) {
      showError("EMPTY INPUT \u2014 type a sentence first.");
      textInput.focus();
      return;
    }

    setLoading(true);

    fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: text })
    })
      .then(function (res) {
        if (res.status === 503) {
          throw { kind: "warming" };
        }
        if (!res.ok) {
          throw { kind: "http", status: res.status };
        }
        return res.json();
      })
      .then(function (data) {
        renderResults(data);
      })
      .catch(function (err) {
        if (err && err.kind === "warming") {
          showError("MODEL WARMING UP \u2014 give it a moment and try again.");
        } else {
          showError("SIGNAL LOST \u2014 couldn't reach the model. Check the connection and try again.");
        }
      })
      .finally(function () {
        setLoading(false);
      });
  }

  function setLoading(loading) {
    isLoading = loading;
    analyzeBtn.disabled = loading;
    analyzeBtn.classList.toggle("loading", loading);
    consoleBox.classList.toggle("scanning", loading);
    btnLabel.textContent = loading ? "Reading Signal\u2026" : "Analyze Text";
  }

  function showError(msg) {
    errorMsg.textContent = msg;
    errorMsg.hidden = false;
  }
  function hideError() {
    errorMsg.hidden = true;
  }

  function renderResults(data) {
    var probs = data.all_probabilites || {};
    var dominant = EMOTIONS.filter(function (e) { return e.key === data.predicted_emotion; })[0] || EMOTIONS[0];

    results.hidden = false;
    requestAnimationFrame(function () { results.classList.add("visible"); });

    dominantEmoji.textContent = dominant.emoji;
    dominantLabel.textContent = dominant.label.toUpperCase();
    results.style.setProperty("--dominant-color", dominant.hex);

    animateCount(confidenceValue, (data.confidence || 0) * 100, reduceMotion ? 0 : 800);

    analyzedTextEl.textContent = "\u201C" + data.text + "\u201D";

    EMOTIONS.forEach(function (e, i) {
      var pct = (probs[e.key] || 0) * 100;
      var refs = rowEls[e.key];
      refs.row.classList.toggle("is-dominant", e.key === dominant.key);
      var delay = reduceMotion ? 0 : i * 80;
      window.setTimeout(function () {
        refs.fill.style.width = pct.toFixed(1) + "%";
      }, delay);
      refs.value.textContent = pct.toFixed(1) + "%";
    });
  }

  function animateCount(el, target, duration) {
    if (!duration) {
      el.textContent = target.toFixed(1);
      return;
    }
    var start = performance.now();
    function tick(now) {
      var t = Math.min(1, (now - start) / duration);
      var eased = 1 - Math.pow(1 - t, 3);
      el.textContent = (target * eased).toFixed(1);
      if (t < 1) requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  }

  /* ---------- model status ---------- */

  fetch("/health")
    .then(function (r) { return r.json(); })
    .then(function (d) {
      var online = !!d.model_loaded;
      statusDot.classList.add(online ? "online" : "offline");
      statusText.textContent = online ? "ONLINE" : "LOADING";
    })
    .catch(function () {
      statusDot.classList.add("offline");
      statusText.textContent = "OFFLINE";
    });

  /* ---------- ambient waveform (hero backdrop) ---------- */

  var canvas = document.getElementById("waveCanvas");
  if (canvas && canvas.getContext) {
    var ctx = canvas.getContext("2d");
    var colors = EMOTIONS.map(function (e) { return e.hex; });
    var w = 0, h = 0, dpr = 1;

    function resize() {
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      w = canvas.clientWidth;
      h = canvas.clientHeight;
      canvas.width = Math.max(1, w * dpr);
      canvas.height = Math.max(1, h * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }
    window.addEventListener("resize", resize);
    resize();

    function drawFrame(t) {
      ctx.clearRect(0, 0, w, h);
      colors.forEach(function (color, i) {
        var amp = h * 0.055;
        var freq = 0.007 + i * 0.0014;
        var phase = t * 0.00035 + i * 1.3;
        var yBase = h * (0.28 + i * 0.1);
        ctx.beginPath();
        for (var x = 0; x <= w; x += 6) {
          var y = yBase + Math.sin(x * freq + phase) * amp * (0.6 + 0.4 * Math.sin(phase * 0.5));
          if (x === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.strokeStyle = color;
        ctx.globalAlpha = 0.16;
        ctx.lineWidth = 1.5;
        ctx.stroke();
      });
      ctx.globalAlpha = 1;
    }

    if (reduceMotion) {
      drawFrame(0);
    } else {
      (function loop(t) {
        drawFrame(t);
        requestAnimationFrame(loop);
      })(0);
    }
  }
})();
