import { useState, useEffect, useRef, useMemo } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  ResponsiveContainer, ComposedChart, Area, Tooltip
} from "recharts";

// =============================================================================
// CONFIG — adjust base path depending on your setup
// =============================================================================
const DATA_BASE = "/data"; // points to public/data/
// Prefer webp for max-activating images (smaller size); fallback to png for legacy data
const IMG_EXT = "webp";
const IMG_EXT_FALLBACK = "png";

// =============================================================================
// DATA LOADING HOOKS
// =============================================================================

function useExperiments() {
  const [experiments, setExperiments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch(`${DATA_BASE}/experiments.json`)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status} — is public/data/ generated?`);
        return r.json();
      })
      .then(setExperiments)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  return { experiments, loading, error };
}

function useExperimentData(expId) {
  const [meta, setMeta] = useState(null);
  const [evolTraj, setEvolTraj] = useState([]);
  const [loading, setLoading] = useState(false);
  const cache = useRef({});

  useEffect(() => {
    if (!expId) return;
    setMeta(null);
    setEvolTraj([]);
    setLoading(true);
    let cancelled = false;

    Promise.all([
      fetch(`${DATA_BASE}/${expId}/meta.json`).then((r) => r.json()),
      fetch(`${DATA_BASE}/${expId}/evol_traj.json`).then((r) => r.json()),
    ])
      .then(([m, traj]) => {
        if (cancelled) return;
        setMeta(m);
        setEvolTraj(traj);
      })
      .catch((e) => { if (!cancelled) console.error("Failed to load experiment:", e); })
      .finally(() => { if (!cancelled) setLoading(false); });

    return () => { cancelled = true; };
  }, [expId]);

  return { meta, evolTraj, loading, cache };
}

function useGenerationData(expId, genNum, cache) {
  const [genData, setGenData] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!expId || !genNum) return;

    const cacheKey = `${expId}-${genNum}`;
    if (cache.current[cacheKey]) {
      setGenData(cache.current[cacheKey]);
      setLoading(false);
      return;
    }

    setGenData(null);
    setLoading(true);
    let cancelled = false;
    const genDir = `${DATA_BASE}/${expId}/gen_${String(genNum).padStart(2, "0")}`;

    Promise.all([
      fetch(`${genDir}/deepsim_psth.json`).then((r) => r.json()),
      fetch(`${genDir}/biggan_psth.json`).then((r) => r.json()),
    ])
      .then(([dsPsth, bgPsth]) => {
        if (cancelled) return;
        const data = {
          deepsim: { psth: dsPsth, imageSrc: `${genDir}/deepsim_img.${IMG_EXT}`, imageSrcFallback: `${genDir}/deepsim_img.${IMG_EXT_FALLBACK}` },
          biggan: { psth: bgPsth, imageSrc: `${genDir}/biggan_img.${IMG_EXT}`, imageSrcFallback: `${genDir}/biggan_img.${IMG_EXT_FALLBACK}` },
        };
        cache.current[cacheKey] = data;
        setGenData(data);
      })
      .catch((e) => { if (!cancelled) console.error("Failed to load generation:", e); })
      .finally(() => { if (!cancelled) setLoading(false); });

    return () => { cancelled = true; };
  }, [expId, genNum, cache]);

  return { genData, loading };
}

// Preload next N generations in background
function usePreloader(expId, currentGen, maxGen, cache) {
  useEffect(() => {
    if (!expId) return;
    const AHEAD = 3;
    for (let g = currentGen + 1; g <= Math.min(currentGen + AHEAD, maxGen); g++) {
      const key = `${expId}-${g}`;
      if (cache.current[key]) continue;
      const genDir = `${DATA_BASE}/${expId}/gen_${String(g).padStart(2, "0")}`;
      Promise.all([
        fetch(`${genDir}/deepsim_psth.json`).then((r) => r.json()),
        fetch(`${genDir}/biggan_psth.json`).then((r) => r.json()),
      ]).then(([ds, bg]) => {
        cache.current[key] = {
          deepsim: { psth: ds, imageSrc: `${genDir}/deepsim_img.${IMG_EXT}`, imageSrcFallback: `${genDir}/deepsim_img.${IMG_EXT_FALLBACK}` },
          biggan: { psth: bg, imageSrc: `${genDir}/biggan_img.${IMG_EXT}`, imageSrcFallback: `${genDir}/biggan_img.${IMG_EXT_FALLBACK}` },
        };
      }).catch(() => {});
      // Preload images into browser cache (prefer webp)
      new Image().src = `${genDir}/deepsim_img.${IMG_EXT}`;
      new Image().src = `${genDir}/biggan_img.${IMG_EXT}`;
    }
  }, [expId, currentGen, maxGen, cache]);
}

// Load PSTH mean_rate for all generations (for "all generations as gray" overlay)
function useAllGenerationsPsth(expId, maxGen, enabled) {
  const [allGenPsth, setAllGenPsth] = useState({ deepsim: [], biggan: [], time_ms: [], loading: false });
  useEffect(() => {
    if (!expId || !maxGen || !enabled) {
      setAllGenPsth({ deepsim: [], biggan: [], time_ms: [], loading: false });
      return;
    }
    setAllGenPsth((prev) => ({ ...prev, loading: true }));
    let cancelled = false;
    const promises = [];
    for (let g = 1; g <= maxGen; g++) {
      const genDir = `${DATA_BASE}/${expId}/gen_${String(g).padStart(2, "0")}`;
      promises.push(
        Promise.all([
          fetch(`${genDir}/deepsim_psth.json`).then((r) => r.json()).catch(() => null),
          fetch(`${genDir}/biggan_psth.json`).then((r) => r.json()).catch(() => null),
        ]).then(([ds, bg]) => ({ g, deepsim: ds?.mean_rate ?? null, biggan: bg?.mean_rate ?? null, time_ms: ds?.time_ms ?? bg?.time_ms ?? [] }))
      );
    }
    Promise.all(promises).then((results) => {
      if (cancelled) return;
      const deepsim = results.map((r) => r.deepsim);
      const biggan = results.map((r) => r.biggan);
      const time_ms = results[0]?.time_ms ?? [];
      setAllGenPsth({ deepsim, biggan, time_ms, loading: false });
    }).catch(() => { if (!cancelled) setAllGenPsth((prev) => ({ ...prev, loading: false })); });
    return () => { cancelled = true; };
  }, [expId, maxGen, enabled]);
  return allGenPsth;
}

// =============================================================================
// COMPONENTS
// =============================================================================

const MONO = "'JetBrains Mono', 'Fira Code', 'Consolas', monospace";
const DISPLAY = "'Space Grotesk', 'IBM Plex Sans', system-ui, sans-serif";
const BODY = "'IBM Plex Sans', system-ui, sans-serif";

function SectionLabel({ children }) {
  return (
    <div style={{
      fontSize: 10, fontFamily: MONO,
      letterSpacing: "0.08em", textTransform: "uppercase",
      color: "#555", marginBottom: 6,
    }}>
      {children}
    </div>
  );
}

function MethodImage({ src, rate, gen, fallbackSrc }) {
  const [imgError, setImgError] = useState(false);
  const [triedFallback, setTriedFallback] = useState(false);
  const [currentSrc, setCurrentSrc] = useState(src);
  useEffect(() => {
    setImgError(false);
    setTriedFallback(false);
    setCurrentSrc(src);
  }, [src]);

  const handleError = () => {
    if (fallbackSrc && !triedFallback) {
      setTriedFallback(true);
      setCurrentSrc(fallbackSrc);
    } else {
      setImgError(true);
    }
  };

  const displaySrc = currentSrc || src;

  return (
    <div>
      <SectionLabel>Representative Image</SectionLabel>
      <div style={{
        position: "relative", aspectRatio: "1",
        background: "#0a0c14", borderRadius: 6, overflow: "hidden",
      }}>
        {displaySrc && !imgError ? (
          <img
            src={displaySrc}
            alt={`Gen ${gen}`}
            onError={handleError}
            style={{ width: "100%", height: "100%", objectFit: "cover", display: "block" }}
          />
        ) : (
          <div style={{
            width: "100%", height: "100%", display: "flex",
            alignItems: "center", justifyContent: "center",
            color: "#333", fontSize: 12, fontFamily: MONO,
          }}>
            No image
          </div>
        )}
        {rate != null && (
          <div style={{
            position: "absolute", bottom: 6, left: 6, right: 6,
            background: "rgba(0,0,0,0.75)", borderRadius: 4,
            padding: "3px 8px", fontSize: 11,
            fontFamily: MONO, color: "#ccc",
          }}>
            Gen{gen} — {rate} Hz
          </div>
        )}
      </div>
    </div>
  );
}


function collectPsthValues(psth, showTrials, allGenRates) {
  const vals = [];
  if (!psth) return vals;
  if (psth.mean_rate) vals.push(...psth.mean_rate);
  if (showTrials && psth.trial_rates?.length) {
    psth.trial_rates.forEach((row) => { if (row) vals.push(...row); });
  }
  if (allGenRates?.length) {
    allGenRates.forEach((arr) => { if (arr) vals.push(...arr); });
  }
  return vals;
}

function PSTHChart({ psth, gen, label, color, showIndividualTrials = true, allGenerationsMeanRates = null, yDomain = null }) {
  if (!psth) return null;

  const chartData = psth.time_ms.map((t, i) => {
    const d = { time: t, mean: psth.mean_rate[i] };
    if (showIndividualTrials && psth.trial_rates?.length) {
      const maxTrials = Math.min(psth.trial_rates.length, 20);
      for (let ti = 0; ti < maxTrials; ti++) {
        d[`t${ti}`] = psth.trial_rates[ti]?.[i] ?? 0;
      }
    }
    if (allGenerationsMeanRates?.length) {
      allGenerationsMeanRates.forEach((rateArr, gi) => {
        if (rateArr && rateArr[i] != null) d[`gen${gi}`] = rateArr[i];
      });
    }
    return d;
  });

  const numTrials = showIndividualTrials && psth.trial_rates?.length ? Math.min(psth.trial_rates.length, 20) : 0;
  const numGenTraces = allGenerationsMeanRates?.length ?? 0;

  return (
    <div>
      <SectionLabel>
        Evoked PSTH · Gen{gen} · {label} · {psth.evoked_rate} Hz · n={psth.n_trials}
      </SectionLabel>
      <ResponsiveContainer width="100%" height={180}>
        <LineChart data={chartData} margin={{ top: 4, right: 8, bottom: 20, left: 4 }}>
          <CartesianGrid stroke="#1a1d2e" strokeDasharray="3 3" />
          <XAxis
            dataKey="time" stroke="#444" tick={{ fontSize: 10, fill: "#555" }}
            label={{ value: "time (ms)", position: "insideBottom", offset: -12, fontSize: 10, fill: "#555" }}
          />
          <YAxis
            stroke="#444" tick={{ fontSize: 10, fill: "#555" }}
            domain={yDomain ?? ["auto", "auto"]}
            label={{ value: "PSTH (Hz)", angle: -90, position: "insideLeft", offset: 10, fontSize: 10, fill: "#555" }}
          />
          {numGenTraces > 0 && Array.from({ length: numGenTraces }, (_, i) => (
            <Line key={`gen${i}`} dataKey={`gen${i}`} stroke="#3a3d50" strokeWidth={0.8}
              dot={false} isAnimationActive={false} />
          ))}
          {Array.from({ length: numTrials }, (_, i) => (
            <Line key={`t${i}`} dataKey={`t${i}`} stroke="#2a2d40" strokeWidth={0.7}
              dot={false} isAnimationActive={false} />
          ))}
          <Line dataKey="mean" stroke={color} strokeWidth={2.5} dot={false} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}


function EvolTrajChart({ evolTraj, currentGen, totalGens }) {
  if (!evolTraj || evolTraj.length === 0) return null;

  // Build chart data with confidence band as [low, high] range; include ref (reference image) when present
  const hasRef = evolTraj.some((pt) => pt.ref_deepsim != null || pt.ref_biggan != null);
  const chartData = evolTraj.slice(0, currentGen).map((pt) => {
    const out = {
      gen: pt.gen,
      deepsim: pt.deepsim,
      biggan: pt.biggan,
      dsBand: [pt.deepsim_low, pt.deepsim_high],
      bgBand: [pt.biggan_low, pt.biggan_high],
    };
    if (hasRef) {
      out.ref_deepsim = pt.ref_deepsim ?? null;
      out.ref_biggan = pt.ref_biggan ?? null;
      out.ref_dsBand = pt.ref_deepsim_low != null ? [pt.ref_deepsim_low, pt.ref_deepsim_high] : null;
      out.ref_bgBand = pt.ref_biggan_low != null ? [pt.ref_biggan_low, pt.ref_biggan_high] : null;
    }
    return out;
  });

  return (
    <div>
      <SectionLabel>CMAES Evolution Trajectory</SectionLabel>
      <ResponsiveContainer width="100%" height={180}>
        <ComposedChart data={chartData} margin={{ top: 4, right: 8, bottom: 20, left: 4 }}>
          <CartesianGrid stroke="#1a1d2e" strokeDasharray="3 3" />
          <XAxis
            dataKey="gen" stroke="#444" tick={{ fontSize: 10, fill: "#555" }}
            domain={[1, totalGens || 20]} type="number"
            label={{ value: "Generations", position: "insideBottom", offset: -12, fontSize: 10, fill: "#555" }}
          />
          <YAxis
            stroke="#444" tick={{ fontSize: 10, fill: "#555" }}
            label={{ value: "Response fr (Hz)", angle: -90, position: "insideLeft", offset: 10, fontSize: 10, fill: "#555" }}
          />
          <Tooltip
            contentStyle={{
              background: "#141620", border: "1px solid #2a2d40",
              borderRadius: 6, fontSize: 11, fontFamily: MONO,
            }}
            labelStyle={{ color: "#888" }}
            labelFormatter={(v) => `Gen ${v}`}
            formatter={(value, name) => {
              if (name === "deepsim") return [value, "DeepSim (Hz)"];
              if (name === "biggan") return [value, "BigGAN (Hz)"];
              if (name === "ref_deepsim") return [value, "Ref DeepSim (Hz)"];
              if (name === "ref_biggan") return [value, "Ref BigGAN (Hz)"];
              return [value, name];
            }}
          />
          {/* Confidence bands */}
          <Area dataKey="dsBand" fill="rgba(96,165,250,0.12)" stroke="none" isAnimationActive={false} />
          <Area dataKey="bgBand" fill="rgba(248,113,113,0.12)" stroke="none" isAnimationActive={false} />
          {/* Reference image bands (when present) */}
          {hasRef && (
            <>
              <Area dataKey="ref_dsBand" fill="rgba(96,165,250,0.06)" stroke="none" isAnimationActive={false} />
              <Area dataKey="ref_bgBand" fill="rgba(248,113,113,0.06)" stroke="none" isAnimationActive={false} />
            </>
          )}
          {/* Mean lines */}
          <Line dataKey="deepsim" stroke="#60a5fa" strokeWidth={2} dot={false} isAnimationActive={false} />
          <Line dataKey="biggan" stroke="#f87171" strokeWidth={2} dot={false} isAnimationActive={false} />
          {/* Reference image lines (dashed) */}
          {hasRef && (
            <>
              <Line dataKey="ref_deepsim" stroke="#60a5fa" strokeWidth={1.5} strokeDasharray="4 4" dot={false} isAnimationActive={false} />
              <Line dataKey="ref_biggan" stroke="#f87171" strokeWidth={1.5} strokeDasharray="4 4" dot={false} isAnimationActive={false} />
            </>
          )}
        </ComposedChart>
      </ResponsiveContainer>
      <div style={{ display: "flex", gap: 16, justifyContent: "center", marginTop: -4, flexWrap: "wrap" }}>
        <span style={{ fontSize: 11, color: "#60a5fa", fontFamily: MONO }}>● DeepSim</span>
        <span style={{ fontSize: 11, color: "#f87171", fontFamily: MONO }}>● BigGAN</span>
        {hasRef && <span style={{ fontSize: 11, color: "#888", fontFamily: MONO }}>— — Reference images</span>}
      </div>
    </div>
  );
}


function PlaybackControls({
  currentGen, maxGen, isPlaying, playSpeed,
  onTogglePlay, onSetGen, onSetSpeed, onReset,
}) {
  const btnBase = {
    width: 32, height: 32, borderRadius: 6,
    border: "1px solid #1e2030", background: "transparent",
    cursor: "pointer", color: "#777", fontSize: 13,
    display: "flex", alignItems: "center", justifyContent: "center",
    transition: "all 0.15s",
  };

  return (
    <div style={{
      background: "#10121c", borderRadius: 10, border: "1px solid #1a1d2e",
      padding: "14px 20px", display: "flex", alignItems: "center", gap: 14, flexWrap: "wrap",
    }}>
      {/* Play / Pause */}
      <button onClick={onTogglePlay} style={{
        ...btnBase, width: 42, height: 42, borderRadius: "50%",
        background: isPlaying ? "#1e2030" : "rgba(96,165,250,0.08)",
        borderColor: isPlaying ? "rgba(248,113,113,0.3)" : "rgba(96,165,250,0.3)",
        color: isPlaying ? "#f87171" : "#60a5fa", fontSize: 18,
      }}>
        {isPlaying ? "⏸" : "▶"}
      </button>

      {/* Step back */}
      <button onClick={() => onSetGen(Math.max(1, currentGen - 1))} style={btnBase}>◀</button>

      {/* Slider */}
      <div style={{ flex: 1, display: "flex", alignItems: "center", gap: 10, minWidth: 200 }}>
        <span style={{ fontFamily: MONO, fontSize: 12, color: "#aaa", minWidth: 54, fontWeight: 600 }}>
          Gen {currentGen}
        </span>
        <input
          type="range" min={1} max={maxGen} value={currentGen}
          onChange={(e) => onSetGen(Number(e.target.value))}
          style={{ flex: 1, accentColor: "#60a5fa" }}
        />
        <span style={{ fontFamily: MONO, fontSize: 11, color: "#444" }}>/ {maxGen}</span>
      </div>

      {/* Step forward */}
      <button onClick={() => onSetGen(Math.min(maxGen, currentGen + 1))} style={btnBase}>▶</button>

      <div style={{ width: 1, height: 24, background: "#1e2030" }} />

      {/* Speed selector */}
      <div style={{ display: "flex", gap: 4, alignItems: "center" }}>
        <span style={{ fontFamily: MONO, fontSize: 10, color: "#555", marginRight: 4 }}>SPEED</span>
        {[
          { label: "0.5×", val: 1000 },
          { label: "1×", val: 500 },
          { label: "2×", val: 250 },
          { label: "4×", val: 125 },
        ].map((s) => (
          <button key={s.val} onClick={() => onSetSpeed(s.val)} style={{
            fontFamily: MONO, padding: "4px 8px", borderRadius: 4, fontSize: 11, cursor: "pointer",
            border: `1px solid ${playSpeed === s.val ? "rgba(96,165,250,0.3)" : "#1e2030"}`,
            background: playSpeed === s.val ? "rgba(96,165,250,0.08)" : "transparent",
            color: playSpeed === s.val ? "#60a5fa" : "#666",
            transition: "all 0.15s",
          }}>
            {s.label}
          </button>
        ))}
      </div>

      {/* Reset */}
      <button onClick={onReset} style={{
        ...btnBase, width: "auto", padding: "6px 14px",
        fontFamily: MONO, fontSize: 11, letterSpacing: "0.05em",
      }}>
        RESET
      </button>
    </div>
  );
}


// =============================================================================
// MAIN APP
// =============================================================================

export default function App() {
  const { experiments, loading: loadingList, error } = useExperiments();
  const [selectedExp, setSelectedExp] = useState(null);
  const [currentGen, setCurrentGen] = useState(1);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playSpeed, setPlaySpeed] = useState(500);
  const [showIndividualTrials, setShowIndividualTrials] = useState(false);
  const [showAllGenerationsPsth, setShowAllGenerationsPsth] = useState(true);
  const [tiePsthYLim, setTiePsthYLim] = useState(true);
  const [filterAnimal, setFilterAnimal] = useState("");
  const [filterArea, setFilterArea] = useState("");
  const [filterChannel, setFilterChannel] = useState("");
  const intervalRef = useRef(null);
  const lastExpIdForGen = useRef(null);

  // Extract Chxx from unit string (e.g. "Ch12U3" -> "Ch12")
  const getChannelFromUnit = (unit) => {
    if (!unit) return "";
    const m = String(unit).match(/Ch\d+/i);
    return m ? m[0] : "";
  };

  // Unique filter options from experiments (channel = Chxx only, not U)
  const filterOptions = (() => {
    const animals = [...new Set(experiments.map((e) => e.animal).filter(Boolean))].sort();
    const areas = [...new Set(experiments.map((e) => e.area).filter(Boolean))].sort();
    const channelSet = new Set(experiments.map((e) => getChannelFromUnit(e.unit)).filter(Boolean));
    const channels = [...channelSet].sort((a, b) => {
      const na = parseInt(a.replace(/\D/g, ""), 10) || 0;
      const nb = parseInt(b.replace(/\D/g, ""), 10) || 0;
      return na - nb || a.localeCompare(b);
    });
    return { animals, areas, channels };
  })();

  const filteredExperiments = experiments.filter((exp) => {
    if (filterAnimal && exp.animal !== filterAnimal) return false;
    if (filterArea && exp.area !== filterArea) return false;
    if (filterChannel && getChannelFromUnit(exp.unit) !== filterChannel) return false;
    return true;
  });

  // Auto-select first experiment when list loads; keep selection valid when filters change
  useEffect(() => {
    if (experiments.length > 0 && !selectedExp) {
      setSelectedExp(experiments[0].id);
    }
  }, [experiments, selectedExp]);

  useEffect(() => {
    const stillVisible = filteredExperiments.some((e) => e.id === selectedExp);
    if (filteredExperiments.length > 0 && !stillVisible) {
      setSelectedExp(filteredExperiments[0].id);
    }
  }, [filteredExperiments, selectedExp]);

  const { meta, evolTraj, loading: loadingExp, cache } = useExperimentData(selectedExp);
  const maxGen = meta?.num_generations || 20;
  const { genData, loading: loadingGen } = useGenerationData(selectedExp, currentGen, cache);
  const allGenPsth = useAllGenerationsPsth(selectedExp, maxGen, showAllGenerationsPsth);

  // When switching to an experiment, show last generation by default
  useEffect(() => {
    if (selectedExp && meta && meta.id === selectedExp && lastExpIdForGen.current !== selectedExp) {
      lastExpIdForGen.current = selectedExp;
      setCurrentGen(meta.num_generations ?? 1);
    }
  }, [selectedExp, meta]);

  // Shared Y domain for both PSTH charts when tiePsthYLim is on
  const psthYDomain = useMemo(() => {
    if (!tiePsthYLim) return null;
    const dsVals = collectPsthValues(genData?.deepsim?.psth, showIndividualTrials, showAllGenerationsPsth ? allGenPsth.deepsim : null);
    const bgVals = collectPsthValues(genData?.biggan?.psth, showIndividualTrials, showAllGenerationsPsth ? allGenPsth.biggan : null);
    const allVals = [...dsVals, ...bgVals].filter((v) => typeof v === "number" && !Number.isNaN(v));
    if (allVals.length === 0) return null;
    const lo = Math.min(0, ...allVals);
    const hi = Math.max(...allVals);
    const pad = Math.max(10, (hi - lo) * 0.05);
    return [Math.floor(lo - pad), Math.ceil(hi + pad)];
  }, [tiePsthYLim, genData?.deepsim?.psth, genData?.biggan?.psth, showIndividualTrials, showAllGenerationsPsth, allGenPsth.deepsim, allGenPsth.biggan]);

  // Preload upcoming generations for smooth playback
  usePreloader(selectedExp, currentGen, maxGen, cache);

  // Playback animation loop
  useEffect(() => {
    if (isPlaying) {
      intervalRef.current = setInterval(() => {
        setCurrentGen((prev) => {
          if (prev >= maxGen) {
            setIsPlaying(false);
            return prev;
          }
          return prev + 1;
        });
      }, playSpeed);
    }
    return () => clearInterval(intervalRef.current);
  }, [isPlaying, playSpeed, maxGen]);

  const handleExpChange = (id) => {
    setSelectedExp(id);
    setIsPlaying(false);
    // currentGen will be set to last generation when meta loads (see effect above)
  };

  const handleSetGen = (g) => {
    setIsPlaying(false);
    setCurrentGen(g);
  };

  const togglePlay = () => {
    if (currentGen >= maxGen) setCurrentGen(1);
    setIsPlaying(!isPlaying);
  };

  // --- Loading state ---
  if (loadingList) {
    return (
      <div style={{ ...rootStyle, display: "flex", alignItems: "center", justifyContent: "center" }}>
        <div style={{ fontFamily: MONO, color: "#555", fontSize: 14 }}>Loading experiments…</div>
      </div>
    );
  }

  // --- Error state ---
  if (error) {
    return (
      <div style={{ ...rootStyle, display: "flex", alignItems: "center", justifyContent: "center" }}>
        <div style={{ textAlign: "center", maxWidth: 500, padding: 40 }}>
          <div style={{ fontSize: 18, color: "#f87171", marginBottom: 16, fontFamily: DISPLAY }}>
            Failed to load data
          </div>
          <div style={{ fontFamily: MONO, color: "#666", fontSize: 12, marginBottom: 20, lineHeight: 1.6 }}>
            {error}
          </div>
          <div style={{
            fontFamily: MONO, color: "#444", fontSize: 11, lineHeight: 1.8,
            background: "#10121c", padding: 16, borderRadius: 8, border: "1px solid #1a1d2e",
            textAlign: "left",
          }}>
            <div style={{ color: "#888", marginBottom: 8 }}>Checklist:</div>
            <div>1. Run: <code style={{ color: "#60a5fa" }}>python generate_pseudo_data.py</code></div>
            <div>2. Check: <code style={{ color: "#60a5fa" }}>public/data/experiments.json</code> exists</div>
            <div>3. Restart: <code style={{ color: "#60a5fa" }}>npm run dev</code></div>
          </div>
        </div>
      </div>
    );
  }

  // --- Main UI ---
  return (
    <div style={rootStyle}>
      {/* Google Fonts */}
      <link
        href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&family=Space+Grotesk:wght@500;600;700&display=swap"
        rel="stylesheet"
      />

      {/* Header */}
      <header style={{
        flexShrink: 0,
        borderBottom: "1px solid #1a1d2e", padding: "16px 24px",
        display: "flex", alignItems: "center", justifyContent: "space-between",
        background: "#0e1019",
      }}>
        <div style={{ display: "flex", alignItems: "baseline", gap: 12 }}>
          <h1 style={{
            margin: 0, fontSize: 20, fontWeight: 700,
            letterSpacing: "-0.02em", color: "#f0f2f8", fontFamily: DISPLAY,
          }}>
            Neural Evolution Explorer
          </h1>
          <span style={{
            fontFamily: MONO, fontSize: 11, color: "#444",
            letterSpacing: "0.08em", textTransform: "uppercase",
          }}>
            DeepSim × BigGAN
          </span>
        </div>
        <div style={{ fontFamily: MONO, fontSize: 11, color: "#444" }}>
          {filteredExperiments.length === experiments.length
            ? `${experiments.length} experiment${experiments.length !== 1 ? "s" : ""}`
            : `${filteredExperiments.length} / ${experiments.length} experiments`}
        </div>
      </header>

      <div style={{ display: "flex", flex: 1, minHeight: 0, overflow: "hidden" }}>
        {/* Sidebar — scrolls experiment list only */}
        <aside style={{
          width: 260, borderRight: "1px solid #1a1d2e",
          padding: "16px 0", background: "#0b0d14",
          flexShrink: 0, overflowY: "auto", minHeight: 0,
        }}>
          <div style={{
            fontFamily: MONO, padding: "0 16px 12px", fontSize: 10,
            letterSpacing: "0.12em", textTransform: "uppercase", color: "#444",
          }}>
            Experiments
          </div>
          <div style={{ padding: "0 12px 12px" }}>
            <div style={{ marginBottom: 6, fontFamily: MONO, fontSize: 9, letterSpacing: "0.1em", textTransform: "uppercase", color: "#555" }}>Animal</div>
            <select
              value={filterAnimal}
              onChange={(e) => setFilterAnimal(e.target.value)}
              style={{
                width: "100%", padding: "6px 8px", fontFamily: MONO, fontSize: 11,
                background: "#0e1019", border: "1px solid #1a1d2e", borderRadius: 4, color: "#c0c4d0",
                cursor: "pointer",
              }}
            >
              <option value="">All</option>
              {filterOptions.animals.map((a) => (
                <option key={a} value={a}>{a}</option>
              ))}
            </select>
            <div style={{ marginBottom: 6, marginTop: 10, fontFamily: MONO, fontSize: 9, letterSpacing: "0.1em", textTransform: "uppercase", color: "#555" }}>Area</div>
            <select
              value={filterArea}
              onChange={(e) => setFilterArea(e.target.value)}
              style={{
                width: "100%", padding: "6px 8px", fontFamily: MONO, fontSize: 11,
                background: "#0e1019", border: "1px solid #1a1d2e", borderRadius: 4, color: "#c0c4d0",
                cursor: "pointer",
              }}
            >
              <option value="">All</option>
              {filterOptions.areas.map((a) => (
                <option key={a} value={a}>{a}</option>
              ))}
            </select>
            <div style={{ marginBottom: 6, marginTop: 10, fontFamily: MONO, fontSize: 9, letterSpacing: "0.1em", textTransform: "uppercase", color: "#555" }}>Channel</div>
            <select
              value={filterChannel}
              onChange={(e) => setFilterChannel(e.target.value)}
              style={{
                width: "100%", padding: "6px 8px", fontFamily: MONO, fontSize: 11,
                background: "#0e1019", border: "1px solid #1a1d2e", borderRadius: 4, color: "#c0c4d0",
                cursor: "pointer",
              }}
            >
              <option value="">All</option>
              {filterOptions.channels.map((c) => (
                <option key={c} value={c}>{c}</option>
              ))}
            </select>
          </div>
          <div style={{ borderTop: "1px solid #1a1d2e", marginTop: 4, paddingTop: 8 }} />
          {filteredExperiments.length === 0 ? (
            <div style={{ padding: "12px 16px", fontFamily: MONO, fontSize: 11, color: "#555" }}>
              No experiments match filters
            </div>
          ) : null}
          {filteredExperiments.map((exp) => {
            const active = selectedExp === exp.id;
            return (
              <button
                key={exp.id}
                onClick={() => handleExpChange(exp.id)}
                style={{
                  display: "block", width: "100%", textAlign: "left",
                  border: "none", cursor: "pointer",
                  padding: "10px 16px", transition: "all 0.15s",
                  background: active ? "#161929" : "transparent",
                  borderLeft: active ? "2px solid #60a5fa" : "2px solid transparent",
                }}
              >
                <div style={{
                  fontSize: 13, fontWeight: active ? 600 : 400,
                  color: active ? "#e0e4f0" : "#777", fontFamily: MONO,
                }}>
                  {exp.animal}-{exp.unit.replace("Unit ", "")}
                </div>
                <div style={{ fontFamily: MONO, fontSize: 10, color: "#444", marginTop: 2 }}>
                  {exp.area} · {exp.date}
                </div>
              </button>
            );
          })}
        </aside>

        {/* Main content — scrolls content area only */}
        <main style={{ flex: 1, minHeight: 0, padding: 24, overflow: "auto" }}>
          {meta && (
            <>
              {/* Experiment info bar */}
              <div style={{ display: "flex", gap: 24, marginBottom: 20, alignItems: "center", flexWrap: "wrap" }}>
                {[
                  { label: "Experiment", value: `${meta.animal} — ${meta.unit}` },
                  { label: "Area", value: meta.area },
                  { label: "Date", value: meta.date },
                  { label: "Generations", value: meta.num_generations },
                  ...(meta.image_size_deg != null && (meta.image_size_deg || []).length
                    ? [{ label: "Image size", value: `${(meta.image_size_deg || [])[0]} deg` }]
                    : []),
                  ...(meta.image_pos_deg != null && (meta.image_pos_deg || []).length >= 2
                    ? [{ label: "Image pos", value: `(${(meta.image_pos_deg || []).slice(0, 2).join(", ")}) deg` }]
                    : []),
                  ...(meta.thread0_space != null
                    ? [{ label: "Thread 0", value: meta.thread0_space }]
                    : []),
                  ...(meta.thread1_space != null
                    ? [{ label: "Thread 1", value: meta.thread1_space }]
                    : []),
                ].map((item, i) => (
                  <div key={item.label} style={{ display: "flex", alignItems: "center", gap: 24 }}>
                    {i > 0 && <div style={{ width: 1, height: 28, background: "#1a1d2e" }} />}
                    <div>
                      <div style={{
                        fontFamily: MONO, fontSize: 10, color: "#444",
                        letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 2,
                      }}>
                        {item.label}
                      </div>
                      <div style={{ fontSize: 15, fontWeight: 500, color: "#c0c4d0" }}>
                        {item.value}
                      </div>
                    </div>
                  </div>
                ))}
                {loadingGen && (
                  <div style={{ fontFamily: MONO, fontSize: 11, color: "#60a5fa", marginLeft: "auto" }}>
                    loading…
                  </div>
                )}
              </div>

              {/* PSTH display toggles */}
              <div style={{
                display: "flex", alignItems: "center", gap: 20, marginBottom: 16,
                flexWrap: "wrap",
              }}>
                <label style={{ display: "flex", alignItems: "center", gap: 8, cursor: "pointer", fontFamily: MONO, fontSize: 12, color: "#888" }}>
                  <input
                    type="checkbox"
                    checked={showIndividualTrials}
                    onChange={(e) => setShowIndividualTrials(e.target.checked)}
                  />
                  Show individual trial responses
                </label>
                <label style={{ display: "flex", alignItems: "center", gap: 8, cursor: "pointer", fontFamily: MONO, fontSize: 12, color: "#888" }}>
                  <input
                    type="checkbox"
                    checked={showAllGenerationsPsth}
                    onChange={(e) => setShowAllGenerationsPsth(e.target.checked)}
                  />
                  Show all generations (PSTH as gray)
                </label>
                <label style={{ display: "flex", alignItems: "center", gap: 8, cursor: "pointer", fontFamily: MONO, fontSize: 12, color: "#888" }}>
                  <input
                    type="checkbox"
                    checked={tiePsthYLim}
                    onChange={(e) => setTiePsthYLim(e.target.checked)}
                  />
                  Tie PSTH Y limits
                </label>
                {showAllGenerationsPsth && allGenPsth.loading && (
                  <span style={{ fontFamily: MONO, fontSize: 11, color: "#60a5fa" }}>loading…</span>
                )}
              </div>

              {/* Two-column: DeepSim | BigGAN */}
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24, marginBottom: 24 }}>
                {/* DeepSim */}
                <div style={cardStyle}>
                  <div style={{ ...cardTitleStyle, color: "#60a5fa" }}>Pattern (DeepSim)</div>
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 16 }}>
                    <MethodImage
                      src={genData?.deepsim?.imageSrc}
                      fallbackSrc={genData?.deepsim?.imageSrcFallback}
                      rate={genData?.deepsim?.psth?.evoked_rate}
                      gen={currentGen}
                    />
                    <EvolTrajChart evolTraj={evolTraj} currentGen={currentGen} totalGens={maxGen} />
                  </div>
                  <PSTHChart
                    psth={genData?.deepsim?.psth}
                    gen={currentGen}
                    label="DeepSim"
                    color="#60a5fa"
                    showIndividualTrials={showIndividualTrials}
                    allGenerationsMeanRates={showAllGenerationsPsth ? allGenPsth.deepsim : null}
                    yDomain={psthYDomain}
                  />
                </div>

                {/* BigGAN */}
                <div style={cardStyle}>
                  <div style={{ ...cardTitleStyle, color: "#f87171" }}>Object (BigGAN)</div>
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 16 }}>
                    <MethodImage
                      src={genData?.biggan?.imageSrc}
                      fallbackSrc={genData?.biggan?.imageSrcFallback}
                      rate={genData?.biggan?.psth?.evoked_rate}
                      gen={currentGen}
                    />
                    <EvolTrajChart evolTraj={evolTraj} currentGen={currentGen} totalGens={maxGen} />
                  </div>
                  <PSTHChart
                    psth={genData?.biggan?.psth}
                    gen={currentGen}
                    label="BigGAN"
                    color="#f87171"
                    showIndividualTrials={showIndividualTrials}
                    allGenerationsMeanRates={showAllGenerationsPsth ? allGenPsth.biggan : null}
                    yDomain={psthYDomain}
                  />
                </div>
              </div>

              {/* Playback Controls */}
              <PlaybackControls
                currentGen={currentGen}
                maxGen={maxGen}
                isPlaying={isPlaying}
                playSpeed={playSpeed}
                onTogglePlay={togglePlay}
                onSetGen={handleSetGen}
                onSetSpeed={setPlaySpeed}
                onReset={() => { setIsPlaying(false); setCurrentGen(1); }}
              />
            </>
          )}

          {!meta && !loadingExp && (
            <div style={{
              display: "flex", alignItems: "center", justifyContent: "center",
              height: 300, color: "#444", fontFamily: MONO, fontSize: 13,
            }}>
              Select an experiment from the sidebar
            </div>
          )}
        </main>
      </div>
    </div>
  );
}

// =============================================================================
// SHARED STYLES
// =============================================================================

const rootStyle = {
  minHeight: "100vh",
  height: "100vh",
  overflow: "hidden",
  display: "flex",
  flexDirection: "column",
  background: "#0c0e16",
  color: "#d0d4e0",
  fontFamily: "'IBM Plex Sans', system-ui, sans-serif",
};

const cardStyle = {
  background: "#10121c",
  borderRadius: 10,
  border: "1px solid #1a1d2e",
  padding: 20,
};

const cardTitleStyle = {
  fontSize: 16,
  fontWeight: 700,
  marginBottom: 16,
  fontFamily: "'Space Grotesk', sans-serif",
  letterSpacing: "-0.01em",
};
