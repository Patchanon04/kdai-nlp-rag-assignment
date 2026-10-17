import { useEffect, useMemo, useRef, useState } from "react";

const SYSTEMS = [
  { key: "baseline", num: "①", name: "ระบบเดิม", sub: "โค้ดตามสไลด์ Hands-on 1–2", tags: ["standard analyzer", "MarkdownNodeParser", "BM25 + kNN + RRF"] },
  { key: "improved", num: "②", name: "ระบบใหม่", sub: "ตัดคำไทย · chunk ตามหัวข้อ · จัดอันดับใหม่", tags: ["thai analyzer", "chunk ≤ 800 + breadcrumb", "bge-reranker-v2-m3"] },
  { key: "jev", num: "③", name: "ใหม่ + Jev", sub: "ให้ JEV-9B คัดท่อนที่ไม่ช่วยตอบทิ้ง", tags: ["② ทั้งหมด", "JEV-9B relevance", "threshold 0.2"] },
];
const TYPES = {
  official: "คำถามของครู",
  paraphrase: "ถามแบบเปลี่ยนคำ",
  specific: "เจาะข้อเท็จจริง",
  keyword: "คีย์เวิร์ดตรงตัว",
};
const MAX_CHARS = 8404; // chunk ที่ยาวที่สุดของระบบเดิม
const fmt = (n) => n.toLocaleString("th-TH");

// คิวบน server (ค้นทีละคำถาม, AI ตอบได้ทีละ 2) — poll ระหว่างรอ เพื่อบอกผู้ใช้ว่ามีคนรอก่อนหน้ากี่คน
function useQueue(active) {
  const [w, setW] = useState(null);
  useEffect(() => {
    if (!active) { setW(null); return; }  // effect ต้องคืน cleanup function หรือไม่คืนอะไรเลย
    const t = setInterval(() => fetch("./api/status").then((r) => r.json()).then(setW).catch(() => {}), 1500);
    return () => clearInterval(t);
  }, [active]);
  return w;
}

async function post(path, body) {
  const r = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}

export default function App() {
  const [data, setData] = useState([]);
  const [metrics, setMetrics] = useState(null);
  const [live, setLive] = useState(false);
  const [row, setRow] = useState(null);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [view, setView] = useState("compare");
  const queue = useQueue(busy);

  useEffect(() => {
    fetch("./demo_data.json").then((r) => r.json()).then((d) => {
      setData(d.questions);
      setRow(d.questions.find((q) => q.id === "q27") || d.questions[0]);
    }).catch(() => setError("โหลดข้อมูลไม่ได้"));
    fetch("./metrics.json").then((r) => (r.ok ? r.json() : null)).then(setMetrics).catch(() => {});
    fetch("./api/health").then((r) => r.ok && r.json()).then((h) => setLive(!!h?.live)).catch(() => setLive(false));
  }, []);

  const groups = useMemo(() => {
    const g = {};
    for (const q of data) (g[q.type] ||= []).push(q);
    return Object.keys(TYPES).filter((t) => g[t]).map((t) => [t, g[t]]);
  }, [data]);

  async function ask(e) {
    e.preventDefault();
    const q = draft.trim();
    if (!q || busy) return;
    const preset = data.find((x) => x.question === q);
    if (preset && (preset.jev || !live)) return setRow(preset);
    setBusy(true);
    setError("");
    try {
      setRow({ ...(preset || {}), ...(await post("./api/compare", { q })), question: q });
    } catch {
      setError("ค้นไม่สำเร็จ — server อาจปิดอยู่");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="page">
      <header className="mast">
        <div className="mast-meta">
          <span>KDAI · Natural Language Processing</span>
          <span>Assignment · 4 ต.ค. 2026</span>
          <span className={`pill ${live ? "on" : ""}`}>
            <i /> {live ? "โหมดสด — ถามอะไรก็ได้" : "ผลที่บันทึกไว้"}
          </span>
        </div>
        <h1>
          หลักฐานชิ้นไหน<br />
          <em>ถูกส่งให้ AI?</em>
        </h1>
        <p className="lede">
          คำถามเดียวกัน ค้นจากบทความโรค 4 เรื่องเดียวกัน ตอบด้วย LLM ตัวเดียวกัน (Gemma 4 31B) —
          ต่างกันแค่ <b>วิธีหั่น ค้น และคัดเอกสาร</b> ดูว่าแต่ละระบบหยิบท่อนไหนมาให้ AI อ่าน
          ท่อนที่มีคำตอบจริงจะถูก <mark className="hl static">ไฮไลต์สีเขียว</mark>
        </p>
      </header>

      <nav className="tabs">
        <button className={view === "compare" ? "on" : ""} onClick={() => setView("compare")}>เทียบ 3 ระบบ</button>
        <button className={view === "chat" ? "on" : ""} onClick={() => setView("chat")}>แชท</button>
      </nav>

      {view === "chat" ? <Chat live={live} samples={data} /> : <>
      {metrics && <Scoreboard metrics={metrics} />}

      <section className="asker">
        <form onSubmit={ask}>
          <input
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            placeholder={live ? "พิมพ์คำถามเกี่ยวกับ หัดเยอรมัน · อหิวาตกโรค · ต้อกระจก · กรดไหลย้อน" : "โหมดออฟไลน์ — เลือกคำถามด้านล่าง"}
            disabled={!live}
            maxLength={300}
          />
          <button disabled={!live || busy}>{busy ? "กำลังค้น…" : "ค้นหลักฐาน"}</button>
        </form>
        <div className="presets">
          {groups.map(([type, qs]) => (
            <div key={type} className="preset-group">
              <span className="preset-label">{TYPES[type]}</span>
              {qs.map((q) => (
                <button
                  key={q.id}
                  className={`chip ${row?.id === q.id ? "active" : ""}`}
                  onClick={() => { setRow(q); setDraft(q.question); }}
                  title={q.question}
                >
                  {q.question}
                </button>
              ))}
            </div>
          ))}
        </div>
        {busy && (
          <p className="queue">
            กำลังค้นด้วย 3 ระบบ…{queue?.search > 0 && ` มีอีก ${queue.search} คำถามรอคิวอยู่`}
          </p>
        )}
        {error && <p className="error">{error}</p>}
      </section>

      {row && <Dossier key={row.question} row={row} live={live} />}
      </>}

      <footer className="foot">
        ระบบเดิม: <a href="https://github.com/aekanun2020/2025-authenticRAG">aekanun2020/2025-authenticRAG</a> ·
        Jev: <a href="https://huggingface.co/autotrust/JEV-9B">autotrust/JEV-9B</a> (open-weight student ของ TypeSafe Jev) ·
        แนวคิด: <a href="https://huggingface.co/blog/hotchpotch/introducing-jev-reranker">jev-reranker</a>
      </footer>
    </div>
  );
}

// แชท: แต่ละข้อความค้นแยกกัน (ไม่จำบริบทคำถามก่อนหน้า) — ใช้ cache/คิวเดียวกับหน้าเทียบ
function Chat({ live, samples }) {
  const [sys, setSys] = useState("jev");
  const [msgs, setMsgs] = useState([]);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const queue = useQueue(busy);
  const end = useRef(null);
  // ต้องมีปีกกา: Chrome รุ่นใหม่ให้ scrollIntoView คืน Promise — ถ้า return ออกไป React จะเรียกมันเป็น cleanup แล้วพัง
  useEffect(() => { end.current?.scrollIntoView({ behavior: "smooth", block: "end" }); }, [msgs, busy]);
  const picks = useMemo(() => samples.filter((q) => ["q09", "q17", "q27", "q29"].includes(q.id)), [samples]);

  async function send(text, system = sys) {
    const q = (text ?? draft).trim();
    if (!q || busy) return;
    setDraft("");
    setMsgs((m) => [...m, { role: "user", text: q, system }]);
    setBusy(true);
    try {
      const r = await post("./api/chat", { q, system });
      setMsgs((m) => [...m, { role: "bot", q, text: r.answer, system: r.system, note: r.note, sources: r.sources }]);
    } catch {
      setMsgs((m) => [...m, { role: "bot", q, text: "เรียกระบบไม่สำเร็จ — ลองอีกครั้ง", system, sources: [] }]);
    } finally {
      setBusy(false);
    }
  }

  if (!live) return <section className="chat"><p className="chat-off">แชทใช้ได้เฉพาะตอนเปิด server (โหมดสด)</p></section>;
  return (
    <section className="chat">
      <div className="chat-sys">
        <span>คุยกับ</span>
        {SYSTEMS.map((s) => (
          <button key={s.key} className={`sys-${s.key} ${sys === s.key ? "on" : ""}`} onClick={() => setSys(s.key)}>
            {s.num} {s.name}
          </button>
        ))}
      </div>
      <div className="chat-log">
        {msgs.length === 0 && (
          <div className="chat-empty">
            <p>ถามเรื่อง <b>หัดเยอรมัน · อหิวาตกโรค · ต้อกระจก · กรดไหลย้อน</b> — ระบบจะค้นบทความแล้วให้ Gemma ตอบพร้อมแหล่งอ้างอิง</p>
            <div className="chat-picks">{picks.map((q) => <button key={q.id} className="chip" onClick={() => send(q.question)}>{q.question}</button>)}</div>
          </div>
        )}
        {msgs.map((m, i) => m.role === "user" ? (
          <div key={i} className="msg user"><div className="bubble">{m.text}</div></div>
        ) : (
          <BotMsg key={i} m={m} busy={busy} onAsk={(s) => send(m.q, s)} />
        ))}
        {busy && (
          <div className="msg bot"><div className="bubble answer-wait">
            <span /><span /><span /> กำลังค้นเอกสารแล้วให้ AI อ่าน… (~30-60 วินาที)
            {(queue?.search > 0 || queue?.answer > 0) && ` · รอคิวอยู่ ${queue.search + queue.answer}`}
          </div></div>
        )}
        <div ref={end} />
      </div>
      <form className="chat-input" onSubmit={(e) => { e.preventDefault(); send(); }}>
        <input value={draft} onChange={(e) => setDraft(e.target.value)} maxLength={300} placeholder="พิมพ์คำถาม…" disabled={busy} />
        <button disabled={busy || !draft.trim()}>ส่ง</button>
      </form>
      <p className="chat-hint">แต่ละข้อความค้นแยกกัน ระบบไม่จำคำถามก่อนหน้า — ถามให้ครบในประโยคเดียว เช่น "ต้อกระจกรักษายังไง" แทน "แล้วรักษายังไง"</p>
    </section>
  );
}

function BotMsg({ m, busy, onAsk }) {
  const s = SYSTEMS.find((x) => x.key === m.system);
  const chars = m.sources.reduce((a, c) => a + c.chars, 0);
  return (
    <div className={`msg bot sys-${m.system}`}>
      <div className="bubble">
        <div className="msg-sys">{s.num} {s.name}</div>
        {m.note && <div className="msg-note">{m.note}</div>}
        <div className="answer-body">{renderMd(m.text)}</div>
        {m.sources.length > 0 && (
          <details className="msg-src">
            <summary>แหล่งอ้างอิง {m.sources.length} ท่อน · {fmt(chars)} ตัวอักษรที่ส่งให้ AI</summary>
            <ol>
              {m.sources.map((c, i) => (
                <li key={i}>
                  <b>{c.source}</b>{c.crumb && ` › ${c.crumb.split(" > ").slice(1).join(" › ")}`}
                  {c.jev !== undefined && <code> P={c.jev.toFixed(2)}</code>}
                  <p>{c.text.slice(0, 220)}{c.text.length > 220 && " …"}</p>
                </li>
              ))}
            </ol>
          </details>
        )}
        <div className="msg-again">
          ถามข้อนี้กับ{SYSTEMS.filter((x) => x.key !== m.system).map((x) => (
            <button key={x.key} disabled={busy} onClick={() => onAsk(x.key)}>{x.num} {x.name}</button>
          ))}
        </div>
      </div>
    </div>
  );
}

function Scoreboard({ metrics }) {
  const rows = SYSTEMS.map((s) => ({ ...s, m: metrics.rows.find((r) => r.key === s.key) })).filter((s) => s.m);
  const bars = [
    ["hit1", "อันดับ 1 ถูก", "Hit@1"],
    ["hit5", "มีท่อนที่ถูกใน 5 อันดับ", "Hit@5"],
    ["precision", "ท่อนที่ส่งให้ AI มีคำตอบจริง", "Precision@5"],
  ];
  return (
    <section className="score">
      <div className="score-head">
        <h2>ผลวัดรวม</h2>
        <p>{metrics.n} คำถามที่รู้คำตอบล่วงหน้า · วัดว่าค้นเจอท่อนที่มีคำตอบหรือไม่</p>
      </div>
      <div className="score-grid">
        {bars.map(([k, label, code]) => (
          <div key={k} className="metric">
            <div className="metric-name"><b>{label}</b><code>{code}</code></div>
            {rows.map((s) => (
              <div key={s.key} className={`bar sys-${s.key}`}>
                <span className="bar-who">{s.num} {s.name}</span>
                <span className="bar-track"><span className="bar-fill" style={{ "--w": `${s.m[k] * 100}%` }} /></span>
                <span className="bar-val">{Math.round(s.m[k] * 100)}%</span>
              </div>
            ))}
          </div>
        ))}
        <div className="metric">
          <div className="metric-name"><b>ตัวอักษรที่ส่งให้ AI อ่าน</b><code>avg / คำถาม</code></div>
          {rows.map((s) => (
            <div key={s.key} className={`bar sys-${s.key}`}>
              <span className="bar-who">{s.num} {s.name}</span>
              <span className="bar-track"><span className="bar-fill" style={{ "--w": `${(s.m.chars / Math.max(...rows.map((r) => r.m.chars))) * 100}%` }} /></span>
              <span className="bar-val">{fmt(Math.round(s.m.chars))}</span>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

function Dossier({ row, live }) {
  const ev = row.evidence || [];
  const cols = {
    baseline: row.baseline.map((c) => ({ ...c, kept: true })),
    improved: row.improved.map((c) => ({ ...c, kept: true })),
    jev: row.jev
      ? row.improved.map((c, i) => ({ ...c, p: row.jev[i], kept: row.jev[i] >= 0.2 })).sort((a, b) => b.p - a.p)
      : null,
  };
  return (
    <section className="dossier">
      <div className="case">
        <span className="case-tag">{TYPES[row.type] || "คำถามจากผู้ชม"}{row.file ? ` · ${row.file}` : ""}</span>
        <h2>“{row.question}”</h2>
        {ev.length > 0 ? (
          <p className="case-ev">
            ประโยคที่เป็นคำตอบในเอกสาร: {ev.map((e, i) => <mark key={i} className="hl static">{e}</mark>)}
          </p>
        ) : row.reference ? (
          <p className="case-ev none">เอกสารไม่มีคำตอบของคำถามนี้ — ระบบที่ดีควรบอกว่าไม่มีข้อมูล ไม่ใช่แต่งเอง</p>
        ) : (
          <p className="case-ev none">คำถามใหม่ ไม่มีเฉลย — ดูเองว่าท่อนไหนตรงคำถาม</p>
        )}
        {row.reference && ev.length > 0 && <p className="case-ref"><b>คำตอบอ้างอิง:</b> {row.reference}</p>}
      </div>
      <div className="cols">
        {SYSTEMS.map((s, ci) => (
          <Column key={s.key} sys={s} cards={cols[s.key]} ev={ev} delay={ci} />
        ))}
      </div>
      <Answers row={row} live={live} jevReady={!!row.jev} />
    </section>
  );
}

function Answers({ row, live, jevReady }) {
  const [text, setText] = useState(row.answers || {});
  const [busy, setBusy] = useState({});
  async function run(key) {
    setBusy((b) => ({ ...b, [key]: true }));
    try {
      const { answer } = await post("./api/answer", { q: row.question, system: key });
      setText((t) => ({ ...t, [key]: answer }));
    } catch {
      setText((t) => ({ ...t, [key]: "เรียก AI ไม่สำเร็จ ลองอีกครั้ง" }));
    } finally {
      setBusy((b) => ({ ...b, [key]: false }));
    }
  }
  const missing = SYSTEMS.filter((s) => !text[s.key] && (s.key !== "jev" || jevReady));
  const queue = useQueue(Object.values(busy).some(Boolean));
  return (
    <div className="answers">
      <div className="answers-head">
        <h3>คำตอบของ AI <small>Gemma 4 31B · prompt เดียวกันทุกระบบ ต่างกันแค่หลักฐานที่ได้รับ</small></h3>
        {live && missing.length > 0 && (
          <button className="ask-all" onClick={() => missing.forEach((s) => !busy[s.key] && run(s.key))}>
            ให้ AI ตอบ{missing.length === 3 ? "ทั้ง 3 ระบบ" : "ที่เหลือ"}
          </button>
        )}
      </div>
      <div className="cols">
        {SYSTEMS.map((s) => (
          <div key={s.key} className={`answer sys-${s.key}`}>
            <div className="answer-head">{s.num} {s.name}</div>
            {text[s.key] ? (
              <div className="answer-body">{renderMd(text[s.key])}</div>
            ) : busy[s.key] ? (
              <div className="answer-wait">
                <span /><span /><span /> กำลังอ่านหลักฐาน…{queue?.answer > 0 && ` (รอคิว AI อีก ${queue.answer})`}
              </div>
            ) : (
              <div className="answer-none">
                {s.key === "jev" && !jevReady ? "JEV-9B ยังไม่เชื่อมต่อ" : live ? "ยังไม่ได้ถาม" : "ไม่ได้บันทึกคำตอบของข้อนี้ไว้"}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

function Column({ sys, cards, ev, delay }) {
  if (!cards)
    return (
      <div className={`col sys-${sys.key}`}>
        <ColHead sys={sys} />
        <div className="offline">JEV-9B ยังไม่เชื่อมต่อ<br /><small>(รันบน GPU ในเครื่อง — เปิด jev_local_server.py)</small></div>
      </div>
    );
  const kept = cards.filter((c) => c.kept);
  const hits = kept.filter((c) => c.hit).length;
  const first = cards.findIndex((c) => c.kept && c.hit);
  const chars = kept.reduce((a, c) => a + c.chars, 0);
  return (
    <div className={`col sys-${sys.key}`}>
      <ColHead sys={sys} />
      <div className="tally">
        {ev.length > 0 && (
          <div><b>{hits}</b><span>/ {kept.length} ท่อนมีคำตอบ</span></div>
        )}
        {ev.length > 0 && (
          <div><b>{first >= 0 ? `#${first + 1}` : "—"}</b><span>อันดับแรกที่ถูก</span></div>
        )}
        <div><b>{fmt(chars)}</b><span>ตัวอักษรถึง AI</span></div>
      </div>
      <ol className="cards">
        {cards.map((c, i) => <Card key={i} c={c} i={i} ev={ev} jev={sys.key === "jev"} delay={delay * 0.12 + i * 0.07} />)}
      </ol>
    </div>
  );
}

function ColHead({ sys }) {
  return (
    <div className="col-head">
      <span className="col-num">{sys.num}</span>
      <div>
        <h3>{sys.name}</h3>
        <p>{sys.sub}</p>
        <div className="tags">{sys.tags.map((t) => <span key={t}>{t}</span>)}</div>
      </div>
    </div>
  );
}

function highlight(text, ev) {
  if (!ev.length) return text;
  const out = [];
  let rest = text;
  for (;;) {
    let best = -1, len = 0;
    for (const e of ev) {
      const at = rest.indexOf(e);
      if (at >= 0 && (best < 0 || at < best)) [best, len] = [at, e.length];
    }
    if (best < 0) break;
    out.push(rest.slice(0, best), <mark key={out.length} className="hl">{rest.slice(best, best + len)}</mark>);
    rest = rest.slice(best + len);
  }
  out.push(rest);
  return out;
}

function Card({ c, i, ev, jev, delay }) {
  const [open, setOpen] = useState(false);
  const at = c.hit ? Math.min(...ev.map((e) => c.text.indexOf(e)).filter((x) => x >= 0)) : -1;
  // ย่อ: ถ้ามีคำตอบ ตัดหน้าต่างรอบคำตอบมาโชว์ (คำตอบอาจอยู่ลึกในท่อนยาว 8,000 ตัวอักษร)
  const from = at > 160 ? at - 160 : 0;
  const shown = open ? c.text : c.text.slice(from, from + 380);
  const state = c.hit === true ? "hit" : c.hit === false ? "miss" : "";
  return (
    <li className={`card ${state} ${c.kept ? "" : "dropped"}`} style={{ animationDelay: `${delay}s` }}>
      <div className="card-top">
        <span className="rank">{i + 1}</span>
        <span className="src">{c.source}</span>
        {c.hit === true && <span className="verdict yes">มีคำตอบ</span>}
        {c.hit === false && <span className="verdict no">ไม่มีคำตอบ</span>}
      </div>
      {c.crumb && <div className="crumb">{c.crumb.split(" > ").slice(1).join(" › ") || c.crumb}</div>}
      <div className="ruler" title={`${fmt(c.chars)} ตัวอักษร`}>
        <span style={{ width: `${Math.min(c.chars / MAX_CHARS, 1) * 100}%` }} className={c.chars > 2000 ? "long" : ""} />
        <em>{fmt(c.chars)} ตัวอักษร{at > 0 && !open ? ` · คำตอบอยู่ตัวที่ ${fmt(at)}` : ""}</em>
      </div>
      <p className="text" onClick={() => setOpen(!open)}>
        {from > 0 && !open && "… "}
        {highlight(shown, ev)}
        {!open && from + 380 < c.text.length && " …"}
      </p>
      {jev && (
        <div className="jev">
          <span className="jev-track"><span style={{ width: `${c.p * 100}%` }} /><i style={{ left: "20%" }} /></span>
          <span className="jev-val">P(เกี่ยวข้อง) = {c.p.toFixed(2)}</span>
        </div>
      )}
      {!c.kept && <span className="stamp">ตัดทิ้ง</span>}
    </li>
  );
}

// markdown เล็กที่สุดที่ต้องใช้: ย่อหน้า, bullet, **ตัวหนา**, [1] อ้างอิง — React escape ข้อความให้เอง
function renderMd(s) {
  const inline = (t) =>
    t.split(/(\*\*[^*]+\*\*|\[\d+(?:\s*,\s*\d+)*\])/g).map((p, i) =>
      p.startsWith("**") ? <b key={i}>{p.slice(2, -2)}</b> : /^\[\d/.test(p) ? <sup key={i} className="cite">{p}</sup> : p,
    );
  return s.split(/\n+/).filter((l) => l.trim()).map((l, i) => {
    const m = l.match(/^\s*(?:[-*•]|\d+\.)\s+(.*)/);
    return m ? <p key={i} className="li">{inline(m[1])}</p> : <p key={i}>{inline(l.replace(/^#+\s*/, ""))}</p>;
  });
}
