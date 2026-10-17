# KDAI NLP Assignment — Authentic RAG: ระบบเดิม vs ระบบปรับปรุง

Assignment แทน quiz (KDAI 05177141 Natural Language Processing, Part 4) — 4 ต.ค. 2026

| ข้อ | อยู่ที่ |
|---|---|
| (1) ระบบ RAG ตามสไลด์หน้า 22, 25 | [`part1_baseline/`](part1_baseline/) — โค้ดของครูเอ้ ([aekanun2020/2025-authenticRAG](https://github.com/aekanun2020/2025-authenticRAG)) |
| (2) ระบบที่ปรับปรุง (chunking, analyzer, search, scoring, reranking) | [`part2_improved/improvedRAG.py`](part2_improved/improvedRAG.py) |
| (3) อธิบายความแตกต่าง ดีขึ้นอย่างไร เพราะอะไร | README นี้ + ตัวเลขใน [`results/`](results/) |
| ชุดคำถามวัดผล + สคริปต์วัดผล | [`eval/questions.json`](eval/questions.json), [`eval/evaluate.py`](eval/evaluate.py) |
| Jev บนเครื่องตัวเอง | [`part2_improved/jev_local_server.py`](part2_improved/jev_local_server.py) |
| Demo เทียบ 3 ระบบ (React + FastAPI + Docker) | [`demo/`](demo/), [`docker-compose.yml`](docker-compose.yml) |

> **LLM เป็นตัวเดียวกันทั้งสองระบบ**: Gemma 4 31B (`gemma-4-31b-it`) ผ่าน Gemini API ([`llm.py`](llm.py))
> เปลี่ยนจาก Qwen/OpenRouter ในสไลด์เพราะไม่มี OpenRouter key — แต่ใช้ตัวเดียวกันทั้ง Part 1 และ Part 2
> ดังนั้นความต่างที่วัดได้มาจาก retrieval pipeline ล้วน ๆ ตามโจทย์ "ไม่ใช่การเปลี่ยน LLM"
> Embedding model ก็ใช้ตัวเดิม (`BAAI/bge-m3`) ทั้งสองระบบ

## ผลสรุป

**Demo เทียบ 3 ระบบแบบเห็นกับตา:** ดู [หัวข้อ Demo](#demo-เทียบ-3-ระบบ) ด้านล่าง

วัดบน 29 คำถาม ([`eval/questions.json`](eval/questions.json)) — 28 ข้อรู้ตำแหน่งคำตอบในเอกสาร ใช้คำนวณตัวเลขด้านล่าง
ส่วนข้อ 29 (ข้อ 5 ของครู) เอกสารไม่มีคำตอบ จึงใช้ดูว่าระบบยอมรับว่า "ไม่มีข้อมูล" หรือไม่ · top-5 —
รายละเอียดทุกแบบ + รายข้อใน [`results/retrieval_metrics.md`](results/retrieval_metrics.md)

| ระบบ | Hit@1 | Hit@5 | MRR@5 | Precision@5 | ตัวอักษรที่ส่งให้ LLM |
|---|---|---|---|---|---|
| ① ระบบเดิม (Part 1) | 0.68 | 0.96 | 0.80 | 0.21 | 11,079 |
| ② ระบบใหม่: thai analyzer + chunk ใหม่ + context ใหม่ + bge-reranker | **0.82** | **1.00** | **0.90** | 0.25 | 3,229 |
| ③ ② + JEV-9B relevance filter | 0.79 | 1.00 | 0.86 | 0.40 | 2,399 |
| ②′ ② + Gemma relevance filter (แนวคิด Jev ด้วย LLM ตัวเดิม) | 0.82 | 1.00 | 0.90 | **0.68** | **1,526** |

ทีละขั้น (ablation):

| | Hit@1 | Hit@5 | MRR@5 |
|---|---|---|---|
| BM25 เดิม (`standard` analyzer) | 0.00 | **0.07** | 0.04 |
| BM25 ใหม่ (`thai` analyzer + chunk/context ใหม่) | 0.68 | **0.89** | 0.74 |
| dense เดิม → ใหม่ | 0.71 → 0.79 | 0.93 → 0.96 | 0.80 → 0.86 |
| hybrid RRF เดิม → ใหม่ | 0.68 → 0.79 | 0.96 → 1.00 | 0.80 → 0.86 |
| + bge-reranker-v2-m3 | 0.82 | 1.00 | 0.90 |
| hybrid top-20 → JEV-9B จัดอันดับ+กรองเอง (แบบในบทความ jev-reranker) | 0.79 | 0.96 | 0.84 |

**อ่านผลยังไง**
- **ค้นเจอแม่นขึ้น:** ข้อที่ chunk อันดับ 1 มีคำตอบเพิ่มจาก 68% → 82% และทุกข้อมีคำตอบอยู่ใน top-5 (96% → 100%)
- **ส่งขยะให้ LLM น้อยลงมาก:** ตัวอักษรที่ LLM ต้องอ่านลดจาก ~11,000 → ~3,200 (ลด 71%) และเหลือ ~1,500 เมื่อเพิ่ม filter
  โดยที่ Hit@5 ไม่ลดเลย — chunk ของระบบเดิมยาวถึง 8,404 ตัวอักษร คำตอบจริงจมอยู่กลางข้อความยาว
- **Precision@5** = สัดส่วนของ chunk ที่ส่งให้ LLM ที่มีคำตอบจริง — filter ทั้งสองแบบทำให้ค่านี้สูงขึ้น (0.25 → 0.40 / 0.68)
- หมายเหตุความยุติธรรม: การนับ "เจอคำตอบ" ด้วยข้อความ evidence **เข้าข้างระบบเดิม** เพราะ chunk ยิ่งยาวยิ่งมีโอกาสครอบ evidence
  ระบบเดิมจึงได้ Hit@5 สูงถึง 0.96 ทั้งที่ BM25 ของมันแทบใช้ไม่ได้ (Hit@5 = 0.07)

**คุณภาพคำตอบสุดท้าย** (LLM-as-judge 1-5, 9 คำถาม showcase, [`results/answer_judgement.md`](results/answer_judgement.md)):
① 4.33 · ② 4.44 · ③ 4.44 — ต่างกันไม่มากเพราะ Gemma ตอบได้ดีถ้าคำตอบอยู่ในกองข้อความ แต่มีสองข้อที่น่าสนใจ:
- **q09 ถามด้วยคำชาวบ้าน "โรคห่า"** (ไม่มีคำว่าอหิวาตกโรค): ระบบเดิม 2 → ระบบใหม่ 5 — context ภาษาไทยที่ใส่คำพ้องช่วยตรงนี้
- **q26 ถามสองเรื่องในข้อเดียว (สาเหตุ *และ* การป้องกัน)**: ระบบเดิม 4 → ระบบใหม่ 2 — chunk เล็ก 5 ชิ้นแรกพูดถึง "สาเหตุ" ทั้งหมด
  LLM จึงตอบถูกว่า "บริบทไม่มีข้อมูลการป้องกัน" ส่วน chunk ยาวของระบบเดิมบังเอิญครอบหัวข้อการป้องกันด้วย —
  เป็นราคาของ chunk เล็กกับคำถามหลายประเด็น แนวทางต่อไปคือ **query decomposition** (แยกคำถามย่อยแล้วค้นทีละประเด็น)
- **q29 (ข้อ 5 ของครู — เอกสารไม่มีคำตอบ)**: JEV-9B ให้คะแนน chunk ทั้ง 5 ต่ำกว่า 0.2 จึงตัดทิ้งหมด ส่ง 0 ตัวอักษรให้ LLM
  ซึ่งตอบสั้น ๆ ว่าไม่พบข้อมูล ขณะที่ระบบเดิมส่ง 11,737 ตัวอักษรไปให้ LLM อ่านก่อนจะบอกว่าไม่มี

---

## (1) ระบบเดิม — Part 1

นำเข้าเอกสาร (`authenticRAG.py`, Hands-on 1) และ ถาม-ค้น-สรุป (`onlysearchAuthenticRAG.py`, Hands-on 2) ตามสไลด์:

```
.md ─► MarkdownNodeParser ─► LLM สร้าง context ต่อ chunk ─► bge-m3 embedding ─► anthropic-vector-index (kNN)
                                                         └──────────────────► anthropic-bm25-index (BM25, standard analyzer)
คำถาม ─► BM25 top-10 + kNN top-10 ─► RRF ─► top-5 ─► LLM ตอบ
```

แก้จากต้นฉบับเฉพาะส่วนที่จำเป็นต่อการรัน (ทุกจุดมี comment `[PORT]`):
- เรียก LLM ผ่าน `llm.py` แทน OpenRouter
- `TextLoader(..., encoding="utf-8")` — บน Windows ถ้าไม่ระบุจะอ่านไฟล์ไทยไม่ได้
- path ของ corpus/ผลลัพธ์ ให้ชี้ไปโฟลเดอร์ของ repo นี้

ส่วน chunking / indexing / search / RRF / prompt **ไม่ได้แตะ** เพื่อให้เป็น baseline ที่ตรงกับสไลด์

## (2) ระบบปรับปรุง — Part 2

```
.md ─► แบ่งตามหัวข้อ + จำกัด 800 ตัวอักษร + overlap + breadcrumb
    ─► LLM สร้าง context (เห็นสารบัญเอกสาร + section แม่, ตอบภาษาไทย + คำพ้อง)
    ─► bge-m3 embedding ของ (breadcrumb + context + chunk)
    ─► improved-rag-index  (kNN HNSW + BM25 ด้วย analyzer "thai" บน content/context/breadcrumb)
คำถาม ─► BM25 top-20 + kNN top-20 ─► RRF (rank เริ่ม 1) ─► top-20
      ─► cross-encoder bge-reranker-v2-m3 ─► top-5
      ─► JEV-9B relevance filter (instruction-steered, P ≥ 0.2) ─► ตอบพร้อมอ้างอิง [n]
```

## (3) ต่างกันตรงไหน ดีขึ้นอย่างไร เพราะอะไร

### 3.1 BM25 ของเดิมแทบใช้กับภาษาไทยไม่ได้ → ใช้ analyzer `thai`

ระบบเดิมสร้าง BM25 index ด้วย `standard` analyzer ซึ่งตัดคำด้วยช่องว่าง/เครื่องหมาย แต่ภาษาไทยไม่เว้นวรรคระหว่างคำ
ผลคือ **ทั้งวลีกลายเป็น token เดียว** ทดสอบกับ OpenSearch `_analyze` จริง:

| analyzer | ข้อความ `ภาวะแทรกซ้อนในหญิงตั้งครรภ์ช่วงไตรมาสแรก` ถูกตัดเป็น |
|---|---|
| `standard` (เดิม) | `[ภาวะแทรกซ้อนในหญิงตั้งครรภ์ช่วงไตรมาสแรก]` — 1 token |
| `thai` (ใหม่) | `[ภาวะ, แทรกซ้อน, หญิง, ครรภ์, ไตร, มาส]` |

คำถามจะ match กับเอกสารได้ก็ต่อเมื่อพิมพ์วลีระหว่างช่องว่างตรงกันทุกตัวอักษร ซึ่งแทบไม่เกิด ทำให้ขา sparse ของ hybrid search
เกือบไม่ช่วยอะไร (ดู Hit@5 ของ "P1 baseline: BM25" ในตาราง) analyzer `thai` (มีอยู่ใน OpenSearch อยู่แล้ว ไม่ต้องติดตั้งอะไรเพิ่ม)
ตัดคำด้วยพจนานุกรม ทำให้ TF/DF ในสไลด์หน้า 24 คำนวณบน "คำ" จริง BM25 จึงกลับมาทำงาน

### 3.2 Contextual Retrieval ของเดิมไม่ได้ส่ง "เอกสาร" ให้ LLM → ส่งสารบัญ + section แม่

หัวใจของ Contextual Retrieval (Anthropic) คือให้ LLM อ่านเอกสารทั้งฉบับ แล้วเขียนบริบทบอกว่า chunk นี้อยู่ตรงไหนของเอกสาร
แต่ในโค้ดเดิม:

```python
full_doc_content = doc.page_content  # ถ้าไม่มีเอกสารเต็ม ใช้ chunk เอง
context = self.generate_context(full_doc_content, doc.page_content)
```

`<document>` กับ `<chunk>` ที่ส่งให้ LLM คือข้อความเดียวกัน LLM จึงทำได้แค่สรุป chunk ซ้ำ ไม่รู้ว่า chunk อยู่ในเอกสารเรื่องอะไร
หัวข้อไหน และ prompt เป็นภาษาอังกฤษ บริบทที่ได้จึงออกมาเป็น**ภาษาอังกฤษ** (เช่น *"Overview of German Measles (Rubella), …"*)
ซึ่งไม่ช่วย BM25 เมื่อผู้ใช้ถามเป็นภาษาไทย

ระบบใหม่ส่ง **ชื่อเอกสาร + สารบัญหัวข้อทั้งหมด + section แม่ของ chunk** และขอบริบทภาษาไทย 1-2 ประโยค ที่ระบุชื่อโรค
แง่มุม (สาเหตุ/อาการ/รักษา) และคำพ้องที่คนอาจใช้ค้น — ช่วยคำถามแบบ paraphrase เช่นถาม "โรคห่า" แทน "อหิวาตกโรค"
(ไม่ส่งเอกสารเต็มเพราะ free tier ของ Gemma จำกัด 16k input token/นาที แต่เอกสารแต่ละไฟล์ยาว ~20k token —
สารบัญ + section แม่ให้ข้อมูล "ตำแหน่งในเอกสาร" ที่จำเป็นในขนาดที่รับได้)

### 3.3 Chunk ของเดิมใหญ่และไม่สม่ำเสมอ → แบ่งตามหัวข้อ + จำกัดขนาด + breadcrumb

`MarkdownNodeParser(chunk_size=256)` แบ่งตามหัวข้อ markdown อย่างเดียว **ไม่สนใจ `chunk_size`**
ได้ 53 chunk ขนาด 112 – 8,404 ตัวอักษร (median 1,188; 22 chunk เกิน 1,500)

| | เดิม | ใหม่ |
|---|---|---|
| จำนวน chunk | 53 | 159 |
| ขนาด (ตัวอักษร) min / median / max | 112 / 1,188 / 8,404 | 26 / 637 / 1,074 |

ปัญหาของ chunk ใหญ่: (ก) embedding ของข้อความยาว ๆ เป็นค่าเฉลี่ยของหลายเรื่อง ความหมายเฉพาะจุดเลือนหาย
(ข) bge-m3 / reranker อ่านได้จำกัด (reranker ตั้งไว้ 512 token) ส่วนท้ายของ chunk ยาวจะถูกตัดทิ้ง
(ค) context ที่ส่งให้ LLM ตอบยาวและมีส่วนไม่เกี่ยวข้องเยอะ — ระบบเดิมยังประมาณ token ด้วย `len(context.split()) * 1.3`
ซึ่งผิดมากกับภาษาไทย (ไม่มีช่องว่าง) จึงไม่เคยตัด context จริง

ระบบใหม่แบ่งตามหัวข้อก่อน แล้วตัดแต่ละ section ไม่เกิน ~800 ตัวอักษรตามขอบบรรทัด overlap 1 บรรทัด
และแปะ **breadcrumb** (`ชื่อเอกสาร > หัวข้อ > หัวข้อย่อย`) ไว้กับทุก chunk ทั้งใน embedding และ BM25 (boost ×2)
chunk เล็ก ๆ อย่าง "ระยะฟักตัว 12-24 วัน" จึงยังรู้ว่าเป็นของโรคหัดเยอรมัน

### 3.4 RRF + Reranking ด้วย cross-encoder

- RRF เดิมใช้ `1/(k + rank)` โดย rank เริ่ม 0 (สูตรมาตรฐานเริ่ม 1) และดึง candidate แค่ฝั่งละ 10 → ใหม่: rank เริ่ม 1, ฝั่งละ 20
- เพิ่ม **`BAAI/bge-reranker-v2-m3`** (cross-encoder, รองรับภาษาไทย) จัดอันดับ top-20 ใหม่ — bi-encoder (bge-m3) เข้ารหัสคำถาม
  กับเอกสารแยกกัน ส่วน cross-encoder อ่าน *คำถาม+chunk พร้อมกัน* จึงจับความสัมพันธ์ละเอียดกว่า เหมาะกับขั้นสุดท้ายที่ candidate น้อย
  ตรงกับผลของ Anthropic ในสไลด์หน้า 19 (reranking ลด failed retrieval จาก 2.9% → 1.9%)

### 3.5 Jev relevance filter — ตามที่ครูเอ้แนะนำ (jev-reranker)

[jev-reranker](https://huggingface.co/blog/hotchpotch/introducing-jev-reranker) ต่างจาก reranker ทั่วไปตรงที่
**ใช้ instruction กำหนดว่า "เกี่ยวข้อง" หมายถึงอะไร** แล้วให้ความน่าจะเป็นที่ calibrate แล้ว (0-1) จึง**คัดทิ้ง**ด้วย threshold ได้
ไม่ใช่แค่เรียงลำดับ (bge-reranker ให้คะแนนดิบที่ใช้เทียบกันได้ แต่ตั้ง threshold ไม่ได้)

- Jev ตัวจริง (TypeSafe Jev 1.13) เป็นโมเดลปิด เรียกผ่าน API เสียเงิน → ใช้ **[autotrust/JEV-9B](https://huggingface.co/autotrust/JEV-9B)**
  ซึ่งเป็น open-weight (Apache-2.0) ที่ distill จาก Jev 1.13 (ผู้ทำรายงาน KL ≈ 0.019 เทียบตัวจริง) **รันบนเครื่องตัวเอง**
  ([`part2_improved/jev_local_server.py`](part2_improved/jev_local_server.py), RX 7800 XT 16 GB + ROCm)
- ใช้ instruction + criteria **ชุดเดียวกับไลบรารี jev-reranker** (pointwise relevance), ใช้ P(true) เป็นคะแนน, threshold 0.2 ตามค่า default
- ทดสอบภาษาไทย: "หัดเยอรมันเกิดจากเชื้ออะไร" กับข้อความที่ตอบตรง → 0.986, กับข้อความเรื่องต้อกระจก → 0.004

ผล: JEV-9B ตัด chunk ที่ไม่ช่วยตอบได้จริง (Precision@5 0.25 → 0.40, ตัวอักษรลด 26%) โดย Hit@5 ไม่ลด
แต่**คัดได้หลวมกว่า** filter ที่ใช้ Gemma ทำแนวคิดเดียวกัน (Precision 0.68) และการเรียงใหม่ตามคะแนน Jev ทำให้ Hit@1 ลดเล็กน้อย
(0.82 → 0.79) — สอดคล้องกับ model card ที่ระบุว่าฝึกด้วยข้อมูลภาษาอังกฤษ (*English-centric*) ภาษาไทยอาศัยความสามารถของ
Qwen3.5 ที่เป็นฐานเท่านั้น ข้อดีของ JEV-9B คือเร็วและไม่เสีย quota LLM: 1 การตัดสิน = 1 forward pass (~0.6 วินาที/chunk บนการ์ดนี้)
ไม่ต้อง generate ข้อความ

บันทึกการทำให้รันบนการ์ด 16 GB ได้ (เผื่อเพื่อนทำต่อ): โหลด nf4 ผ่าน bitsandbytes บน ROCm ใช้ RAM ระหว่างโหลดเกิน → ใช้ bf16
แล้ววาง `lm_head` (ไม่ได้ใช้) และ `embed_tokens` (lookup เอง) ไว้บน CPU, ชั้น transformer ทั้ง 32 ชั้นบน GPU,
และติดตั้ง `flash-linear-attention` (Gated DeltaNet kernel) — เร็วขึ้นจาก ~32 → ~0.6 วินาทีต่อ chunk

### 3.6 Prompt ตอบคำถาม

ระบบใหม่ใส่แหล่งที่มา (`ไฟล์ > หัวข้อ`) ให้แต่ละ chunk ให้ตอบจาก context เท่านั้น อ้างอิง `[n]` และบอกตรง ๆ ถ้าเอกสารไม่มีคำตอบ
(LLM ตัวเดิม เปลี่ยนแค่ prompt)

---

## Demo เทียบ 3 ระบบ

หน้าเว็บ ([`demo/`](demo/), React + FastAPI) วางคำถามเดียวกันให้ 3 ระบบค้นพร้อมกัน แล้วโชว์ข้างกัน:

- top-5 ของแต่ละระบบ — chunk ที่มีคำตอบจริงขึ้นสีเขียวและไฮไลต์ประโยคคำตอบ, แถบความยาวแต่ละ chunk
  (chunk ยาวของระบบเดิมเห็นชัดว่าคำตอบจมอยู่ตรงไหน), คอลัมน์ ③ แสดงคะแนน Jev และประทับ "ตัดทิ้ง" ใน chunk ที่ไม่ผ่าน
- ตัวอักษรที่แต่ละระบบส่งให้ LLM และคำตอบของ LLM ทั้ง 3 ระบบเรียงเทียบกัน
- ตัวเลขผลวัดรวมด้านบน + คำถามตัวอย่าง 29 ข้อ (ผลคำนวณไว้แล้ว เปิดได้ทันที)
- **โหมดสด** — พิมพ์คำถามใหม่ได้ (~3 วินาทีต่อคำถามเมื่อ reranker และ Jev อยู่บน GPU), มีคิวเมื่อหลายคนถามพร้อมกัน
  (ค้นทีละคำถาม, LLM ตอบพร้อมกันได้ 2 เพื่อไม่ชน rate limit), คำถามซ้ำตอบจาก cache, ใช้บนมือถือได้
- **แท็บแชท** — คุยถาม-ตอบแบบแชทบอทได้จริง เลือกระบบที่จะคุยด้วย (① / ② / ③), คำตอบมีอ้างอิง [n]
  และกดดูแหล่งอ้างอิง (ท่อนที่ส่งให้ LLM + จำนวนตัวอักษร) ได้, ปุ่ม "ถามข้อนี้กับ…" ส่งคำถามเดิมให้อีกระบบเพื่อเทียบ —
  แต่ละข้อความค้นแยกกัน (ไม่จำบริบทคำถามก่อนหน้า)
- ถ้า JEV-9B ไม่ได้เปิด คอลัมน์ ③ ขึ้นว่ายังไม่เชื่อมต่อ (แชทระบบ ③ ใช้ ② แทนและบอกไว้); ถ้า GPU reranker ไม่ได้เปิด จะใช้ CPU แทน (~25 วินาที)

reranker บน GPU (fp16) ให้ผลเหมือนบน CPU (fp32) ทุกอันดับ — รันวัดผลชุดเต็มซ้ำด้วย GPU แล้ว ตัวเลขทั้ง 10 แบบและอันดับรายข้อตรงกันทุกบรรทัด

## วิธีวัดผล

[`eval/questions.json`](eval/questions.json): 29 คำถาม — 24 ข้อสร้างจากเอกสาร (ไฟล์ละ 6: keyword 2, paraphrase 2, specific 2) + 5 ข้อของครู
แต่ละข้อมี **evidence** = ข้อความที่คัดลอกตรงจากเอกสารซึ่งเป็นคำตอบ (ตรวจแล้วว่าทุก evidence พบในไฟล์ 1 ครั้ง)

- chunk ที่ "ถูก" = chunk ที่มีข้อความ evidence อยู่ — ไม่ผูกกับวิธีตัด chunk จึงเทียบสองระบบได้
  (หมายเหตุ: วิธีนี้**เข้าข้างระบบเดิม** เพราะ chunk ยาว 8,000 ตัวอักษรมีโอกาสครอบ evidence มากกว่า)
- **Hit@k** = สัดส่วนคำถามที่มี chunk ถูกอยู่ใน top-k, **MRR@5** = ค่าเฉลี่ย 1/rank ของ chunk ถูกตัวแรก
- **Precision@5** = สัดส่วน chunk ที่ส่งให้ LLM ที่มีคำตอบจริง (ส่ง 0 chunk นับเป็น 0)
- **avg chars to LLM** = ความยาว context เฉลี่ยที่ส่งให้ LLM ตอบ (ยิ่งน้อยยิ่งประหยัด/ไม่มีขยะ ถ้า Hit ไม่ลด)
- คำตอบของคำถาม showcase (5 ข้อของครู + 4 ข้อตัวอย่าง) ให้ LLM ตัวเดิมเป็นกรรมการ เทียบคำตอบอ้างอิงจากเอกสาร
  → [`results/answer_judgement.md`](results/answer_judgement.md)

## วิธีรัน

### เปิด demo บนเครื่องไหนก็ได้ — Docker อย่างเดียว

```bash
git clone https://github.com/Patchanon04/kdai-nlp-rag-assignment && cd kdai-nlp-rag-assignment
docker compose up -d                  # ไม่มี GPU: ระบบ ① ② ครบ, ③ แสดงผลที่บันทึกไว้ของ 29 คำถามตัวอย่าง
docker compose --profile gpu up -d    # มี NVIDIA ≥ 16 GB: เพิ่ม JEV-9B + reranker บน GPU (คำถามใหม่ ~3 วินาที)
```

เปิด http://localhost:8091 — ครั้งแรกใช้เวลาสักพัก: build image, service `init` สร้าง index ทั้งสองระบบ
และโหลด bge-m3 + bge-reranker (~4 GB) ลง volume (โปรไฟล์ gpu โหลด JEV-9B อีก ~18 GB)

- **ไม่ต้องมี API key ก็เปิดได้** — `init` สร้าง index จาก context ที่ cache ไว้ใน repo (`results/llm_cache.json`,
  `results/improved_contexts.json`) จึงได้ index เดียวกับที่ใช้วัดผลโดยไม่เรียก LLM
- ถ้าอยากให้ปุ่ม "ให้ AI ตอบ" ทำงาน: สร้างไฟล์ `.env` ที่ root ใส่ `GEMINI_API_KEY=...` (ไม่มี key จะขึ้นข้อความบอก)
- เปลี่ยน port ได้ด้วย `APP_PORT=... OS_PORT=... docker compose up -d`
- RAM แนะนำ ≥ 12 GB สำหรับ Docker (OpenSearch + bge-m3 + reranker บน CPU)
- ⚠ โปรไฟล์ `gpu` ([`Dockerfile.jev`](Dockerfile.jev)) ยังไม่ได้ทดสอบบนการ์ด NVIDIA จริง — Docker ใช้ GPU ได้เฉพาะ NVIDIA
  ส่วนเครื่องที่ทำงานนี้เป็น AMD จึงรัน JEV-9B นอก Docker ด้วย ROCm ใน WSL (ด้านล่าง) ด้วยโค้ดชุดเดียวกัน

### เครื่องที่ทำงานนี้ (AMD RX 7800 XT + WSL)

ดับเบิลคลิก [`demo/start.cmd`](demo/start.cmd) — เปิด Docker (แก้ปัญหา socket ค้างของ Docker Desktop ให้ด้วย),
สตาร์ท JEV-9B + reranker บน GPU ใน WSL ([`demo/run_jev.sh`](demo/run_jev.sh)), อัปเดต IP ของ WSL ใน `.env`, เปิด compose แล้วเปิดเบราว์เซอร์
(ใน `.env` ตั้ง `HF_CACHE=` ชี้ cache โมเดลบนเครื่องเพื่อไม่ต้องโหลดซ้ำ; ปิดโปรแกรมอื่นที่ใช้ GPU ก่อน)

### รันแต่ละส่วนเอง (ทำซ้ำการทดลอง)

```bash
pip install -r requirements.txt                 # Python 3.10+
docker compose up -d opensearch
export GEMINI_API_KEY=...  LLM_CACHE=results/llm_cache.json   # LLM_CACHE: รันซ้ำได้ผลเดิมโดยไม่เรียก API
# Part 1 — นำเข้า แล้วถาม 5 ข้อของครู
cd part1_baseline && python authenticRAG.py && python onlysearchAuthenticRAG.py && cd ..
# Part 2 — นำเข้า
cd part2_improved && python improvedRAG.py ingest && cd ..
# JEV-9B + reranker บน GPU (Linux/WSL)
hf download autotrust/JEV-9B --local-dir ~/models/JEV-9B --include "*.json" "*.jinja" "model-*.safetensors" "head.safetensors" "adapter/*"
hf download BAAI/bge-reranker-v2-m3 --local-dir ~/models/bge-reranker-v2-m3
pip install flash-linear-attention einops   # ถ้าไม่มีจะช้าลง ~50 เท่า
RERANKER=~/models/bge-reranker-v2-m3 python part2_improved/jev_local_server.py ~/models/JEV-9B   # :8766 /score + /rerank
# วัดผล (ตั้ง RERANK_URL=http://localhost:8766/rerank เพื่อใช้ reranker บน GPU — ผลเท่ากับ CPU ทุกอันดับ)
python eval/evaluate.py
python demo/server.py build q25,q26,q27,q28,q29,q09,q14,q17,q22   # ข้อมูลหน้า demo
python eval/evaluate.py judge
```

ปัญหาที่เจอตอนรัน JEV-9B บน ROCm/WSL และวิธีแก้ (อยู่ในโค้ดแล้ว): transformers จองหน่วยความจำ GPU ก้อนเดียว ~13 GB ก่อนโหลดแล้วค้าง
→ ปิด `caching_allocator_warmup`; reranker ทีละ 20 คู่ OOM เมื่อรันคู่กับ JEV-9B → แบ่งทีละ 5 คู่ ความยาวคงที่ 512 token
(ROCm คอมไพล์ kernel ใหม่ทุกครั้งที่เจอรูปร่าง input ใหม่ ครั้งแรก ~10-20 วินาที จึง pad ให้รูปร่างเดียวและ warm-up ตอนเปิด)

เอกสารประกอบต้นฉบับของครูอยู่ใน [`docs_original/`](docs_original/)
