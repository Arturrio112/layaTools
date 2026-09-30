//! BM25 search over the chunk index, plus an optional Laya re-rank.

use crate::index::Index;
use crate::text::query_terms;
use anyhow::Result;
use serde::Serialize;
use std::collections::HashMap;

#[derive(Serialize, Clone)]
pub struct Hit {
    pub path: String,
    pub start: i64,
    pub end: i64,
    pub bm25: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub laya: Option<f64>,
    pub snippet: String,
    #[serde(skip)]
    pub text: String,
}

/// Build an FTS5 MATCH expression: every term OR-ed together (the porter tokenizer handles stems).
fn match_expr(terms: &[String]) -> String {
    terms
        .iter()
        .map(|t| format!("\"{t}\""))
        .collect::<Vec<_>>()
        .join(" OR ")
}

type Row = (String, i64, i64, String, f64);

fn fts_rows(index: &Index, expr: &str, limit: usize) -> Result<Vec<Row>> {
    // Column weights: path and symbol matches count far more than a body mention.
    let mut stmt = index.conn.prepare(
        "SELECT file, start, end, text, bm25(chunks, 6.0, 4.0, 1.0) AS score
         FROM chunks WHERE chunks MATCH ?1 ORDER BY score LIMIT ?2",
    )?;
    let rows = stmt.query_map(rusqlite::params![expr, limit as i64], |r| {
        Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?))
    })?;
    Ok(rows.collect::<rusqlite::Result<_>>()?)
}

/// Top chunks by cosine similarity to `q` (vectors are unit length, so a dot product). The
/// returned score is the negated cosine, to sort like bm25 (lower is better).
fn semantic_rows(index: &Index, q: &[f32], limit: usize) -> Result<Vec<Row>> {
    let vecs = index.vectors()?;
    let mut scored: Vec<(usize, f32)> =
        vecs.iter().enumerate().map(|(i, c)| (i, c.4.iter().zip(q).map(|(a, b)| a * b).sum())).collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(limit);
    scored
        .into_iter()
        .map(|(i, dot)| {
            let (rowid, file, start, end, _) = &vecs[i];
            Ok((file.clone(), *start, *end, index.chunk_text(*rowid)?, -(dot as f64)))
        })
        .collect()
}

/// Tuning knob read from the environment (`LT_<NAME>`), used by the eval sweeps; defaults apply otherwise.
fn knob(name: &str, default: f64) -> f64 {
    std::env::var(format!("LT_{name}")).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// Weight of the Laya judge relative to the retrieval order when fusing.
const JUDGE_WEIGHT: f64 = 1.0;

/// Weight of the semantic list relative to the whole-chunk keyword list.
const SEM_WEIGHT: f64 = 1.0;

/// Reciprocal-rank fusion constant: larger flattens the influence of the very top ranks.
const RRF_K: f64 = 10.0;
/// How much a file/symbol-name match counts relative to a whole-chunk match.
const NAME_WEIGHT: f64 = 0.7;

/// Search the index. Two ranked lists are fused per file: one over everything (path, symbols, body)
/// and one over only path and symbol names, so a file *called* what you asked about beats a file
/// that merely mentions it.
pub fn search(
    index: &Index,
    question: &str,
    k: usize,
    per_file: usize,
    snippet_lines: usize,
    qvec: Option<&[f32]>,
) -> Result<Vec<Hit>> {
    let terms = query_terms(question);
    if terms.is_empty() && qvec.is_none() {
        return Ok(vec![]);
    }
    let expr = match_expr(&terms);
    let (all, names) = if terms.is_empty() {
        (vec![], vec![])
    } else {
        (fts_rows(index, &expr, k * per_file * 6)?, fts_rows(index, &format!("{{path symbols}} : ({expr})"), 60)?)
    };
    let sem = match qvec {
        Some(q) => semantic_rows(index, q, k * per_file * 6)?,
        None => vec![],
    };

    let mut score: HashMap<&str, f64> = HashMap::new();
    let (rrf_k, name_w, sem_w) = (knob("RRF_K", RRF_K), knob("NAME_WEIGHT", NAME_WEIGHT), knob("SEM_WEIGHT", SEM_WEIGHT));
    for (weight, rows) in [(1.0, &all), (name_w, &names), (sem_w, &sem)] {
        let mut seen = std::collections::HashSet::new();
        for (rank, row) in rows.iter().filter(|r| seen.insert(r.0.as_str())).enumerate() {
            *score.entry(row.0.as_str()).or_insert(0.0) += weight / (rrf_k + rank as f64);
        }
    }
    let mut files: Vec<&str> = score.keys().copied().collect();
    files.sort_by(|a, b| score[b].partial_cmp(&score[a]).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(b)));

    let mut hits = Vec::new();
    for file in files.into_iter().take(k) {
        // Prefer the file's best whole-chunk matches; fall back to its best name match.
        // The semantic best chunk leads (it is the one that answers the question), then keyword chunks.
        let mut chunks: Vec<&Row> = sem.iter().find(|r| r.0 == file).into_iter().collect();
        for r in all.iter().filter(|r| r.0 == file) {
            if chunks.len() < per_file && !chunks.iter().any(|c| c.1 == r.1) {
                chunks.push(r);
            }
        }
        if chunks.is_empty() {
            chunks.extend(names.iter().find(|r| r.0 == file));
        }
        chunks.truncate(per_file);
        for (path, start, end, text, bm) in chunks {
            let snippet = best_snippet(text, &terms, snippet_lines, *start);
            hits.push(Hit { path: path.clone(), start: *start, end: *end, bm25: -bm, laya: None, snippet, text: text.clone() });
        }
    }
    Ok(hits)
}

/// The few lines of the chunk that mention the most query terms, prefixed with line numbers.
fn best_snippet(text: &str, terms: &[String], n: usize, first_line: i64) -> String {
    let lines: Vec<&str> = text.lines().collect();
    let score = |l: &str| {
        let low = l.to_lowercase();
        terms.iter().filter(|t| low.contains(t.as_str())).count()
    };
    let best = (0..lines.len()).max_by_key(|&i| (score(lines[i]), std::cmp::Reverse(i))).unwrap_or(0);
    let from = best.saturating_sub(n / 3);
    let to = (from + n).min(lines.len());
    lines[from..to]
        .iter()
        .enumerate()
        .map(|(i, l)| format!("{}: {}", first_line + (from + i) as i64, l.trim_end().chars().take(160).collect::<String>()))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Re-rank hits with Laya's `relevance` profile via the local daemon (auto-started).
pub fn judge(question: &str, hits: &mut Vec<Hit>) -> Result<()> {
    crate::judge::ensure_daemon()?;
    let items: HashMap<String, String> = hits
        .iter()
        .map(|h| {
            let body: String = h.text.chars().take(1500).collect();
            (format!("{}:{}-{}", h.path, h.start, h.end), format!("file: {}\n{}", h.path, body))
        })
        .collect();
    let ranked: Vec<serde_json::Value> = ureq::post(&format!("{}/v1/rank", crate::judge::url()))
        .timeout(std::time::Duration::from_secs(120))
        .send_json(serde_json::json!({"task": question, "items": items}))?
        .into_json()?;
    let scores: HashMap<String, f64> = ranked
        .iter()
        .filter_map(|r| Some((r["id"].as_str()?.to_string(), r["score"].as_f64()?)))
        .collect();
    for h in hits.iter_mut() {
        h.laya = scores.get(&format!("{}:{}-{}", h.path, h.start, h.end)).copied();
    }
    let laya: Vec<Option<f64>> = hits.iter().map(|h| h.laya).collect();
    let order = fuse_judge(&laya, knob("RRF_K", RRF_K), knob("JUDGE_WEIGHT", JUDGE_WEIGHT));
    let sorted: Vec<Hit> = order.into_iter().map(|i| hits[i].clone()).collect();
    *hits = sorted;
    Ok(())
}

/// Fuse rather than replace: keep the retrieval order (name + keyword + semantic signals the judge
/// cannot see) and blend in the judge's order by reciprocal rank. `laya[i]` is the judge score of
/// the hit at retrieval rank `i` (None = not scored, ranked last by the judge). Returns the new
/// order as retrieval indices; ties keep the retrieval order.
pub fn fuse_judge(laya: &[Option<f64>], rrf_k: f64, judge_w: f64) -> Vec<usize> {
    let n = laya.len();
    let mut by_judge: Vec<usize> = (0..n).collect();
    by_judge.sort_by(|&a, &b| laya[b].partial_cmp(&laya[a]).unwrap_or(std::cmp::Ordering::Equal));
    let mut judge_rank = vec![0usize; n];
    for (rank, &i) in by_judge.iter().enumerate() {
        judge_rank[i] = rank;
    }
    let fused: Vec<f64> = (0..n).map(|i| 1.0 / (rrf_k + i as f64) + judge_w / (rrf_k + judge_rank[i] as f64)).collect();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| fused[b].partial_cmp(&fused[a]).unwrap_or(std::cmp::Ordering::Equal));
    order
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::params;
    use std::fs;

    /// A throwaway project plus its index (kept outside the project, like the real cache).
    fn project(files: &[(&str, &str)]) -> (tempfile::TempDir, tempfile::TempDir, Index) {
        let (dir, cache) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
        for (path, body) in files {
            let p = dir.path().join(path);
            fs::create_dir_all(p.parent().unwrap()).unwrap();
            fs::write(p, body).unwrap();
        }
        let mut idx = Index::open_at(dir.path(), &cache.path().join("index.db")).unwrap();
        idx.sync().unwrap();
        (dir, cache, idx)
    }

    fn paths(hits: &[Hit]) -> Vec<&str> {
        hits.iter().map(|h| h.path.as_str()).collect()
    }

    fn set_vec(idx: &Index, file: &str, v: [f32; 2]) {
        let blob: Vec<u8> = v.iter().flat_map(|f| f.to_le_bytes()).collect();
        idx.conn.execute("INSERT OR REPLACE INTO vecs(file, start, vec) VALUES(?1, 1, ?2)", params![file, blob]).unwrap();
    }

    #[test]
    fn file_named_like_the_question_ranks_first() {
        let (_d, _c, idx) = project(&[
            ("src/ContactForm.jsx", "export function ContactForm() {\n  return <form/>;\n}\n"),
            ("src/notes.md", "The contact form is nice.\n"),
            ("src/footer.js", "export const year = 2026;\n"),
        ]);
        let hits = search(&idx, "where is the contact form", 5, 1, 4, None).unwrap();
        assert_eq!(paths(&hits)[..2], ["src/ContactForm.jsx", "src/notes.md"]);
        assert!(hits[0].snippet.starts_with("1: export function ContactForm"));
        assert!(search(&idx, "the of and", 5, 1, 4, None).unwrap().is_empty());
    }

    #[test]
    fn sync_tracks_edits_and_deletions() {
        let (dir, _c, mut idx) = project(&[("a.txt", "alpha\n"), ("b.txt", "beta\n")]);
        let stats = idx.sync().unwrap();
        assert_eq!((stats.scanned, stats.reindexed, stats.removed), (2, 0, 0));
        fs::write(dir.path().join("a.txt"), "gamma gamma\n").unwrap();
        fs::remove_file(dir.path().join("b.txt")).unwrap();
        let stats = idx.sync().unwrap();
        assert_eq!((stats.reindexed, stats.removed), (1, 1));
        assert!(search(&idx, "alpha", 5, 1, 4, None).unwrap().is_empty());
        assert!(search(&idx, "beta", 5, 1, 4, None).unwrap().is_empty());
        assert_eq!(paths(&search(&idx, "gamma", 5, 1, 4, None).unwrap()), ["a.txt"]);
    }

    #[test]
    fn semantic_ranking_uses_vectors_and_sees_new_ones() {
        let (_d, _c, idx) = project(&[("one.txt", "apples\n"), ("two.txt", "pears\n")]);
        set_vec(&idx, "one.txt", [1.0, 0.0]);
        set_vec(&idx, "two.txt", [0.0, 1.0]);
        // No keyword overlap: only the embedding can rank these.
        let hits = search(&idx, "zz", 2, 1, 4, Some(&[0.0, 1.0])).unwrap();
        assert_eq!(paths(&hits), ["two.txt", "one.txt"]);
        assert_eq!(hits[0].text, "pears");

        // Another connection (like `lt index --embed` next to `lt serve`) changes a vector: the
        // cached copy must be refreshed, not reused.
        let other = rusqlite::Connection::open(idx.conn.path().unwrap()).unwrap();
        let blob: Vec<u8> = [0.0f32, -1.0].iter().flat_map(|f| f.to_le_bytes()).collect();
        other.execute("UPDATE vecs SET vec = ?1 WHERE file = 'two.txt'", params![blob]).unwrap();
        assert_eq!(paths(&search(&idx, "zz", 2, 1, 4, Some(&[0.0, 1.0])).unwrap()), ["one.txt", "two.txt"]);
    }

    #[test]
    fn judge_fusion_blends_rather_than_replaces() {
        // The judge agrees with retrieval: order unchanged.
        assert_eq!(fuse_judge(&[Some(3.0), Some(2.0), Some(1.0)], 10.0, 1.0), [0, 1, 2]);
        // The judge prefers #1 over #0: they swap, and ties go to the retrieval order.
        assert_eq!(fuse_judge(&[Some(1.0), Some(3.0), Some(0.5)], 10.0, 1.0), [0, 1, 2]);
        assert_eq!(fuse_judge(&[Some(1.0), Some(3.0), Some(2.0)], 10.0, 1.0), [1, 0, 2]);
        // One strong judge vote cannot lift the last hit over a hit both lists like more.
        assert_eq!(fuse_judge(&[Some(2.0), Some(1.0), Some(0.0), Some(3.0)], 10.0, 1.0), [0, 3, 1, 2]);
        // Weight 0 ignores the judge; unscored hits rank last on the judge side.
        assert_eq!(fuse_judge(&[Some(0.0), Some(9.0)], 10.0, 0.0), [0, 1]);
        assert_eq!(fuse_judge(&[Some(1.0), None, Some(2.0)], 10.0, 1.0), [0, 2, 1]);
        assert!(fuse_judge(&[], 10.0, 1.0).is_empty());
    }
}
