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

/// Reciprocal-rank fusion constant: larger flattens the influence of the very top ranks.
const RRF_K: f64 = 10.0;
/// How much a file/symbol-name match counts relative to a whole-chunk match.
const NAME_WEIGHT: f64 = 0.7;

/// Search the index. Two ranked lists are fused per file: one over everything (path, symbols, body)
/// and one over only path and symbol names, so a file *called* what you asked about beats a file
/// that merely mentions it.
pub fn search(index: &Index, question: &str, k: usize, per_file: usize, snippet_lines: usize) -> Result<Vec<Hit>> {
    let terms = query_terms(question);
    if terms.is_empty() {
        return Ok(vec![]);
    }
    let expr = match_expr(&terms);
    let all = fts_rows(index, &expr, k * per_file * 6)?;
    let names = fts_rows(index, &format!("{{path symbols}} : ({expr})"), 60)?;

    let mut score: HashMap<&str, f64> = HashMap::new();
    for (weight, rows) in [(1.0, &all), (NAME_WEIGHT, &names)] {
        let mut seen = std::collections::HashSet::new();
        for (rank, row) in rows.iter().filter(|r| seen.insert(r.0.as_str())).enumerate() {
            *score.entry(row.0.as_str()).or_insert(0.0) += weight / (RRF_K + rank as f64);
        }
    }
    let mut files: Vec<&str> = score.keys().copied().collect();
    files.sort_by(|a, b| score[b].partial_cmp(&score[a]).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(b)));

    let mut hits = Vec::new();
    for file in files.into_iter().take(k) {
        // Prefer the file's best whole-chunk matches; fall back to its best name match.
        let mut chunks: Vec<&Row> = all.iter().filter(|r| r.0 == file).take(per_file).collect();
        if chunks.is_empty() {
            chunks.extend(names.iter().find(|r| r.0 == file));
        }
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
        .map(|h| (format!("{}:{}-{}", h.path, h.start, h.end), h.text.chars().take(1500).collect()))
        .collect();
    let ranked: Vec<serde_json::Value> = ureq::post(&format!("{}/v1/rank", crate::judge::URL))
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
    hits.sort_by(|a, b| b.laya.partial_cmp(&a.laya).unwrap_or(std::cmp::Ordering::Equal));
    Ok(())
}
