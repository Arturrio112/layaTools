//! Per-project SQLite FTS5 index, synced incrementally by (mtime, size) on every use.

use crate::text::expand_for_index;
use anyhow::{Context, Result};
use ignore::WalkBuilder;
use regex::Regex;
use rusqlite::{params, Connection};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

const MAX_FILE_BYTES: u64 = 512 * 1024;
const CHUNK_LINES: usize = 40;
const CHUNK_STRIDE: usize = 32;
const SKIP_EXT: &[&str] = &[
    "png", "jpg", "jpeg", "gif", "webp", "avif", "ico", "svg", "woff", "woff2", "ttf", "otf", "eot",
    "mp4", "webm", "mov", "mp3", "wav", "pdf", "zip", "gz", "lock", "map", "min", "wasm",
];
const SKIP_NAMES: &[&str] = &["package-lock.json", "pnpm-lock.yaml", "yarn.lock", "Cargo.lock"];

pub struct Index {
    pub conn: Connection,
    pub root: PathBuf,
}

pub struct SyncStats {
    pub scanned: usize,
    pub reindexed: usize,
    pub removed: usize,
}

/// Where the index for `root` lives: ~/.cache/layatools/<name>-<hash>/index.db (kept out of the project).
fn db_path(root: &Path) -> Result<PathBuf> {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in root.to_string_lossy().bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    let name = root.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let home = std::env::var("HOME").context("HOME not set")?;
    let dir = PathBuf::from(home).join(".cache/layatools").join(format!("{name}-{h:016x}"));
    std::fs::create_dir_all(&dir)?;
    Ok(dir.join("index.db"))
}

impl Index {
    pub fn open(root: &Path) -> Result<Self> {
        let root = root.canonicalize().with_context(|| format!("no such directory: {}", root.display()))?;
        let conn = Connection::open(db_path(&root)?)?;
        conn.execute_batch(
            "PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;
             CREATE TABLE IF NOT EXISTS files(path TEXT PRIMARY KEY, mtime_ns INTEGER, size INTEGER);
             CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5(
                 path, symbols, body,
                 file UNINDEXED, start UNINDEXED, end UNINDEXED, text UNINDEXED,
                 tokenize='porter unicode61');",
        )?;
        Ok(Self { conn, root })
    }

    /// Bring the index up to date with the working tree. Cheap when nothing changed (stat only).
    pub fn sync(&mut self) -> Result<SyncStats> {
        let known: HashMap<String, (i64, i64)> = {
            let mut stmt = self.conn.prepare("SELECT path, mtime_ns, size FROM files")?;
            let rows = stmt.query_map([], |r| Ok((r.get::<_, String>(0)?, (r.get(1)?, r.get(2)?))))?;
            rows.collect::<rusqlite::Result<_>>()?
        };
        let symbol_res = symbol_regexes();
        let mut seen = std::collections::HashSet::new();
        let (mut scanned, mut reindexed) = (0, 0);

        let tx = self.conn.transaction()?;
        for entry in WalkBuilder::new(&self.root).hidden(true).git_ignore(true).build().flatten() {
            let path = entry.path();
            if !entry.file_type().map_or(false, |t| t.is_file()) || skip(path) {
                continue;
            }
            let Ok(meta) = entry.metadata() else { continue };
            if meta.len() > MAX_FILE_BYTES {
                continue;
            }
            let rel = path.strip_prefix(&self.root).unwrap_or(path).to_string_lossy().into_owned();
            let mtime = meta
                .modified()
                .ok()
                .and_then(|m| m.duration_since(UNIX_EPOCH).ok())
                .map_or(0, |d| d.as_nanos() as i64);
            let size = meta.len() as i64;
            scanned += 1;
            seen.insert(rel.clone());
            if known.get(&rel) == Some(&(mtime, size)) {
                continue;
            }
            let Ok(bytes) = std::fs::read(path) else { continue };
            if bytes[..bytes.len().min(8000)].contains(&0) {
                continue; // binary
            }
            let content = String::from_utf8_lossy(&bytes);
            tx.execute("DELETE FROM chunks WHERE file = ?1", params![rel])?;
            index_file(&tx, &rel, &content, &symbol_res)?;
            tx.execute(
                "INSERT INTO files(path, mtime_ns, size) VALUES(?1, ?2, ?3)
                 ON CONFLICT(path) DO UPDATE SET mtime_ns=?2, size=?3",
                params![rel, mtime, size],
            )?;
            reindexed += 1;
        }
        let mut removed = 0;
        for path in known.keys().filter(|p| !seen.contains(*p)) {
            tx.execute("DELETE FROM chunks WHERE file = ?1", params![path])?;
            tx.execute("DELETE FROM files WHERE path = ?1", params![path])?;
            removed += 1;
        }
        tx.commit()?;
        Ok(SyncStats { scanned, reindexed, removed })
    }
}

fn skip(path: &Path) -> bool {
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
    if SKIP_NAMES.contains(&name) || name.ends_with(".min.js") || name.ends_with(".min.css") {
        return true;
    }
    path.extension()
        .and_then(|e| e.to_str())
        .map_or(false, |e| SKIP_EXT.contains(&e.to_ascii_lowercase().as_str()))
}

fn symbol_regexes() -> Vec<Regex> {
    [
        r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?(?:function\*?|class|interface|type|enum|const|let|var)\s+([A-Za-z_$][\w$]*)",
        r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?(?:fn|struct|enum|trait|mod|static|type|const)\s+([A-Za-z_]\w*)",
        r"^\s*(?:async\s+)?(?:def|class)\s+(\w+)",
        r"^#{1,6}\s+(.+?)\s*$",
    ]
    .iter()
    .map(|p| Regex::new(p).expect("valid symbol regex"))
    .collect()
}

fn index_file(tx: &rusqlite::Transaction, rel: &str, content: &str, symbol_res: &[Regex]) -> Result<()> {
    let lines: Vec<&str> = content.lines().collect();
    if lines.is_empty() {
        return Ok(());
    }
    let path_terms = expand_for_index(&rel.replace(['/', '.'], " "));
    let mut start = 0;
    while start < lines.len() {
        let end = (start + CHUNK_LINES).min(lines.len());
        let window = &lines[start..end];
        let mut symbols = Vec::new();
        for line in window {
            for re in symbol_res {
                if let Some(c) = re.captures(line) {
                    symbols.push(c[1].to_string());
                    break;
                }
            }
        }
        let text = window.join("\n");
        tx.execute(
            "INSERT INTO chunks(path, symbols, body, file, start, end, text) VALUES(?1,?2,?3,?4,?5,?6,?7)",
            params![
                path_terms,
                expand_for_index(&symbols.join(" ")),
                expand_for_index(&text),
                rel,
                (start + 1) as i64,
                end as i64,
                text
            ],
        )?;
        if end == lines.len() {
            break;
        }
        start += CHUNK_STRIDE;
    }
    Ok(())
}
