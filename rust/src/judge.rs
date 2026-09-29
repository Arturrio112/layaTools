//! Talks to the warm Laya daemon (Python, `layatools serve-http`), starting it when needed.

use anyhow::{bail, Result};
use std::os::unix::process::CommandExt;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

pub const URL: &str = "http://127.0.0.1:8765";

pub fn healthy() -> bool {
    ureq::get(&format!("{URL}/health")).timeout(Duration::from_secs(1)).call().is_ok()
}

pub fn ensure_daemon() -> Result<()> {
    if healthy() {
        return Ok(());
    }
    let log = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(std::path::Path::new(&std::env::var("HOME")?).join(".cache/layatools/daemon.log"))?;
    Command::new("layatools")
        .arg("serve-http")
        .stdin(Stdio::null())
        .stdout(log.try_clone()?)
        .stderr(log)
        .process_group(0)
        .spawn()
        .map_err(|e| anyhow::anyhow!("could not start `layatools serve-http` ({e}); is it on PATH?"))?;
    let deadline = Instant::now() + Duration::from_secs(180);
    while Instant::now() < deadline {
        if healthy() {
            return Ok(());
        }
        std::thread::sleep(Duration::from_millis(500));
    }
    bail!("Laya daemon did not start; see ~/.cache/layatools/daemon.log")
}

/// Embed `texts` with the daemon's embedding model. Returns (model name, one vector per text).
pub fn embed(texts: &[String], query: bool) -> Result<(String, Vec<Vec<f32>>)> {
    use base64::Engine;
    let resp: serde_json::Value = ureq::post(&format!("{URL}/v1/embed"))
        .timeout(Duration::from_secs(300))
        .send_json(serde_json::json!({"texts": texts, "query": query}))?
        .into_json()?;
    let dim = resp["dim"].as_u64().unwrap_or(0) as usize;
    let raw = base64::engine::general_purpose::STANDARD.decode(resp["data"].as_str().unwrap_or(""))?;
    if dim == 0 || raw.len() != dim * 4 * texts.len() {
        bail!("bad embedding response: dim={dim}, {} bytes for {} texts", raw.len(), texts.len());
    }
    let vecs = raw
        .chunks_exact(dim * 4)
        .map(|c| c.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect())
        .collect();
    Ok((resp["model"].as_str().unwrap_or("").to_string(), vecs))
}
