//! Talks to the warm Laya daemon (Python, `layatools serve-http`), starting it when needed.

use anyhow::{bail, Result};
use std::os::unix::process::CommandExt;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

pub const URL: &str = "http://127.0.0.1:8765";

fn healthy() -> bool {
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
