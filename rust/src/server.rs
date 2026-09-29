//! `lt serve`: keeps indexes open in memory and answers searches over local HTTP.
//!
//!   GET  /health
//!   POST /v1/search  {"dir": "/abs/project", "question": "...", "k": 8, "per_file": 1, "lines": 4, "judge": false}

use crate::{index::Index, search};
use anyhow::Result;
use serde::Deserialize;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tiny_http::{Header, Method, Request, Response, Server};

#[derive(Deserialize)]
struct SearchReq {
    dir: PathBuf,
    question: String,
    #[serde(default = "d_k")]
    k: usize,
    #[serde(default = "d_one")]
    per_file: usize,
    #[serde(default = "d_lines")]
    lines: usize,
    #[serde(default)]
    judge: bool,
}
fn d_k() -> usize { 8 }
fn d_one() -> usize { 1 }
fn d_lines() -> usize { 4 }

type Indexes = Arc<Mutex<HashMap<PathBuf, Index>>>;

pub fn serve(port: u16) -> Result<()> {
    // Bind loopback only: this exposes the contents of whatever directory a caller names.
    let server = Server::http(("127.0.0.1", port)).map_err(|e| anyhow::anyhow!("{e}"))?;
    eprintln!("lt serving on http://127.0.0.1:{port}");
    let indexes: Indexes = Arc::default();
    for req in server.incoming_requests() {
        let indexes = Arc::clone(&indexes);
        std::thread::spawn(move || handle(req, indexes));
    }
    Ok(())
}

fn respond(req: Request, code: u16, body: serde_json::Value) {
    let json = Header::from_bytes("Content-Type", "application/json").expect("static header");
    let _ = req.respond(Response::from_string(body.to_string()).with_status_code(code).with_header(json));
}

fn handle(mut req: Request, indexes: Indexes) {
    match (req.method().clone(), req.url().to_string().as_str()) {
        (Method::Get, "/health") => respond(req, 200, serde_json::json!({"ok": true})),
        (Method::Post, "/v1/search") => {
            let mut body = String::new();
            let _ = req.as_reader().read_to_string(&mut body);
            match serde_json::from_str::<SearchReq>(&body)
                .map_err(anyhow::Error::from)
                .and_then(|r| run(&indexes, r))
            {
                Ok(hits) => respond(req, 200, hits),
                Err(e) => respond(req, 400, serde_json::json!({"error": e.to_string()})),
            }
        }
        _ => respond(req, 404, serde_json::json!({"error": "not found"})),
    }
}

fn run(indexes: &Indexes, r: SearchReq) -> Result<serde_json::Value> {
    let root = r.dir.canonicalize()?;
    let mut hits = {
        // Held only for the local index work; the (slow) judge call below runs unlocked.
        let mut map = indexes.lock().map_err(|_| anyhow::anyhow!("index lock poisoned"))?;
        if !map.contains_key(&root) {
            map.insert(root.clone(), Index::open(&root)?);
        }
        let idx = map.get_mut(&root).expect("just inserted");
        idx.sync()?;
        search::search(idx, &r.question, if r.judge { r.k * 2 } else { r.k }, r.per_file, r.lines)?
    };
    if r.judge && !hits.is_empty() {
        search::judge(&r.question, &mut hits)?;
        hits.truncate(r.k);
    }
    Ok(serde_json::to_value(hits)?)
}
