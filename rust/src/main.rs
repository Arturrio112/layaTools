mod index;
mod judge;
mod search;
mod server;
mod text;

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "lt", about = "Search a codebase before exploring it. Prints path:lines and a snippet per hit.")]
struct Cli {
    /// Project directory (default: current directory)
    #[arg(short = 'C', long, global = true, default_value = ".")]
    dir: PathBuf,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Search the project with a question or keywords
    Search {
        question: String,
        /// Number of hits
        #[arg(short, default_value_t = 8)]
        k: usize,
        /// Max chunks per file
        #[arg(long, default_value_t = 1)]
        per_file: usize,
        /// Snippet lines per hit
        #[arg(long, default_value_t = 4)]
        lines: usize,
        /// Re-rank with the Laya relevance judge (needs the layatools daemon)
        #[arg(long)]
        judge: bool,
        #[arg(long)]
        json: bool,
    },
    /// Re-sync the index and print stats
    Index,
    /// Serve searches over local HTTP (keeps indexes in memory)
    Serve {
        #[arg(long, default_value_t = 8766)]
        port: u16,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    if let Cmd::Serve { port } = cli.cmd {
        return server::serve(port);
    }
    let mut idx = index::Index::open(&cli.dir)?;
    let stats = idx.sync()?;
    match cli.cmd {
        Cmd::Serve { .. } => unreachable!("handled above"),
        Cmd::Index => println!(
            "{} files scanned, {} (re)indexed, {} removed",
            stats.scanned, stats.reindexed, stats.removed
        ),
        Cmd::Search { question, k, per_file, lines, judge, json } => {
            // Judge only helps if it can reorder something: fetch extra candidates first.
            let fetch = if judge { k * 2 } else { k };
            let mut hits = search::search(&idx, &question, fetch, per_file, lines)?;
            if judge && !hits.is_empty() {
                search::judge(&question, &mut hits)?;
                hits.truncate(k);
            }
            if json {
                println!("{}", serde_json::to_string(&hits)?);
            } else if hits.is_empty() {
                println!("no matches");
            } else {
                for h in &hits {
                    match h.laya {
                        Some(l) => println!("{}:{}-{}  laya={:.1}", h.path, h.start, h.end, l),
                        None => println!("{}:{}-{}", h.path, h.start, h.end),
                    }
                    println!("{}", h.snippet);
                }
            }
        }
    }
    Ok(())
}
