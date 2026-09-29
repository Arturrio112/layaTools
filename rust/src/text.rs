//! Tokenisation shared by indexing and querying, so both sides split identifiers the same way.

use std::collections::HashSet;

const STOPWORDS: &[&str] = &[
    "a", "an", "the", "is", "are", "was", "were", "be", "to", "of", "in", "on", "for", "and", "or",
    "how", "what", "where", "which", "who", "why", "when", "do", "does", "did", "i", "we", "it",
    "this", "that", "with", "from", "by", "as", "at", "can", "should", "there", "any", "all",
    "find", "show", "get", "me", "my", "our", "code", "file", "files", "used", "use", "using",
    "list", "defined", "define", "configured", "configure", "come", "comes", "served", "make", "made",
    "work", "works", "happen", "happens", "need", "called", "name", "named", "live", "yet", "each",
    "not", "about", "into", "so", "if", "no", "then", "than",
];

/// Split `fooBarBaz`, `foo_bar`, `foo-bar`, `Foo2Bar` into lowercase word parts.
pub fn split_identifier(word: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut cur = String::new();
    let mut prev: Option<char> = None;
    for c in word.chars() {
        if !c.is_alphanumeric() {
            if !cur.is_empty() {
                parts.push(std::mem::take(&mut cur));
            }
            prev = None;
            continue;
        }
        if let Some(p) = prev {
            let boundary = (p.is_lowercase() && c.is_uppercase())
                || (p.is_alphabetic() && c.is_ascii_digit())
                || (p.is_ascii_digit() && c.is_alphabetic());
            if boundary && !cur.is_empty() {
                parts.push(std::mem::take(&mut cur));
            }
        }
        cur.extend(c.to_lowercase());
        prev = Some(c);
    }
    if !cur.is_empty() {
        parts.push(cur);
    }
    parts
}

/// Expand text for indexing: original text plus the split parts of every long identifier, so a
/// query for "contact form" matches `ContactForm` and `contact_form`.
pub fn expand_for_index(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + text.len() / 4);
    out.push_str(text);
    out.push('\n');
    for word in text.split(|c: char| !(c.is_alphanumeric() || c == '_' || c == '-' || c == '$')) {
        if word.len() < 4 {
            continue;
        }
        let parts = split_identifier(word);
        if parts.len() > 1 {
            out.push_str(&parts.join(" "));
            out.push(' ');
        }
    }
    out
}

/// Query terms: split identifiers, drop stopwords and 1-char tokens, dedupe, keep order.
pub fn query_terms(question: &str) -> Vec<String> {
    let stop: HashSet<&str> = STOPWORDS.iter().copied().collect();
    let mut seen = HashSet::new();
    let mut terms = Vec::new();
    for word in question.split(|c: char| !(c.is_alphanumeric() || c == '_' || c == '-' || c == '$')) {
        for part in split_identifier(word) {
            if part.chars().count() < 2 || stop.contains(part.as_str()) {
                continue;
            }
            if seen.insert(part.clone()) {
                terms.push(part);
            }
        }
    }
    terms
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_identifiers() {
        assert_eq!(split_identifier("ContactForm"), ["contact", "form"]);
        assert_eq!(split_identifier("contact_form-v2"), ["contact", "form", "v", "2"]);
    }

    #[test]
    fn query_drops_stopwords() {
        assert_eq!(query_terms("where is the contactForm handled?"), ["contact", "form", "handled"]);
    }
}
