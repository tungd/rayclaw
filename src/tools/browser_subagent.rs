//! Browser tool - spawns rayclaw-browser (WKWebView + page-agent)
//!
//! This tool spawns a native macOS browser app that embeds page-agent.js
//! and communicates via stdio. The browser handles multi-step automation
//! internally, returning results when tasks complete.
//!
//! Architecture:
//!   RayClaw → spawn rayclaw-browser → stdin/stdout → WKWebView → page-agent
//!
//! Design:
//!   - Request/response model (no tool loops)
//!   - page-agent has full DOM context
//!   - WKWebView is system-optimized (memory/battery)
//!   - Profile isolation per session

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::sync::Mutex;
use tracing::{debug, info};

use crate::llm_types::ToolDefinition;
use crate::tools::{auth_context_from_input, schema_object, Tool, ToolResult};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Path to the rayclaw-browser binary.
/// Can be overridden via RAYCLAW_BROWSER_PATH env var.
fn browser_binary_path() -> PathBuf {
    std::env::var("RAYCLAW_BROWSER_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            // Default: look in ~/.local/bin or /usr/local/bin
            let home = std::env::var("HOME").unwrap_or_else(|_| "/".to_string());
            let paths = [
                format!("{}/.local/bin/rayclaw-browser", home),
                "/usr/local/bin/rayclaw-browser".to_string(),
                "/opt/local/bin/rayclaw-browser".to_string(),
                "/opt/homebrew/bin/rayclaw-browser".to_string(),
            ];
            for p in paths {
                if PathBuf::from(&p).exists() {
                    return PathBuf::from(p);
                }
            }
            // Fallback: assume it's on PATH
            PathBuf::from("rayclaw-browser")
        })
}

fn default_timeout_secs() -> u64 {
    600
}

fn extract_first_url(text: &str) -> Option<String> {
    let start = text.find("https://").or_else(|| text.find("http://"))?;
    let candidate = text[start..]
        .split_whitespace()
        .next()?
        .trim_matches(|ch: char| matches!(ch, '"' | '\'' | ')' | ']' | '}' | '>' | ',' | '.'));

    if candidate.is_empty() {
        None
    } else {
        Some(candidate.to_string())
    }
}

// ---------------------------------------------------------------------------
// Browser session (spawned subprocess)
// ---------------------------------------------------------------------------

struct BrowserSessionInner {
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    _child: Child,
    session_id: String,
    current_url: String,
    page_title: String,
}

/// A spawned browser subagent session.
pub struct BrowserSession {
    inner: Mutex<BrowserSessionInner>,
    timeout: Duration,
}

impl BrowserSession {
    /// Spawn a new browser subagent process.
    pub async fn spawn(session_id: &str, timeout_secs: u64) -> Result<Self, String> {
        let binary = browser_binary_path();

        info!(
            "Browser subagent: spawning '{}' --session '{}'",
            binary.display(),
            session_id
        );

        let mut cmd = Command::new(&binary);
        cmd.arg("--session").arg(session_id);
        cmd.env(
            "RAYCLAW_BROWSER_TASK_TIMEOUT_SECS",
            timeout_secs.to_string(),
        );
        cmd.stdin(std::process::Stdio::piped());
        cmd.stdout(std::process::Stdio::piped());
        cmd.stderr(std::process::Stdio::piped());

        let mut child = cmd.spawn().map_err(|e| {
            format!(
                "Failed to spawn browser subagent: {e} (binary: {})",
                binary.display()
            )
        })?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| "Failed to capture browser stdin")?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| "Failed to capture browser stdout")?;

        // Drain stderr to tracing
        if let Some(stderr) = child.stderr.take() {
            let sid = session_id.to_string();
            tokio::spawn(async move {
                let mut reader = BufReader::new(stderr);
                let mut line = String::new();
                loop {
                    line.clear();
                    match reader.read_line(&mut line).await {
                        Ok(0) | Err(_) => break,
                        Ok(_) => {
                            let trimmed = line.trim();
                            if !trimmed.is_empty() {
                                debug!("Browser [{}] stderr: {}", sid, trimmed);
                            }
                        }
                    }
                }
            });
        }

        let session = BrowserSession {
            inner: Mutex::new(BrowserSessionInner {
                stdin,
                stdout: BufReader::new(stdout),
                _child: child,
                session_id: session_id.to_string(),
                current_url: String::new(),
                page_title: String::new(),
            }),
            timeout: Duration::from_secs(timeout_secs),
        };

        // Wait for ready signal
        session.wait_for_ready().await?;

        Ok(session)
    }

    /// Wait for the browser to signal it's ready.
    async fn wait_for_ready(&self) -> Result<(), String> {
        let mut inner = self.inner.lock().await;
        let mut line = String::new();

        let deadline = tokio::time::Instant::now() + Duration::from_secs(10);

        match tokio::time::timeout_at(deadline, inner.stdout.read_line(&mut line)).await {
            Ok(Ok(_)) => {
                let trimmed = line.trim();
                let msg: serde_json::Value = serde_json::from_str(trimmed)
                    .map_err(|_| format!("Invalid ready signal: {}", trimmed))?;
                if msg.get("status").and_then(|s| s.as_str()) == Some("ready") {
                    info!("Browser [{}] ready", inner.session_id);
                    return Ok(());
                }
                Err(format!("Unexpected initial message: {}", trimmed))
            }
            Ok(Err(e)) => Err(format!("Read error waiting for ready: {e}")),
            Err(_) => Err("Timeout waiting for browser ready signal".to_string()),
        }
    }

    /// Send a task to the browser and wait for result.
    pub async fn execute(&self, task: &str) -> Result<BrowserResult, String> {
        let mut inner = self.inner.lock().await;

        // Write task to stdin
        let payload = format!("{}\n", task);
        inner
            .stdin
            .write_all(payload.as_bytes())
            .await
            .map_err(|e| format!("Write error: {e}"))?;
        inner
            .stdin
            .flush()
            .await
            .map_err(|e| format!("Flush error: {e}"))?;

        // Read result from stdout
        let mut line = String::new();
        let deadline = tokio::time::Instant::now() + self.timeout;

        match tokio::time::timeout_at(deadline, inner.stdout.read_line(&mut line)).await {
            Ok(Ok(0)) => Err("Browser process exited unexpectedly".to_string()),
            Ok(Ok(_)) => {
                let trimmed = line.trim();
                let result: BrowserResult = serde_json::from_str(trimmed)
                    .map_err(|e| format!("Failed to parse result: {e} (raw: {})", trimmed))?;

                // Update cached state
                if let Some(url) = result.url.as_ref() {
                    inner.current_url = url.clone();
                }
                if let Some(title) = result.title.as_ref() {
                    inner.page_title = title.clone();
                }

                Ok(result)
            }
            Ok(Err(e)) => Err(format!("Read error: {e}")),
            Err(_) => Err(format!("Timeout after {}s", self.timeout.as_secs())),
        }
    }

    /// Navigate to a URL.
    pub async fn navigate(&self, url: &str) -> Result<BrowserResult, String> {
        self.execute(&format!("navigate {}", url)).await
    }

    /// Get current page status.
    pub async fn status(&self) -> Result<BrowserResult, String> {
        self.execute("status").await
    }

    /// Shut down the browser process.
    pub async fn shutdown(&self) -> Result<(), String> {
        let mut inner = self.inner.lock().await;

        // Send quit command
        let _ = inner.stdin.write_all(b"quit\n").await;
        let _ = inner.stdin.flush().await;

        // Wait for process to exit
        let _ = inner._child.wait().await;

        info!("Browser [{}] shut down", inner.session_id);
        Ok(())
    }

    /// Check if the browser process is still alive.
    pub async fn is_alive(&self) -> bool {
        let mut inner = self.inner.lock().await;
        matches!(inner._child.try_wait(), Ok(None))
    }
}

// ---------------------------------------------------------------------------
// Result type
// ---------------------------------------------------------------------------

/// Result from a browser task execution.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct BrowserResult {
    /// Whether the task succeeded
    #[serde(default)]
    pub success: bool,

    /// Error message if task failed
    #[serde(default)]
    pub error: Option<String>,

    /// Result content from page-agent
    pub result: Option<serde_json::Value>,

    /// Current URL after task
    #[serde(default)]
    pub url: Option<String>,

    /// Page title after task
    #[serde(default)]
    pub title: Option<String>,

    /// Session ID
    #[serde(default)]
    pub session: Option<String>,

    /// Status message
    #[serde(default)]
    pub status: Option<String>,
}

impl BrowserResult {
    fn to_tool_result(&self) -> ToolResult {
        if let Some(err) = &self.error {
            return ToolResult::error(err.clone()).with_error_type("browser_error");
        }

        let content = if let Some(result) = &self.result {
            match result {
                serde_json::Value::String(s) => s.clone(),
                other => serde_json::to_string_pretty(other).unwrap_or_else(|_| other.to_string()),
            }
        } else if let Some(status) = &self.status {
            status.clone()
        } else if let Some(url) = &self.url {
            format!(
                "URL: {}\nTitle: {}",
                url,
                self.title.as_deref().unwrap_or("")
            )
        } else {
            "Task completed".to_string()
        };

        ToolResult::success(content)
    }
}

// ---------------------------------------------------------------------------
// Session manager
// ---------------------------------------------------------------------------

/// Global browser session manager.
pub struct BrowserSessionManager {
    sessions: Mutex<HashMap<String, Arc<BrowserSession>>>,
    /// Map chat_id → session_id
    chat_sessions: Mutex<HashMap<i64, String>>,
}

impl BrowserSessionManager {
    pub fn new() -> Self {
        BrowserSessionManager {
            sessions: Mutex::new(HashMap::new()),
            chat_sessions: Mutex::new(HashMap::new()),
        }
    }

    /// Get or create a session for a chat.
    pub async fn session_for_chat(
        &self,
        chat_id: i64,
        timeout_secs: u64,
    ) -> Result<Arc<BrowserSession>, String> {
        let chat_sessions = self.chat_sessions.lock().await;

        if let Some(session_id) = chat_sessions.get(&chat_id) {
            let sessions = self.sessions.lock().await;
            if let Some(session) = sessions.get(session_id) {
                // Check if still alive
                if session.is_alive().await {
                    return Ok(session.clone());
                }
            }
        }

        // Need to create new session
        drop(chat_sessions);

        let session_id = format!("chat-{}", chat_id);
        let session = BrowserSession::spawn(&session_id, timeout_secs).await?;
        let session = Arc::new(session);

        self.sessions
            .lock()
            .await
            .insert(session_id.clone(), session.clone());
        self.chat_sessions.lock().await.insert(chat_id, session_id);

        Ok(session)
    }

    /// Get an existing session by ID.
    pub async fn get_session(&self, session_id: &str) -> Option<Arc<BrowserSession>> {
        self.sessions.lock().await.get(session_id).cloned()
    }

    /// End a session.
    pub async fn end_session(&self, session_id: &str) -> Result<(), String> {
        let sessions = self.sessions.lock().await;
        if let Some(session) = sessions.get(session_id) {
            session.shutdown().await?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tool implementation
// ---------------------------------------------------------------------------

pub struct BrowserSubagentTool {
    manager: Arc<BrowserSessionManager>,
}

impl BrowserSubagentTool {
    pub fn new() -> Self {
        BrowserSubagentTool {
            manager: Arc::new(BrowserSessionManager::new()),
        }
    }
}

#[async_trait]
impl Tool for BrowserSubagentTool {
    fn name(&self) -> &str {
        "browser"
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "browser".into(),
            description: "Browser automation via RayClaw Browser (rayclaw-browser), a native WKWebView subagent. \
                The browser handles multi-step tasks internally and returns results when complete. \
                No tool loops - one request, one response. \
                \n\n\
                Usage:\n\
                1. First call with a task will spawn the browser and navigate\n\
                2. Subsequent calls reuse the same browser session\n\
                3. Browser maintains cookies/session state across calls\n\
                \n\
                Examples:\n\
                - 'Navigate to github.com and find trending repos'\n\
                - 'Login to example.com with email X and password Y'\n\
                - 'Fill the search form and submit'"
                .into(),
            input_schema: schema_object(
                json!({
                    "task": {
                        "type": "string",
                        "description": "The browser task to execute. Natural language description of what to do on the current page."
                    },
                    "url": {
                        "type": "string",
                        "description": "Optional URL to navigate to before executing the task."
                    },
                    "timeout_secs": {
                        "type": "integer",
                        "description": "Timeout in seconds (default: 600)"
                    }
                }),
                &["task"],
            ),
        }
    }

    async fn execute(&self, input: serde_json::Value) -> ToolResult {
        let task = input.get("task").and_then(|v| v.as_str()).unwrap_or("");

        if task.is_empty() {
            return ToolResult::error("Missing 'task' parameter".to_string());
        }

        let url = input
            .get("url")
            .and_then(|v| v.as_str())
            .filter(|value| !value.trim().is_empty())
            .map(str::to_string)
            .or_else(|| extract_first_url(task));
        let timeout_secs = input
            .get("timeout_secs")
            .and_then(|v| v.as_u64())
            .unwrap_or(default_timeout_secs());

        let auth = auth_context_from_input(&input);
        let chat_id = auth.as_ref().map(|a| a.caller_chat_id).unwrap_or(0);

        // Get or create session
        let session = match self.manager.session_for_chat(chat_id, timeout_secs).await {
            Ok(s) => s,
            Err(e) => return ToolResult::error(e),
        };

        // Navigate if URL provided
        if let Some(url) = url.as_deref() {
            match session.navigate(url).await {
                Ok(result) => {
                    if result.error.is_some() {
                        return result.to_tool_result();
                    }
                }
                Err(e) => return ToolResult::error(format!("Navigation failed: {e}")),
            }
        }

        // Execute task
        match session.execute(task).await {
            Ok(result) => result.to_tool_result(),
            Err(e) => ToolResult::error(e),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_browser_result_parse() {
        let json = r#"{"success": true, "url": "https://example.com", "title": "Example"}"#;
        let result: BrowserResult = serde_json::from_str(json).unwrap();
        assert!(result.success);
        assert_eq!(result.url, Some("https://example.com".to_string()));
    }

    #[test]
    fn test_browser_result_error() {
        let json = r#"{"error": "Page load timeout"}"#;
        let result: BrowserResult = serde_json::from_str(json).unwrap();
        assert!(!result.success);
        assert_eq!(result.error, Some("Page load timeout".to_string()));
    }

    #[test]
    fn test_browser_tool_is_canonical_browser_tool() {
        let tool = BrowserSubagentTool::new();
        assert_eq!(tool.name(), "browser");

        let definition = tool.definition();
        assert_eq!(definition.name, "browser");
        assert!(definition.description.contains("rayclaw-browser"));
        assert!(definition.input_schema["properties"]["task"].is_object());
        assert!(definition.input_schema["properties"]["url"].is_object());
    }

    #[test]
    fn test_extract_first_url_from_task() {
        assert_eq!(
            extract_first_url("Navigate to https://example.com/path, then summarize."),
            Some("https://example.com/path".to_string())
        );
        assert_eq!(
            extract_first_url("Open (http://localhost:3000/foo)."),
            Some("http://localhost:3000/foo".to_string())
        );
        assert_eq!(extract_first_url("no url here"), None);
    }
}
