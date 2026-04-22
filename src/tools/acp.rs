use std::future::Future;
use std::pin::Pin;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::OnceLock;

use crate::acp::{AcpManager, AcpProgressSummary, AcpPromptResult, JobCompletionCallback};
use crate::db::{call_blocking, Database, Memory, StoredMessage};
use crate::llm_types::ToolDefinition;
use async_trait::async_trait;
use regex::Regex;
use serde_json::json;

use super::{auth_context_from_input, schema_object, Tool, ToolResult};

/// Callback type for sending a notification message to a chat.
pub type NotifyFn =
    Arc<dyn Fn(i64, String) -> Pin<Box<dyn Future<Output = ()> + Send>> + Send + Sync>;

fn progress_callback_from_notify(notify: Option<&NotifyFn>) -> Option<JobCompletionCallback> {
    notify.cloned().map(|notify| {
        Arc::new(move |chat_id: i64, text: String| notify(chat_id, text)) as JobCompletionCallback
    })
}

fn truncate_preview(text: &str, max_chars: usize) -> String {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return "(no visible agent text captured)".to_string();
    }

    let mut preview = trimmed.chars().take(max_chars).collect::<String>();
    if trimmed.chars().count() > max_chars {
        preview.push_str("...");
    }
    preview
}

fn direct_delivery_completion_message(tool_name: &str, result: &AcpPromptResult) -> String {
    let preview = truncate_preview(&result.latest_message_text(), 240);
    let file_note = if result.files_changed.is_empty() {
        "No file changes were reported.".to_string()
    } else {
        format!(
            "Reported file changes: {}.",
            result
                .files_changed
                .iter()
                .take(5)
                .cloned()
                .collect::<Vec<_>>()
                .join(", ")
        )
    };
    let reset_note = if result.context_reset {
        " The ACP session was restarted before this run, so earlier ACP-only context was reset."
    } else {
        ""
    };

    format!(
        "ACP task completed successfully. The coding agent's visible response was already delivered directly to the user in chat. Treat this request as complete and do not call `{tool_name}` again for the same instruction unless the user explicitly asks for more work.{reset_note} Latest agent message preview: {preview}. {file_note}"
    )
}

async fn prompt_with_progress_updates(
    manager: &Arc<AcpManager>,
    session_id: &str,
    message: &str,
    timeout_secs: Option<u64>,
    chat_id: Option<i64>,
    progress_callback: Option<JobCompletionCallback>,
) -> Result<(AcpPromptResult, AcpProgressSummary), String> {
    let (progress_tx, progress_handle) = match (chat_id, progress_callback) {
        (Some(cid), Some(cb)) => {
            let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<crate::acp::AcpProgressEvent>();
            let handle = crate::acp::spawn_progress_forwarder(rx, cid, cb);
            (Some(tx), Some(handle))
        }
        _ => (None, None),
    };

    let result = manager
        .prompt(session_id, message, timeout_secs, progress_tx.as_ref())
        .await;
    drop(progress_tx);
    let progress_summary = if let Some(handle) = progress_handle {
        handle.await.ok().unwrap_or_default()
    } else {
        AcpProgressSummary::default()
    };
    result.map(|result| (result, progress_summary))
}

/// Build all ACP tools sharing a single AcpManager.
pub fn make_acp_tools(manager: Arc<AcpManager>) -> Vec<Box<dyn Tool>> {
    make_acp_tools_with_callback(manager, None, None, None)
}

/// Build all ACP tools with optional job completion and notification callbacks.
pub fn make_acp_tools_with_callback(
    manager: Arc<AcpManager>,
    db: Option<Arc<Database>>,
    on_job_complete: Option<JobCompletionCallback>,
    notify: Option<NotifyFn>,
) -> Vec<Box<dyn Tool>> {
    let tools: Vec<Box<dyn Tool>> = vec![
        Box::new(AcpCodingTool::new(
            manager.clone(),
            db.clone(),
            on_job_complete.clone(),
            notify.clone(),
        )),
        Box::new(AcpNewSessionTool::new(manager.clone(), notify.clone())),
        Box::new(AcpPromptTool::new(manager.clone(), notify.clone())),
        Box::new(AcpEndSessionTool::new(manager.clone())),
        Box::new(AcpListSessionsTool::new(manager.clone())),
        Box::new(AcpSubmitJobTool::new(
            manager.clone(),
            on_job_complete,
            notify,
        )),
        Box::new(AcpJobStatusTool::new(manager)),
    ];
    tools
}

// ---------------------------------------------------------------------------
// acp_coding — high-level unified tool with auto session management
// ---------------------------------------------------------------------------

struct AcpCodingTool {
    manager: Arc<AcpManager>,
    db: Option<Arc<Database>>,
    on_complete: Option<JobCompletionCallback>,
    notify: Option<NotifyFn>,
}

impl AcpCodingTool {
    fn new(
        manager: Arc<AcpManager>,
        db: Option<Arc<Database>>,
        on_complete: Option<JobCompletionCallback>,
        notify: Option<NotifyFn>,
    ) -> Self {
        Self {
            manager,
            db,
            on_complete,
            notify,
        }
    }

    async fn send_notify(&self, chat_id: i64, text: &str) {
        if let Some(ref notify) = self.notify {
            notify(chat_id, text.to_string()).await;
        }
    }

    async fn resolve_agent(&self, input: &serde_json::Value, message: &str) -> String {
        if let Some(agent) = input.get("agent").and_then(|v| v.as_str()) {
            let agent = agent.trim();
            if !agent.is_empty() {
                return agent.to_lowercase();
            }
        }

        let Some(chat_id) = auth_context_from_input(input).map(|ctx| ctx.caller_chat_id) else {
            return "claude".to_string();
        };
        let Some(db) = &self.db else {
            return "claude".to_string();
        };

        let workspace = input
            .get("workspace")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_lowercase();
        let message = message.to_lowercase();
        let context = format!("{workspace}\n{message}");
        let db = db.clone();

        let memories =
            match call_blocking(db, move |db| db.get_memories_for_context(chat_id, 100)).await {
                Ok(mems) => mems,
                Err(_) => return "claude".to_string(),
            };

        infer_agent_from_memories(chat_id, &context, &memories)
            .unwrap_or_else(|| "claude".to_string())
    }

    async fn resolve_workspace(
        &self,
        input: &serde_json::Value,
        message: &str,
    ) -> Option<String> {
        if let Some(workspace) = input.get("workspace").and_then(|v| v.as_str()) {
            let workspace = workspace.trim();
            if !workspace.is_empty() {
                return Some(expand_home_path(workspace).to_string_lossy().to_string());
            }
        }

        let Some(chat_id) = auth_context_from_input(input).map(|ctx| ctx.caller_chat_id) else {
            return None;
        };
        let Some(db) = &self.db else {
            return None;
        };

        let db = db.clone();
        let recent = call_blocking(db, move |db| db.get_recent_messages(chat_id, 24))
            .await
            .ok()?;

        infer_workspace_from_recent_messages(message, &recent)
    }
}

fn infer_agent_from_memories(chat_id: i64, context: &str, memories: &[Memory]) -> Option<String> {
    let chat_specific: Vec<String> = memories
        .iter()
        .filter(|m| m.chat_id == Some(chat_id))
        .map(|m| m.content.to_lowercase())
        .collect();

    let chat_prefers_codex = chat_specific.iter().any(|m| m.contains("codex"));
    let chat_prefers_claude = chat_specific.iter().any(|m| m.contains("claude"));
    if chat_prefers_codex && !chat_prefers_claude {
        return Some("codex".to_string());
    }
    if chat_prefers_claude && !chat_prefers_codex {
        return Some("claude".to_string());
    }

    let all_memories = memories
        .iter()
        .map(|m| m.content.to_lowercase())
        .collect::<Vec<_>>();

    let matches_context = |keywords: &[&str]| keywords.iter().any(|kw| context.contains(kw));
    let memory_has = |agent: &str, keywords: &[&str]| {
        all_memories
            .iter()
            .any(|m| m.contains(agent) && keywords.iter().any(|kw| m.contains(kw)))
    };

    if matches_context(&[
        "careai",
        "school",
        "/personal/",
        "personal project",
        "school project",
    ]) && memory_has("codex", &["careai", "school", "personal"])
    {
        return Some("codex".to_string());
    }

    if matches_context(&["/work/", "work project", "professional", " bw ", " rg "])
        && memory_has("claude", &["work", "professional"])
    {
        return Some("claude".to_string());
    }

    None
}

fn expand_home_path(path: &str) -> PathBuf {
    if path == "~" {
        if let Some(home) = std::env::var_os("HOME") {
            return PathBuf::from(home);
        }
    } else if let Some(rest) = path.strip_prefix("~/") {
        if let Some(home) = std::env::var_os("HOME") {
            return PathBuf::from(home).join(rest);
        }
    }
    PathBuf::from(path)
}

fn workspace_candidate_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r#"(?P<path>(?:~|/)[^\s`"'<>()]+)"#).expect("workspace regex must compile")
    })
}

fn project_root_for_path(candidate: &Path) -> Option<PathBuf> {
    let existing = if candidate.is_dir() {
        candidate.to_path_buf()
    } else if candidate.is_file() {
        candidate.parent()?.to_path_buf()
    } else {
        return None;
    };

    let mut best: Option<(u8, PathBuf)> = None;
    for ancestor in existing.ancestors() {
        let priority = if ancestor.join(".git").exists() {
            4
        } else if ancestor.join("pnpm-workspace.yaml").exists() {
            3
        } else if ancestor.join("Cargo.toml").exists() || ancestor.join("pyproject.toml").exists()
        {
            2
        } else if ancestor.join("package.json").exists() {
            1
        } else {
            0
        };

        if priority > 0 {
            let should_replace = best
                .as_ref()
                .map(|(best_priority, _)| priority >= *best_priority)
                .unwrap_or(true);
            if should_replace {
                best = Some((priority, ancestor.to_path_buf()));
            }
        }
    }

    Some(best.map(|(_, path)| path).unwrap_or(existing))
}

fn normalize_workspace_candidate(raw: &str) -> Option<String> {
    let cleaned = raw.trim_end_matches(&['.', ',', ';', ':', ')', ']', '}'][..]);
    let expanded = expand_home_path(cleaned);
    let workspace = project_root_for_path(&expanded)?;
    let canonical = workspace.canonicalize().unwrap_or(workspace);
    Some(canonical.to_string_lossy().to_string())
}

fn infer_workspace_from_recent_messages(message: &str, recent: &[StoredMessage]) -> Option<String> {
    for source in std::iter::once(message).chain(recent.iter().rev().map(|m| m.content.as_str())) {
        for captures in workspace_candidate_regex().captures_iter(source) {
            let Some(candidate) = captures.name("path") else {
                continue;
            };
            if let Some(workspace) = normalize_workspace_candidate(candidate.as_str()) {
                return Some(workspace);
            }
        }
    }

    None
}

fn workspace_matches(requested: Option<&str>, actual: &str) -> bool {
    let Some(requested) = requested else {
        return true;
    };
    let requested = normalize_workspace_candidate(requested).unwrap_or_else(|| requested.to_string());
    let actual = normalize_workspace_candidate(actual).unwrap_or_else(|| actual.to_string());
    requested == actual
}

#[async_trait]
impl Tool for AcpCodingTool {
    fn name(&self) -> &str {
        "acp_coding"
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "acp_coding".into(),
            description: "Delegate a coding task to an external AI coding agent (for example Codex or Claude Code). \
                Prefer this for repository/project coding work instead of long direct bash/read_file exploration loops. \
                Automatically manages sessions: reuses existing session for the chat or creates a new one. \
                If agent is omitted, it resolves the best agent from chat memory/context. \
                Sends immediate notification to the user, then executes the task. \
                For quick tasks the result is returned directly. \
                If the agent response is streamed directly to the chat, treat that as completion and do not retry the same task automatically. \
                Set async=true for long-running tasks to get a job_id and receive results via push notification."
                .into(),
            input_schema: schema_object(
                json!({
                    "message": {
                        "type": "string",
                        "description": "The coding task or instruction to send to the agent"
                    },
                    "agent": {
                        "type": "string",
                        "description": "Optional agent override. If omitted, RayClaw chooses from chat memory/context and falls back to Claude."
                    },
                    "workspace": {
                        "type": "string",
                        "description": "Working directory for the agent (default: agent's configured workspace)"
                    },
                    "async": {
                        "type": "boolean",
                        "description": "If true, submit as async job and return job_id immediately. Results are pushed to chat when done. Use for tasks that may take > 2 minutes."
                    },
                    "timeout_secs": {
                        "type": "integer",
                        "description": "Max seconds to wait (sync mode only). Default: 300"
                    }
                }),
                &["message"],
            ),
        }
    }

    async fn execute(&self, input: serde_json::Value) -> ToolResult {
        let message = match input.get("message").and_then(|v| v.as_str()) {
            Some(m) => m,
            None => return ToolResult::error("Missing required parameter: message".into()),
        };

        let workspace = input.get("workspace").and_then(|v| v.as_str());
        let is_async = input
            .get("async")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let timeout_secs = input.get("timeout_secs").and_then(|v| v.as_u64());

        let chat_id = auth_context_from_input(&input).map(|ctx| ctx.caller_chat_id);
        let agent = self.resolve_agent(&input, message).await;
        let resolved_workspace = self.resolve_workspace(&input, message).await;

        // Step 1: Try to reuse existing session for this chat
        let session_id = if let Some(cid) = chat_id {
            if let Some(existing) = self.manager.chat_session(cid).await {
                // Reuse only if the bound session is still alive and matches the
                // resolved agent for this task.
                let sessions = self.manager.list_sessions().await;
                if let Some(summary) = sessions.iter().find(|s| s.session_id == existing) {
                    if summary.agent_id == agent
                        && workspace_matches(
                            resolved_workspace.as_deref(),
                            &summary.workspace,
                        )
                    {
                        Some(existing)
                    } else {
                        let _ = self.manager.end_session(&existing).await;
                        self.manager.unbind_chat(cid).await;
                        None
                    }
                } else {
                    // Stale binding, clear it.
                    self.manager.unbind_chat(cid).await;
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        // Step 2: Create new session if needed
        let session_id = match session_id {
            Some(sid) => {
                if let Some(cid) = chat_id {
                    self.send_notify(
                        cid,
                        &format!("♻️ Reusing coding agent session ({agent}), executing task..."),
                    )
                    .await;
                }
                sid
            }
            None => {
                if let Some(cid) = chat_id {
                    self.send_notify(
                        cid,
                        &format!("🚀 Starting {agent} coding agent, please wait..."),
                    )
                    .await;
                }

                let workspace = resolved_workspace.as_deref().or(workspace);
                match self.manager.new_session(&agent, workspace, None).await {
                    Ok(info) => {
                        if let Some(cid) = chat_id {
                            self.manager.bind_chat(cid, &info.session_id).await;
                            self.send_notify(
                                cid,
                                &format!(
                                    "✅ {agent} session started ({})\nWorkspace: {}\nExecuting task...",
                                    &info.session_id[..8],
                                    info.workspace
                                ),
                            )
                            .await;
                        }
                        info.session_id
                    }
                    Err(e) => {
                        return ToolResult::error(format!("Failed to start coding agent: {e}"))
                            .with_error_type("acp_error");
                    }
                }
            }
        };

        // Step 3: Execute task
        if is_async {
            // Async mode — submit job and return immediately
            match self
                .manager
                .submit_job(
                    &session_id,
                    message,
                    timeout_secs,
                    chat_id,
                    self.on_complete.clone(),
                    progress_callback_from_notify(self.notify.as_ref()),
                )
                .await
            {
                Ok(job_id) => ToolResult::success(
                    json!({
                        "mode": "async",
                        "job_id": job_id,
                        "session_id": session_id,
                        "agent": agent,
                        "status": "submitted",
                        "message": "Task submitted. Results will be pushed to the chat when complete."
                    })
                    .to_string(),
                ),
                Err(e) => ToolResult::error(format!("Failed to submit async job: {e}"))
                    .with_error_type("acp_error"),
            }
        } else {
            // Sync mode — wait for result
            match prompt_with_progress_updates(
                &self.manager,
                &session_id,
                message,
                timeout_secs,
                chat_id,
                progress_callback_from_notify(self.notify.as_ref()),
            )
            .await
            {
                Ok((result, progress_summary)) => {
                    if progress_summary.forwarded_agent_text {
                        ToolResult::success(direct_delivery_completion_message(
                            "acp_coding",
                            &result,
                        ))
                    } else {
                        ToolResult::success(result.forwarded_text())
                    }
                }
                Err(e) => ToolResult::error(format!("Coding agent error: {e}"))
                    .with_error_type("acp_error"),
            }
        }
    }
}

// ---------------------------------------------------------------------------
// acp_new_session (low-level — prefer acp_coding for most use cases)
// ---------------------------------------------------------------------------

struct AcpNewSessionTool {
    manager: Arc<AcpManager>,
    notify: Option<NotifyFn>,
}

impl AcpNewSessionTool {
    fn new(manager: Arc<AcpManager>, notify: Option<NotifyFn>) -> Self {
        Self { manager, notify }
    }
}

#[async_trait]
impl Tool for AcpNewSessionTool {
    fn name(&self) -> &str {
        "acp_new_session"
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "acp_new_session".into(),
            description: "Low-level: create a new ACP agent session. \
                PREFER acp_coding instead — it handles session management automatically. \
                Only use this if you need explicit control over session lifecycle. \
                Returns a session_id to use with acp_prompt and acp_end_session."
                .into(),
            input_schema: schema_object(
                json!({
                    "agent": {
                        "type": "string",
                        "description": "Agent name from acp.json config (e.g. \"claude\")"
                    },
                    "workspace": {
                        "type": "string",
                        "description": "Working directory for the agent. Defaults to the agent's configured workspace."
                    },
                    "auto_approve": {
                        "type": "boolean",
                        "description": "Auto-approve the agent's tool calls. Defaults to the config setting."
                    }
                }),
                &["agent"],
            ),
        }
    }

    async fn execute(&self, input: serde_json::Value) -> ToolResult {
        let agent = match input.get("agent").and_then(|v| v.as_str()) {
            Some(a) => a,
            None => return ToolResult::error("Missing required parameter: agent".into()),
        };

        let workspace = input.get("workspace").and_then(|v| v.as_str());
        let auto_approve = input.get("auto_approve").and_then(|v| v.as_bool());
        let chat_id = auth_context_from_input(&input).map(|ctx| ctx.caller_chat_id);

        // Notify user that session is starting
        if let (Some(cid), Some(ref notify)) = (chat_id, &self.notify) {
            notify(
                cid,
                format!("🚀 Starting {agent} coding agent, please wait..."),
            )
            .await;
        }

        match self
            .manager
            .new_session(agent, workspace, auto_approve)
            .await
        {
            Ok(info) => {
                // Notify user session is ready
                if let (Some(cid), Some(ref notify)) = (chat_id, &self.notify) {
                    notify(
                        cid,
                        format!(
                            "✅ {agent} session started ({})\nWorkspace: {}",
                            &info.session_id[..8.min(info.session_id.len())],
                            info.workspace
                        ),
                    )
                    .await;
                }

                ToolResult::success(
                    json!({
                        "session_id": info.session_id,
                        "agent": info.agent_id,
                        "workspace": info.workspace,
                        "status": "active"
                    })
                    .to_string(),
                )
            }
            Err(e) => ToolResult::error(format!("Failed to create ACP session: {e}"))
                .with_error_type("acp_error"),
        }
    }
}

// ---------------------------------------------------------------------------
// acp_prompt
// ---------------------------------------------------------------------------

struct AcpPromptTool {
    manager: Arc<AcpManager>,
    notify: Option<NotifyFn>,
}

impl AcpPromptTool {
    fn new(manager: Arc<AcpManager>, notify: Option<NotifyFn>) -> Self {
        Self { manager, notify }
    }
}

#[async_trait]
impl Tool for AcpPromptTool {
    fn name(&self) -> &str {
        "acp_prompt"
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "acp_prompt".into(),
            description: "Low-level: send a prompt to an existing ACP session. \
                PREFER acp_coding instead — it handles session creation and notifications automatically. \
                Only use this for multi-turn interactions on an already-open session."
                .into(),
            input_schema: schema_object(
                json!({
                    "session_id": {
                        "type": "string",
                        "description": "Session ID returned by acp_new_session"
                    },
                    "message": {
                        "type": "string",
                        "description": "The coding task or instruction to send to the agent"
                    },
                    "timeout_secs": {
                        "type": "integer",
                        "description": "Max seconds to wait for completion. Defaults to config value (300s)."
                    }
                }),
                &["session_id", "message"],
            ),
        }
    }

    async fn execute(&self, input: serde_json::Value) -> ToolResult {
        let session_id = match input.get("session_id").and_then(|v| v.as_str()) {
            Some(s) => s,
            None => return ToolResult::error("Missing required parameter: session_id".into()),
        };

        let message = match input.get("message").and_then(|v| v.as_str()) {
            Some(m) => m,
            None => return ToolResult::error("Missing required parameter: message".into()),
        };

        let timeout_secs = input.get("timeout_secs").and_then(|v| v.as_u64());
        let chat_id = auth_context_from_input(&input).map(|ctx| ctx.caller_chat_id);

        match prompt_with_progress_updates(
            &self.manager,
            session_id,
            message,
            timeout_secs,
            chat_id,
            progress_callback_from_notify(self.notify.as_ref()),
        )
        .await
        {
            Ok((result, progress_summary)) => {
                if progress_summary.forwarded_agent_text {
                    ToolResult::success(direct_delivery_completion_message("acp_prompt", &result))
                } else {
                    ToolResult::success(result.forwarded_text())
                }
            }
            Err(e) => {
                ToolResult::error(format!("ACP prompt failed: {e}")).with_error_type("acp_error")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// acp_end_session
// ---------------------------------------------------------------------------

struct AcpEndSessionTool {
    manager: Arc<AcpManager>,
}

impl AcpEndSessionTool {
    fn new(manager: Arc<AcpManager>) -> Self {
        Self { manager }
    }
}

#[async_trait]
impl Tool for AcpEndSessionTool {
    fn name(&self) -> &str {
        "acp_end_session"
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "acp_end_session".into(),
            description: "End an ACP agent session and terminate the agent subprocess. \
                Call this when you're done with the coding agent to free resources."
                .into(),
            input_schema: schema_object(
                json!({
                    "session_id": {
                        "type": "string",
                        "description": "Session ID returned by acp_new_session"
                    }
                }),
                &["session_id"],
            ),
        }
    }

    async fn execute(&self, input: serde_json::Value) -> ToolResult {
        let session_id = match input.get("session_id").and_then(|v| v.as_str()) {
            Some(s) => s,
            None => return ToolResult::error("Missing required parameter: session_id".into()),
        };

        match self.manager.end_session(session_id).await {
            Ok(()) => ToolResult::success(
                json!({
                    "status": "ended",
                    "session_id": session_id,
                })
                .to_string(),
            ),
            Err(e) => ToolResult::error(format!("Failed to end ACP session: {e}"))
                .with_error_type("acp_error"),
        }
    }
}

// ---------------------------------------------------------------------------
// acp_list_sessions
// ---------------------------------------------------------------------------

struct AcpListSessionsTool {
    manager: Arc<AcpManager>,
}

impl AcpListSessionsTool {
    fn new(manager: Arc<AcpManager>) -> Self {
        Self { manager }
    }
}

#[async_trait]
impl Tool for AcpListSessionsTool {
    fn name(&self) -> &str {
        "acp_list_sessions"
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "acp_list_sessions".into(),
            description: "List all active ACP agent sessions with their status, agent type, \
                workspace, and creation time."
                .into(),
            input_schema: schema_object(json!({}), &[]),
        }
    }

    async fn execute(&self, _input: serde_json::Value) -> ToolResult {
        let sessions = self.manager.list_sessions().await;

        let entries: Vec<serde_json::Value> = sessions
            .iter()
            .map(|s| {
                json!({
                    "session_id": s.session_id,
                    "agent": s.agent_id,
                    "workspace": s.workspace,
                    "status": format!("{:?}", s.status),
                    "created_at": s.created_at,
                    "idle_secs": s.idle_secs,
                })
            })
            .collect();

        let available = self.manager.available_agents();

        ToolResult::success(
            json!({
                "sessions": entries,
                "available_agents": available,
            })
            .to_string(),
        )
    }
}

// ---------------------------------------------------------------------------
// acp_submit_job
// ---------------------------------------------------------------------------

struct AcpSubmitJobTool {
    manager: Arc<AcpManager>,
    on_complete: Option<JobCompletionCallback>,
    notify: Option<NotifyFn>,
}

impl AcpSubmitJobTool {
    fn new(
        manager: Arc<AcpManager>,
        on_complete: Option<JobCompletionCallback>,
        notify: Option<NotifyFn>,
    ) -> Self {
        Self {
            manager,
            on_complete,
            notify,
        }
    }
}

#[async_trait]
impl Tool for AcpSubmitJobTool {
    fn name(&self) -> &str {
        "acp_submit_job"
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "acp_submit_job".into(),
            description: "Submit a long-running coding task to an ACP agent as an async job. \
                Returns a job_id immediately without waiting for completion. The agent executes \
                in the background and the result is pushed to the chat when done. \
                Use acp_job_status to check progress."
                .into(),
            input_schema: schema_object(
                json!({
                    "session_id": {
                        "type": "string",
                        "description": "Session ID returned by acp_new_session"
                    },
                    "message": {
                        "type": "string",
                        "description": "The coding task or instruction to send to the agent"
                    },
                    "timeout_secs": {
                        "type": "integer",
                        "description": "Max seconds for execution. Defaults to config value (300s)."
                    }
                }),
                &["session_id", "message"],
            ),
        }
    }

    async fn execute(&self, input: serde_json::Value) -> ToolResult {
        let session_id = match input.get("session_id").and_then(|v| v.as_str()) {
            Some(s) => s,
            None => return ToolResult::error("Missing required parameter: session_id".into()),
        };

        let message = match input.get("message").and_then(|v| v.as_str()) {
            Some(m) => m,
            None => return ToolResult::error("Missing required parameter: message".into()),
        };

        let timeout_secs = input.get("timeout_secs").and_then(|v| v.as_u64());

        // Extract caller chat_id for completion notification
        let chat_id = auth_context_from_input(&input).map(|ctx| ctx.caller_chat_id);

        match self
            .manager
            .submit_job(
                session_id,
                message,
                timeout_secs,
                chat_id,
                self.on_complete.clone(),
                progress_callback_from_notify(self.notify.as_ref()),
            )
            .await
        {
            Ok(job_id) => ToolResult::success(
                json!({
                    "job_id": job_id,
                    "status": "submitted",
                    "session_id": session_id,
                    "message": "Job submitted. Results will be pushed to the chat when complete. Use acp_job_status to check progress."
                })
                .to_string(),
            ),
            Err(e) => {
                ToolResult::error(format!("Failed to submit ACP job: {e}"))
                    .with_error_type("acp_error")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// acp_job_status
// ---------------------------------------------------------------------------

struct AcpJobStatusTool {
    manager: Arc<AcpManager>,
}

impl AcpJobStatusTool {
    fn new(manager: Arc<AcpManager>) -> Self {
        Self { manager }
    }
}

#[async_trait]
impl Tool for AcpJobStatusTool {
    fn name(&self) -> &str {
        "acp_job_status"
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "acp_job_status".into(),
            description: "Check the status of an async ACP job submitted via acp_submit_job. \
                Returns the current status (running/completed/failed) and result if available."
                .into(),
            input_schema: schema_object(
                json!({
                    "job_id": {
                        "type": "string",
                        "description": "Job ID returned by acp_submit_job"
                    }
                }),
                &["job_id"],
            ),
        }
    }

    async fn execute(&self, input: serde_json::Value) -> ToolResult {
        let job_id = match input.get("job_id").and_then(|v| v.as_str()) {
            Some(id) => id,
            None => return ToolResult::error("Missing required parameter: job_id".into()),
        };

        match self.manager.job_status(job_id).await {
            Ok(summary) => {
                let mut output = json!({
                    "job_id": summary.id,
                    "session_id": summary.session_id,
                    "agent": summary.agent_id,
                    "status": format!("{:?}", summary.status),
                    "created_at": summary.created_at,
                });

                if let Some(completed_at) = &summary.completed_at {
                    output["completed_at"] = json!(completed_at);
                }
                if let Some(duration_ms) = summary.duration_ms {
                    output["duration_ms"] = json!(duration_ms);
                }
                if let Some(error) = &summary.error {
                    output["error"] = json!(error);
                }

                ToolResult::success(output.to_string())
            }
            Err(e) => ToolResult::error(format!("Failed to get job status: {e}"))
                .with_error_type("acp_error"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::Memory;

    fn test_manager() -> Arc<AcpManager> {
        Arc::new(AcpManager::from_config_file("/nonexistent/acp.json"))
    }

    fn memory(chat_id: Option<i64>, content: &str) -> Memory {
        Memory {
            id: 1,
            chat_id,
            content: content.to_string(),
            category: "PROFILE".to_string(),
            created_at: "2026-01-01T00:00:00Z".to_string(),
            updated_at: "2026-01-01T00:00:00Z".to_string(),
            embedding_model: None,
            confidence: 0.8,
            source: "test".to_string(),
            last_seen_at: "2026-01-01T00:00:00Z".to_string(),
            is_archived: false,
            archived_at: None,
        }
    }

    fn stored_message(content: &str) -> StoredMessage {
        StoredMessage {
            id: uuid::Uuid::new_v4().to_string(),
            chat_id: 1,
            sender_name: "tester".to_string(),
            content: content.to_string(),
            is_from_bot: false,
            timestamp: "2026-01-01T00:00:00Z".to_string(),
        }
    }

    #[test]
    fn test_tool_names_unique() {
        let manager = test_manager();
        let tools = make_acp_tools(manager);
        let names: Vec<&str> = tools.iter().map(|t| t.name()).collect();
        assert_eq!(names.len(), 7);

        let mut sorted = names.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), 7, "Tool names must be unique");
    }

    #[test]
    fn test_infer_agent_from_chat_specific_memory_prefers_codex() {
        let memories = vec![
            memory(
                Some(123),
                "Use OpenAI Codex for CareAI school project coding tasks",
            ),
            memory(None, "General note"),
        ];
        let agent = infer_agent_from_memories(123, "/users/td/projects/careai", &memories);
        assert_eq!(agent.as_deref(), Some("codex"));
    }

    #[test]
    fn test_infer_agent_from_context_prefers_claude_for_work() {
        let memories = vec![memory(
            None,
            "Use Claude Code for work/professional coding projects",
        )];
        let agent = infer_agent_from_memories(123, "/users/td/projects/work/bw", &memories);
        assert_eq!(agent.as_deref(), Some("claude"));
    }

    #[test]
    fn test_infer_workspace_from_recent_messages_prefers_existing_project_path() {
        let recent = vec![
            stored_message("Old path: /Users/tungdao/Projects/careai"),
            stored_message("Current repo is /Users/td/Projects/careai/apps/client-pwa/src"),
        ];
        let workspace = infer_workspace_from_recent_messages("Please continue in CareAI", &recent);
        assert_eq!(workspace.as_deref(), Some("/Users/td/Projects/careai"));
    }

    #[test]
    fn test_workspace_matches_normalizes_equivalent_paths() {
        assert!(workspace_matches(
            Some("~/Projects/careai/apps/client-pwa"),
            "/Users/td/Projects/careai"
        ));
    }

    #[test]
    fn test_tool_definitions_valid() {
        let manager = test_manager();
        let tools = make_acp_tools(manager);
        for tool in &tools {
            let def = tool.definition();
            assert!(!def.name.is_empty());
            assert!(!def.description.is_empty());
            assert!(def.input_schema.get("type").is_some());
            // Name must match [a-zA-Z0-9_-]{1,64}
            assert!(def.name.len() <= 64);
            assert!(def
                .name
                .chars()
                .all(|c| c.is_alphanumeric() || c == '_' || c == '-'));
        }
    }

    #[test]
    fn test_direct_delivery_completion_message_discourages_retries() {
        let result = AcpPromptResult {
            messages: vec!["Implemented the fix and added tests.".to_string()],
            tool_outputs: vec![],
            tool_calls: vec![],
            files_changed: vec!["src/tools/acp.rs".to_string()],
            completed: true,
            duration_ms: 1500,
            context_reset: false,
        };

        let message = direct_delivery_completion_message("acp_coding", &result);
        assert!(message.contains("completed successfully"));
        assert!(message.contains("already delivered directly"));
        assert!(message.contains("do not call `acp_coding` again"));
        assert!(message.contains("src/tools/acp.rs"));
        assert!(message.contains("Implemented the fix and added tests."));
    }

    #[test]
    fn test_tool_names_match() {
        let manager = test_manager();
        let tools = make_acp_tools(manager);
        let expected = vec![
            "acp_coding",
            "acp_new_session",
            "acp_prompt",
            "acp_end_session",
            "acp_list_sessions",
            "acp_submit_job",
            "acp_job_status",
        ];
        let names: Vec<&str> = tools.iter().map(|t| t.name()).collect();
        assert_eq!(names, expected);
    }

    #[tokio::test]
    async fn test_new_session_missing_agent_param() {
        let manager = test_manager();
        let tool = AcpNewSessionTool::new(manager, None);
        let result = tool.execute(json!({})).await;
        assert!(result.is_error);
        assert!(result.content.contains("Missing"));
    }

    #[tokio::test]
    async fn test_new_session_unknown_agent() {
        let manager = test_manager();
        let tool = AcpNewSessionTool::new(manager, None);
        let result = tool.execute(json!({"agent": "nonexistent"})).await;
        assert!(result.is_error);
        assert!(result.content.contains("not configured"));
    }

    #[tokio::test]
    async fn test_prompt_missing_params() {
        let manager = test_manager();
        let tool = AcpPromptTool::new(manager, None);

        // Missing session_id
        let r1 = tool.execute(json!({"message": "hello"})).await;
        assert!(r1.is_error);
        assert!(r1.content.contains("session_id"));

        // Missing message
        let r2 = tool.execute(json!({"session_id": "abc"})).await;
        assert!(r2.is_error);
        assert!(r2.content.contains("message"));
    }

    #[tokio::test]
    async fn test_prompt_session_not_found() {
        let manager = test_manager();
        let tool = AcpPromptTool::new(manager, None);
        let result = tool
            .execute(json!({"session_id": "nonexistent", "message": "hello"}))
            .await;
        assert!(result.is_error);
        assert!(result.content.contains("not found"));
    }

    #[tokio::test]
    async fn test_end_session_not_found() {
        let manager = test_manager();
        let tool = AcpEndSessionTool::new(manager);
        let result = tool.execute(json!({"session_id": "nonexistent"})).await;
        assert!(result.is_error);
        assert!(result.content.contains("not found"));
    }

    #[tokio::test]
    async fn test_list_sessions_empty() {
        let manager = test_manager();
        let tool = AcpListSessionsTool::new(manager);
        let result = tool.execute(json!({})).await;
        assert!(!result.is_error);

        let parsed: serde_json::Value = serde_json::from_str(&result.content).unwrap();
        assert_eq!(parsed["sessions"].as_array().unwrap().len(), 0);
        assert!(parsed["available_agents"].as_array().unwrap().is_empty());
    }

    // -----------------------------------------------------------------------
    // Phase 7.2: Additional schema validation tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_tool_schemas_have_required_fields() {
        let manager = test_manager();
        let tools = make_acp_tools(manager);

        for tool in &tools {
            let def = tool.definition();
            let schema = &def.input_schema;

            // Must have "type": "object"
            assert_eq!(
                schema.get("type").and_then(|v| v.as_str()),
                Some("object"),
                "Tool '{}' schema must be type=object",
                def.name
            );

            // Must have "properties" key
            assert!(
                schema.get("properties").is_some(),
                "Tool '{}' schema must have properties",
                def.name
            );
        }
    }

    #[test]
    fn test_tool_schemas_required_params_are_in_properties() {
        let manager = test_manager();
        let tools = make_acp_tools(manager);

        for tool in &tools {
            let def = tool.definition();
            let schema = &def.input_schema;

            let properties = schema.get("properties").and_then(|v| v.as_object());
            let required = schema
                .get("required")
                .and_then(|v| v.as_array())
                .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect::<Vec<_>>())
                .unwrap_or_default();

            if let Some(props) = properties {
                for req_field in &required {
                    assert!(
                        props.contains_key(*req_field),
                        "Tool '{}': required field '{}' not found in properties",
                        def.name,
                        req_field
                    );
                }
            }
        }
    }

    #[test]
    fn test_tool_descriptions_reasonable_length() {
        let manager = test_manager();
        let tools = make_acp_tools(manager);

        for tool in &tools {
            let def = tool.definition();
            assert!(
                def.description.len() >= 10,
                "Tool '{}' description too short ({})",
                def.name,
                def.description.len()
            );
            assert!(
                def.description.len() <= 1024,
                "Tool '{}' description too long ({})",
                def.name,
                def.description.len()
            );
        }
    }

    #[test]
    fn test_acp_new_session_schema_details() {
        let manager = test_manager();
        let tool = AcpNewSessionTool::new(manager, None);
        let def = tool.definition();

        let props = def.input_schema["properties"].as_object().unwrap();
        assert!(props.contains_key("agent"), "Must have 'agent' param");
        assert!(
            props.contains_key("workspace"),
            "Must have 'workspace' param"
        );
        assert!(
            props.contains_key("auto_approve"),
            "Must have 'auto_approve' param"
        );

        let required: Vec<&str> = def.input_schema["required"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        assert!(required.contains(&"agent"), "'agent' must be required");
        assert!(
            !required.contains(&"workspace"),
            "'workspace' should be optional"
        );
    }

    #[test]
    fn test_acp_prompt_schema_details() {
        let manager = test_manager();
        let tool = AcpPromptTool::new(manager, None);
        let def = tool.definition();

        let props = def.input_schema["properties"].as_object().unwrap();
        assert!(props.contains_key("session_id"));
        assert!(props.contains_key("message"));
        assert!(props.contains_key("timeout_secs"));

        let required: Vec<&str> = def.input_schema["required"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        assert!(required.contains(&"session_id"));
        assert!(required.contains(&"message"));
        assert!(!required.contains(&"timeout_secs"));
    }

    #[tokio::test]
    async fn test_end_session_missing_param() {
        let manager = test_manager();
        let tool = AcpEndSessionTool::new(manager);
        let result = tool.execute(json!({})).await;
        assert!(result.is_error);
        assert!(result.content.contains("session_id"));
    }

    #[tokio::test]
    async fn test_list_sessions_ignores_extra_params() {
        let manager = test_manager();
        let tool = AcpListSessionsTool::new(manager);
        // Extra params should be silently ignored
        let result = tool.execute(json!({"unexpected": "param"})).await;
        assert!(!result.is_error);
    }

    #[test]
    fn test_tool_risk_levels() {
        use crate::tools::tool_risk;
        use crate::tools::ToolRisk;

        assert_eq!(tool_risk("acp_coding"), ToolRisk::High);
        assert_eq!(tool_risk("acp_prompt"), ToolRisk::High);
        assert_eq!(tool_risk("acp_submit_job"), ToolRisk::High);
        assert_eq!(tool_risk("acp_new_session"), ToolRisk::Medium);
        // Other ACP tools default to Low
        assert_eq!(tool_risk("acp_end_session"), ToolRisk::Low);
        assert_eq!(tool_risk("acp_list_sessions"), ToolRisk::Low);
        assert_eq!(tool_risk("acp_job_status"), ToolRisk::Low);
    }

    // -----------------------------------------------------------------------
    // Phase 3: Async job tool tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_submit_job_missing_params() {
        let manager = test_manager();
        let tool = AcpSubmitJobTool::new(manager, None, None);

        let r1 = tool.execute(json!({"message": "hello"})).await;
        assert!(r1.is_error);
        assert!(r1.content.contains("session_id"));

        let r2 = tool.execute(json!({"session_id": "abc"})).await;
        assert!(r2.is_error);
        assert!(r2.content.contains("message"));
    }

    #[tokio::test]
    async fn test_submit_job_session_not_found() {
        let manager = test_manager();
        let tool = AcpSubmitJobTool::new(manager, None, None);
        let result = tool
            .execute(json!({"session_id": "nonexistent", "message": "hello"}))
            .await;
        assert!(result.is_error);
        assert!(result.content.contains("not found"));
    }

    #[tokio::test]
    async fn test_job_status_missing_param() {
        let manager = test_manager();
        let tool = AcpJobStatusTool::new(manager);
        let result = tool.execute(json!({})).await;
        assert!(result.is_error);
        assert!(result.content.contains("job_id"));
    }

    #[tokio::test]
    async fn test_job_status_not_found() {
        let manager = test_manager();
        let tool = AcpJobStatusTool::new(manager);
        let result = tool.execute(json!({"job_id": "nonexistent"})).await;
        assert!(result.is_error);
        assert!(result.content.contains("not found"));
    }

    #[test]
    fn test_submit_job_schema_details() {
        let manager = test_manager();
        let tool = AcpSubmitJobTool::new(manager, None, None);
        let def = tool.definition();

        let props = def.input_schema["properties"].as_object().unwrap();
        assert!(props.contains_key("session_id"));
        assert!(props.contains_key("message"));
        assert!(props.contains_key("timeout_secs"));

        let required: Vec<&str> = def.input_schema["required"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        assert!(required.contains(&"session_id"));
        assert!(required.contains(&"message"));
        assert!(!required.contains(&"timeout_secs"));
    }

    #[test]
    fn test_job_status_schema_details() {
        let manager = test_manager();
        let tool = AcpJobStatusTool::new(manager);
        let def = tool.definition();

        let props = def.input_schema["properties"].as_object().unwrap();
        assert!(props.contains_key("job_id"));

        let required: Vec<&str> = def.input_schema["required"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        assert!(required.contains(&"job_id"));
    }
}
