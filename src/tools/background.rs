//! Background task execution manager
//!
//! Allows tools to run in the background and be managed via the `bg` tool.
//! Uses file-based storage for crash resilience.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::sync::{RwLock, watch};
use tokio::task::JoinHandle;
use tracing::info;

use crate::llm_types::ToolDefinition;

use super::{schema_object, Tool, ToolResult};

/// Directory for background task output files
fn task_dir() -> PathBuf {
    std::env::temp_dir().join("rayclaw-bg-tasks")
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum BackgroundTaskStatus {
    Running,
    Completed,
    Failed,
    Cancelled,
}

impl BackgroundTaskStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Running => "running",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskStatusFile {
    pub task_id: String,
    pub tool_name: String,
    pub chat_id: i64,
    pub status: BackgroundTaskStatus,
    pub exit_code: Option<i32>,
    pub error: Option<String>,
    pub started_at: String,
    pub completed_at: Option<String>,
    pub duration_secs: Option<f64>,
    pub command: String,
}

/// Result from a background task execution
pub struct TaskResult {
    pub exit_code: Option<i32>,
    pub error: Option<String>,
}

/// Internal tracking for a running task
struct RunningTask {
    task_id: String,
    tool_name: String,
    chat_id: i64,
    status_path: PathBuf,
    started_at: Instant,
    started_at_rfc3339: String,
    command: String,
    delivery_flags: watch::Sender<bool>,
    handle: JoinHandle<TaskResult>,
}

/// Information returned when a background task is started
#[derive(Debug, Clone)]
pub struct BackgroundTaskInfo {
    pub task_id: String,
    pub output_file: PathBuf,
    pub status_file: PathBuf,
}

/// Manages background task execution
pub struct BackgroundTaskManager {
    tasks: Arc<RwLock<HashMap<String, RunningTask>>>,
    output_dir: PathBuf,
}

impl BackgroundTaskManager {
    pub fn new() -> Self {
        let output_dir = task_dir();
        std::fs::create_dir_all(&output_dir).ok();
        Self {
            tasks: Arc::new(RwLock::new(HashMap::new())),
            output_dir,
        }
    }

    /// Generate a short, unique task ID
    fn generate_task_id() -> String {
        use std::time::{SystemTime, UNIX_EPOCH};
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();
        // Use timestamp + counter for uniqueness
        static COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let counter = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        format!("bg{}{:04x}", timestamp % 1000000, counter % 0x10000)
    }

    pub fn output_path_for(&self, task_id: &str) -> PathBuf {
        self.output_dir.join(format!("{}.output", task_id))
    }

    pub fn status_path_for(&self, task_id: &str) -> PathBuf {
        self.output_dir.join(format!("{}.status.json", task_id))
    }

    async fn write_status_file(&self, path: &std::path::Path, status: &TaskStatusFile) {
        if let Ok(json_str) = serde_json::to_string_pretty(status) {
            let _ = tokio::fs::write(path, json_str).await;
        }
    }

    async fn read_status_file(&self, path: &std::path::Path) -> Option<TaskStatusFile> {
        let content = tokio::fs::read_to_string(path).await.ok()?;
        serde_json::from_str(&content).ok()
    }

    /// Spawn a background task that runs a shell command
    pub async fn spawn_command(
        &self,
        tool_name: &str,
        chat_id: i64,
        command: String,
    ) -> BackgroundTaskInfo {
        let _notify = true;
        let task_id = Self::generate_task_id();
        let output_path = self.output_dir.join(format!("{}.output", task_id));
        let status_path = self.output_dir.join(format!("{}.status.json", task_id));
        let started_at_rfc3339 = chrono::Utc::now().to_rfc3339();

        // Write initial status file
        let initial_status = TaskStatusFile {
            task_id: task_id.clone(),
            tool_name: tool_name.to_string(),
            chat_id,
            status: BackgroundTaskStatus::Running,
            exit_code: None,
            error: None,
            started_at: started_at_rfc3339.clone(),
            completed_at: None,
            duration_secs: None,
            command: command.clone(),
        };
        self.write_status_file(&status_path, &initial_status).await;

        let output_path_clone = output_path.clone();
        let status_path_clone = status_path.clone();
        let task_id_clone = task_id.clone();
        let tool_name_owned = tool_name.to_string();
        let command_for_task = command.clone();
        let started_at = Instant::now();
        let started_at_rfc3339_for_task = started_at_rfc3339.clone();
        let (delivery_flags_tx, delivery_flags_rx) = watch::channel(true);

        // Spawn the background task
        let handle = tokio::spawn(async move {
            let result = tokio::time::timeout(
                std::time::Duration::from_secs(3600), // 1 hour max
                tokio::process::Command::new("sh")
                    .arg("-c")
                    .arg(&command)
                    .output(),
            )
            .await;

            let duration_secs = started_at.elapsed().as_secs_f64();
            let (status, exit_code, error) = match &result {
                Ok(Ok(output)) => {
                    let stdout = String::from_utf8_lossy(&output.stdout);
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    let exit_code = output.status.code();

                    // Write output to file
                    let mut output_text = String::new();
                    if !stdout.is_empty() {
                        output_text.push_str(&stdout);
                    }
                    if !stderr.is_empty() {
                        if !output_text.is_empty() {
                            output_text.push_str("\n--- STDERR ---\n");
                        }
                        output_text.push_str(&stderr);
                    }
                    let _ = tokio::fs::write(&output_path_clone, &output_text).await;

                    let status = if output.status.success() {
                        BackgroundTaskStatus::Completed
                    } else {
                        BackgroundTaskStatus::Failed
                    };
                    (status, exit_code, None)
                }
                Ok(Err(e)) => {
                    let error_msg = format!("Command execution failed: {}", e);
                    let _ = tokio::fs::write(&output_path_clone, &error_msg).await;
                    (BackgroundTaskStatus::Failed, None, Some(error_msg))
                }
                Err(_) => {
                    let error_msg = "Command timed out after 1 hour".to_string();
                    let _ = tokio::fs::write(&output_path_clone, &error_msg).await;
                    (BackgroundTaskStatus::Failed, None, Some(error_msg))
                }
            };

            let _ = delivery_flags_rx.borrow();

            // Update status file
            let final_status = TaskStatusFile {
                task_id: task_id_clone.clone(),
                tool_name: tool_name_owned.clone(),
                chat_id,
                status: status.clone(),
                exit_code,
                error: error.clone(),
                started_at: started_at_rfc3339_for_task,
                completed_at: Some(chrono::Utc::now().to_rfc3339()),
                duration_secs: Some(duration_secs),
                command: command.clone(),
            };
            if let Ok(json_str) = serde_json::to_string_pretty(&final_status) {
                let _ = tokio::fs::write(&status_path_clone, json_str).await;
            }

            TaskResult { exit_code, error }
        });

        // Track the running task
        let running_task = RunningTask {
            task_id: task_id.clone(),
            tool_name: tool_name.to_string(),
            chat_id,
            status_path: status_path.clone(),
            started_at,
            started_at_rfc3339: started_at_rfc3339.clone(),
            command: command_for_task,
            delivery_flags: delivery_flags_tx,
            handle,
        };

        self.tasks
            .write()
            .await
            .insert(task_id.clone(), running_task);

        BackgroundTaskInfo {
            task_id,
            output_file: output_path,
            status_file: status_path,
        }
    }

    /// List all tasks (both running and completed from disk)
    pub async fn list(&self, chat_id: Option<i64>) -> Vec<TaskStatusFile> {
        let mut results = Vec::new();

        // Read all status files from disk
        if let Ok(mut entries) = tokio::fs::read_dir(&self.output_dir).await {
            while let Ok(Some(entry)) = entries.next_entry().await {
                let path = entry.path();
                if path.extension().map(|e| e == "json").unwrap_or(false) {
                    if let Some(status) = self.read_status_file(&path).await {
                        if chat_id.map_or(true, |cid| status.chat_id == cid) {
                            results.push(status);
                        }
                    }
                }
            }
        }

        // Sort by task_id (which includes timestamp)
        results.sort_by(|a, b| b.task_id.cmp(&a.task_id));
        results
    }

    /// Get status of a specific task
    pub async fn status(&self, task_id: &str) -> Option<TaskStatusFile> {
        let status_path = self.status_path_for(task_id);
        self.read_status_file(&status_path).await
    }

    /// Get full output of a task
    pub async fn output(&self, task_id: &str, tail_lines: Option<usize>) -> Option<String> {
        let output_path = self.output_path_for(task_id);
        match tokio::fs::read_to_string(&output_path).await {
            Ok(content) => {
                if let Some(lines) = tail_lines {
                    let collected: Vec<&str> = content.lines().rev().take(lines).collect();
                    Some(collected.into_iter().rev().collect::<Vec<_>>().join("\n"))
                } else {
                    Some(content)
                }
            }
            Err(_) => None,
        }
    }

    /// Wait for a task to finish or timeout
    pub async fn wait(&self, task_id: &str, max_wait_secs: u64) -> Option<TaskStatusFile> {
        let mut bus_rx = {
            let tasks = self.tasks.read().await;
            if let Some(task) = tasks.get(task_id) {
                Some(task.delivery_flags.subscribe())
            } else {
                None
            }
        };

        if let Some(mut rx) = bus_rx {
            let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(max_wait_secs);
            let timeout = tokio::time::sleep_until(deadline);
            tokio::pin!(timeout);

            loop {
                tokio::select! {
                    _ = &mut timeout => {
                        return self.status(task_id).await;
                    }
                    result = rx.changed() => {
                        if result.is_err() {
                            // Channel closed, task finished
                            return self.status(task_id).await;
                        }
                        // Check if task is still running
                        if let Some(status) = self.status(task_id).await {
                            if status.status != BackgroundTaskStatus::Running {
                                return Some(status);
                            }
                        }
                    }
                }
            }
        } else {
            // Task not in memory, check disk
            if let Some(status) = self.status(task_id).await {
                if status.status != BackgroundTaskStatus::Running {
                    return Some(status);
                }
            }
            // Poll for completion
            let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(max_wait_secs);
            loop {
                if tokio::time::Instant::now() >= deadline {
                    return self.status(task_id).await;
                }
                if let Some(status) = self.status(task_id).await {
                    if status.status != BackgroundTaskStatus::Running {
                        return Some(status);
                    }
                }
                tokio::time::sleep(std::time::Duration::from_secs(1)).await;
            }
        }
    }

    /// Cancel a running task
    pub async fn cancel(&self, task_id: &str) -> Result<bool, String> {
        let mut tasks = self.tasks.write().await;
        if let Some(task) = tasks.remove(task_id) {
            task.handle.abort();

            // Update status file
            let final_status = TaskStatusFile {
                task_id: task.task_id,
                tool_name: task.tool_name,
                chat_id: task.chat_id,
                status: BackgroundTaskStatus::Cancelled,
                exit_code: None,
                error: Some("Cancelled by user".to_string()),
                started_at: task.started_at_rfc3339,
                completed_at: Some(chrono::Utc::now().to_rfc3339()),
                duration_secs: Some(task.started_at.elapsed().as_secs_f64()),
                command: task.command,
            };
            self.write_status_file(&task.status_path, &final_status).await;

            Ok(true)
        } else {
            drop(tasks);

            let status_path = self.status_path_for(task_id);
            let Some(mut status) = self.read_status_file(&status_path).await else {
                return Ok(false);
            };

            if status.status != BackgroundTaskStatus::Running {
                return Ok(false);
            }

            status.status = BackgroundTaskStatus::Cancelled;
            status.error = Some("Cancelled by user".to_string());
            status.completed_at = Some(chrono::Utc::now().to_rfc3339());
            self.write_status_file(&status_path, &status).await;
            Ok(true)
        }
    }

    /// Clean up old task files (older than specified hours)
    pub async fn cleanup(&self, max_age_hours: u64) -> Result<usize, String> {
        let mut removed = 0;
        let cutoff = std::time::SystemTime::now()
            - std::time::Duration::from_secs(max_age_hours * 3600);

        if let Ok(mut entries) = tokio::fs::read_dir(&self.output_dir).await {
            while let Ok(Some(entry)) = entries.next_entry().await {
                let path = entry.path();
                let Ok(metadata) = tokio::fs::metadata(&path).await else {
                    continue;
                };
                let Ok(modified) = metadata.modified() else {
                    continue;
                };
                if modified >= cutoff {
                    continue;
                }

                // Skip running tasks
                if path.extension().and_then(|ext| ext.to_str()) == Some("json") {
                    if let Some(status) = self.read_status_file(&path).await {
                        if status.status == BackgroundTaskStatus::Running {
                            continue;
                        }
                    }
                }

                let _ = tokio::fs::remove_file(&path).await;
                removed += 1;
            }
        }

        Ok(removed)
    }
}

impl Default for BackgroundTaskManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Global singleton for background task manager
static BACKGROUND_MANAGER: std::sync::OnceLock<Arc<BackgroundTaskManager>> = std::sync::OnceLock::new();

/// Get the global background task manager
pub fn global() -> Arc<BackgroundTaskManager> {
    BACKGROUND_MANAGER
        .get_or_init(|| Arc::new(BackgroundTaskManager::new()))
        .clone()
}

/// BgTool - manage background tasks
pub struct BgTool {
    manager: Arc<BackgroundTaskManager>,
}

impl BgTool {
    pub fn new() -> Self {
        Self {
            manager: global(),
        }
    }
}

#[derive(Deserialize)]
struct BgInput {
    #[serde(default)]
    action: Option<String>,
    #[serde(default)]
    task_id: Option<String>,
    #[serde(default)]
    max_wait_seconds: Option<u64>,
    #[serde(default)]
    tail_lines: Option<usize>,
    #[serde(default)]
    max_age_hours: Option<u64>,
}

#[async_trait]
impl Tool for BgTool {
    fn name(&self) -> &str {
        "bg"
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "bg".into(),
            description: "Manage background tasks. Actions: list, status, output, tail, cancel, wait, cleanup.".into(),
            input_schema: schema_object(
                json!({
                    "action": {
                        "type": "string",
                        "enum": ["list", "status", "output", "tail", "cancel", "wait", "cleanup"],
                        "description": "Action to perform"
                    },
                    "task_id": {
                        "type": "string",
                        "description": "Task ID (required for single-task actions)"
                    },
                    "max_wait_seconds": {
                        "type": "integer",
                        "description": "Max seconds to wait (default: 60, max: 3600)"
                    },
                    "tail_lines": {
                        "type": "integer",
                        "description": "Number of lines to tail (default: 80)"
                    },
                    "max_age_hours": {
                        "type": "integer",
                        "description": "Max age in hours for cleanup (default: 24)"
                    }
                }),
                &["action"],
            ),
        }
    }

    async fn execute(&self, input: Value) -> ToolResult {
        let params: BgInput = match serde_json::from_value(input) {
            Ok(p) => p,
            Err(e) => return ToolResult::error(format!("Invalid input: {}", e)),
        };

        let action = match params.action.as_deref() {
            Some(a) => a,
            None => return ToolResult::error("Missing required 'action' parameter".into()),
        };

        match action {
            "list" => {
                let tasks = self.manager.list(None).await;
                if tasks.is_empty() {
                    return ToolResult::success("No background tasks found.".into());
                }

                let mut output = String::from("Background Tasks:\n\n");
                output.push_str(&format!(
                    "{:<12} {:<15} {:<10} {:<12} {:<10}\n",
                    "TASK_ID", "TOOL", "STATUS", "DURATION", "CHAT_ID"
                ));
                output.push_str(&"-".repeat(60));
                output.push('\n');

                for task in &tasks {
                    let duration = task
                        .duration_secs
                        .map(|d| format!("{:.1}s", d))
                        .unwrap_or_else(|| "running".to_string());
                    output.push_str(&format!(
                        "{:<12} {:<15} {:<10} {:<12} {}\n",
                        task.task_id,
                        task.tool_name,
                        task.status.as_str(),
                        duration,
                        task.chat_id
                    ));
                }

                ToolResult::success(output)
            }

            "status" => {
                let task_id = match params.task_id {
                    Some(id) => id,
                    None => return ToolResult::error("task_id required for status action".into()),
                };

                match self.manager.status(&task_id).await {
                    Some(task) => {
                        let mut output = format!(
                            "Task: {}\n\
                             Tool: {}\n\
                             Status: {}\n\
                             Chat ID: {}\n\
                             Started: {}\n\
                             Command: {}\n",
                            task.task_id,
                            task.tool_name,
                            task.status.as_str(),
                            task.chat_id,
                            task.started_at,
                            task.command,
                        );

                        if let Some(completed) = task.completed_at {
                            output.push_str(&format!("Completed: {}\n", completed));
                        }
                        if let Some(duration) = task.duration_secs {
                            output.push_str(&format!("Duration: {:.2}s\n", duration));
                        }
                        if let Some(exit_code) = task.exit_code {
                            output.push_str(&format!("Exit code: {}\n", exit_code));
                        }
                        if let Some(error) = task.error {
                            output.push_str(&format!("Error: {}\n", error));
                        }

                        ToolResult::success(output)
                    }
                    None => ToolResult::error(format!("Task not found: {}", task_id)),
                }
            }

            "output" | "tail" => {
                let task_id = match params.task_id {
                    Some(id) => id,
                    None => return ToolResult::error("task_id required for output action".into()),
                };

                let tail = if action == "tail" {
                    Some(params.tail_lines.unwrap_or(80))
                } else {
                    None
                };

                match self.manager.output(&task_id, tail).await {
                    Some(content) => {
                        if content.trim().is_empty() {
                            ToolResult::success("Task output is empty.".into())
                        } else {
                            ToolResult::success(content)
                        }
                    }
                    None => ToolResult::error(format!(
                        "Output not found for task: {}. Task may not exist or output file was deleted.",
                        task_id
                    )),
                }
            }

            "cancel" => {
                let task_id = match params.task_id {
                    Some(id) => id,
                    None => return ToolResult::error("task_id required for cancel action".into()),
                };

                match self.manager.cancel(&task_id).await {
                    Ok(true) => ToolResult::success(format!("Task {} cancelled.", task_id)),
                    Ok(false) => ToolResult::error(format!(
                        "Task {} not found or already completed.",
                        task_id
                    )),
                    Err(e) => ToolResult::error(format!("Failed to cancel task: {}", e)),
                }
            }

            "wait" => {
                let task_id = match params.task_id {
                    Some(id) => id,
                    None => return ToolResult::error("task_id required for wait action".into()),
                };

                let max_wait = params.max_wait_seconds.unwrap_or(60).min(3600);

                match self.manager.wait(&task_id, max_wait).await {
                    Some(task) => {
                        let reason = match task.status {
                            BackgroundTaskStatus::Running => "timeout",
                            BackgroundTaskStatus::Completed => "completed",
                            BackgroundTaskStatus::Failed => "failed",
                            BackgroundTaskStatus::Cancelled => "cancelled",
                        };

                        let mut output = format!(
                            "Wait returned: {}\n\n\
                             Task: {}\n\
                             Status: {}\n",
                            reason, task.task_id, task.status.as_str()
                        );

                        if let Some(error) = &task.error {
                            output.push_str(&format!("Error: {}\n", error));
                        }

                        // Include output preview for failed tasks
                        if task.status == BackgroundTaskStatus::Failed {
                            if let Some(preview) = self.manager.output(&task_id, Some(40)).await {
                                if !preview.trim().is_empty() {
                                    output.push_str("\nOutput preview:\n```\n");
                                    output.push_str(&preview);
                                    output.push_str("\n```\n");
                                }
                            }
                        }

                        ToolResult::success(output)
                    }
                    None => ToolResult::error(format!("Task not found: {}", task_id)),
                }
            }

            "cleanup" => {
                let max_age = params.max_age_hours.unwrap_or(24);
                match self.manager.cleanup(max_age).await {
                    Ok(removed) => ToolResult::success(format!(
                        "Removed {} old task files (older than {} hours).",
                        removed, max_age
                    )),
                    Err(e) => ToolResult::error(format!("Cleanup failed: {}", e)),
                }
            }

            _ => ToolResult::error(format!(
                "Unknown action: {}. Valid actions: list, status, output, tail, cancel, wait, cleanup",
                action
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_bg_tool_name_and_definition() {
        let tool = BgTool::new();
        assert_eq!(tool.name(), "bg");
        let def = tool.definition();
        assert_eq!(def.name, "bg");
        assert!(def.description.contains("background"));
        assert!(def.input_schema["properties"]["action"].is_object());
    }

    #[tokio::test]
    async fn test_bg_missing_action() {
        let tool = BgTool::new();
        let result = tool.execute(json!({})).await;
        assert!(result.is_error);
        assert!(result.content.contains("Missing required 'action'"));
    }

    #[tokio::test]
    async fn test_bg_list_empty() {
        let tool = BgTool::new();
        let result = tool.execute(json!({"action": "list"})).await;
        assert!(!result.is_error);
        assert!(result.content.contains("No background tasks"));
    }

    #[tokio::test]
    async fn test_bg_status_not_found() {
        let tool = BgTool::new();
        let result = tool.execute(json!({"action": "status", "task_id": "nonexistent"})).await;
        assert!(result.is_error);
        assert!(result.content.contains("Task not found"));
    }
}
