use async_trait::async_trait;
use chrono::Utc;
use serde::Serialize;
use serde_json::json;

use crate::llm_types::ToolDefinition;

use super::{schema_object, Tool, ToolResult};

pub struct SyncSkillsTool {
    skills_dir: std::path::PathBuf,
}

impl SyncSkillsTool {
    pub fn new(skills_dir: &str) -> Self {
        Self {
            skills_dir: std::path::PathBuf::from(skills_dir),
        }
    }

    fn normalize_skill_locator(skill_name: &str) -> String {
        let trimmed = skill_name.trim().trim_matches('/');
        let without_prefix = trimmed.strip_prefix("skills/").unwrap_or(trimmed);
        let without_skill_md = without_prefix
            .strip_suffix("/SKILL.md")
            .or_else(|| without_prefix.strip_suffix("/skill.md"))
            .unwrap_or(without_prefix);
        without_skill_md
            .strip_suffix(".md")
            .unwrap_or(without_skill_md)
            .to_string()
    }

    fn default_target_name(skill_name: &str) -> String {
        Self::normalize_skill_locator(skill_name)
            .rsplit('/')
            .next()
            .unwrap_or(skill_name.trim())
            .to_string()
    }

    fn candidate_refs(git_ref: &str) -> Vec<String> {
        let mut refs = vec![git_ref.to_string()];
        match git_ref {
            "main" => refs.push("master".to_string()),
            "master" => refs.push("main".to_string()),
            _ => {}
        }
        refs
    }

    fn candidate_urls(source_repo: &str, git_ref: &str, skill_name: &str) -> Vec<String> {
        vec![
            format!(
                "https://raw.githubusercontent.com/{}/{}/skills/{}/SKILL.md",
                source_repo, git_ref, skill_name
            ),
            format!(
                "https://raw.githubusercontent.com/{}/{}/{}/SKILL.md",
                source_repo, git_ref, skill_name
            ),
            format!(
                "https://raw.githubusercontent.com/{}/{}/{}.md",
                source_repo, git_ref, skill_name
            ),
            format!(
                "https://raw.githubusercontent.com/{}/{}/SKILL.md",
                source_repo, git_ref
            ),
        ]
    }

    async fn fetch_skill_content(
        source_repo: &str,
        skill_name: &str,
        git_ref: &str,
    ) -> Result<(String, String), String> {
        let normalized_skill = Self::normalize_skill_locator(skill_name);

        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(20))
            .build()
            .map_err(|e| e.to_string())?;

        let mut errors = Vec::new();
        for candidate_ref in Self::candidate_refs(git_ref) {
            for url in Self::candidate_urls(source_repo, &candidate_ref, &normalized_skill) {
                match client
                    .get(&url)
                    .header("User-Agent", "RayClaw/1.0")
                    .send()
                    .await
                {
                    Ok(resp) if resp.status().is_success() => {
                        let text = resp.text().await.map_err(|e| e.to_string())?;
                        if !text.trim().is_empty() {
                            return Ok((text, candidate_ref));
                        }
                    }
                    Ok(resp) => errors.push(format!("{} -> HTTP {}", url, resp.status())),
                    Err(e) => errors.push(format!("{} -> {}", url, e)),
                }
            }
        }

        Err(format!(
            "Failed to fetch skill '{skill_name}' from {source_repo}@{git_ref}. Tried URLs:\n{}",
            errors.join("\n")
        ))
    }

    fn split_frontmatter(content: &str) -> (Option<serde_yaml::Value>, String) {
        let trimmed = content.trim_start_matches('\u{feff}');
        if !trimmed.starts_with("---\n") && !trimmed.starts_with("---\r\n") {
            return (None, trimmed.to_string());
        }

        let mut lines = trimmed.lines();
        let _ = lines.next(); // opening ---
        let mut yaml_block = String::new();
        let mut consumed = 0usize;
        for line in lines {
            consumed += line.len() + 1;
            if line.trim() == "---" || line.trim() == "..." {
                break;
            }
            yaml_block.push_str(line);
            yaml_block.push('\n');
        }

        let header_len = if let Some(idx) = trimmed.find("\n---\n") {
            idx + 5
        } else if let Some(idx) = trimmed.find("\n...\n") {
            idx + 5
        } else {
            4 + consumed
        };

        let body = trimmed
            .get(header_len..)
            .unwrap_or_default()
            .trim()
            .to_string();

        if yaml_block.trim().is_empty() {
            (None, body)
        } else {
            let parsed = serde_yaml::from_str::<serde_yaml::Value>(&yaml_block)
                .ok()
                .or_else(|| {
                    let normalized = crate::skills::normalize_wrapped_frontmatter(&yaml_block);
                    serde_yaml::from_str::<serde_yaml::Value>(&normalized).ok()
                });
            (parsed, body)
        }
    }

    fn str_seq(value: Option<&serde_yaml::Value>) -> Vec<String> {
        match value {
            Some(serde_yaml::Value::Sequence(items)) => items
                .iter()
                .filter_map(|v| v.as_str().map(|s| s.trim().to_string()))
                .filter(|s| !s.is_empty())
                .collect(),
            _ => Vec::new(),
        }
    }

    fn normalize_skill_markdown(
        raw: &str,
        source_repo: &str,
        git_ref: &str,
        skill_name: &str,
        target_name: &str,
    ) -> String {
        #[derive(Serialize)]
        struct NormalizedFrontmatter {
            name: String,
            description: String,
            source: String,
            version: String,
            updated_at: String,
            license: String,
            #[serde(skip_serializing_if = "Vec::is_empty")]
            platforms: Vec<String>,
            #[serde(skip_serializing_if = "Vec::is_empty")]
            deps: Vec<String>,
        }

        let (fm, body) = Self::split_frontmatter(raw);
        let fm = fm.unwrap_or(serde_yaml::Value::Null);

        let get = |k: &str| fm.get(k).and_then(|v| v.as_str()).unwrap_or("");

        let description = if !get("description").trim().is_empty() {
            get("description").trim().to_string()
        } else {
            format!("Synced from {source_repo} skill '{skill_name}' and adapted for RayClaw.")
        };

        let mut platforms = Self::str_seq(fm.get("platforms"));
        if platforms.is_empty() {
            platforms = Self::str_seq(fm.get("compatibility").and_then(|c| c.get("os")));
        }

        let mut deps = Self::str_seq(fm.get("deps"));
        if deps.is_empty() {
            deps = Self::str_seq(fm.get("compatibility").and_then(|c| c.get("deps")));
        }

        let frontmatter = NormalizedFrontmatter {
            name: target_name.to_string(),
            description,
            source: format!("remote:{}", source_repo),
            version: git_ref.to_string(),
            updated_at: Utc::now().to_rfc3339(),
            license: "Proprietary. LICENSE.txt has complete terms".to_string(),
            platforms,
            deps,
        };
        let yaml = serde_yaml::to_string(&frontmatter).unwrap_or_default();
        let yaml = yaml.strip_prefix("---\n").unwrap_or(&yaml).trim_end();

        let rendered_body = if body.is_empty() {
            format!(
                "# {}\n\nSynced from `{}` (`{}`).",
                target_name, source_repo, git_ref
            )
        } else {
            body
        };

        format!("---\n{}\n---\n\n{}", yaml, rendered_body)
    }
}

#[async_trait]
impl Tool for SyncSkillsTool {
    fn name(&self) -> &str {
        "sync_skills"
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "sync_skills".into(),
            description: "Sync a skill from an external repository (default: vercel-labs/skills) into local rayclaw.data/skills and normalize frontmatter (source/version/updated_at/platforms/deps).".into(),
            input_schema: schema_object(
                json!({
                    "skill_name": {
                        "type": "string",
                        "description": "Upstream skill name/path to sync"
                    },
                    "target_name": {
                        "type": "string",
                        "description": "Optional local skill directory/name (defaults to skill_name)"
                    },
                    "source_repo": {
                        "type": "string",
                        "description": "GitHub repo in owner/name format (default: vercel-labs/skills)"
                    },
                    "git_ref": {
                        "type": "string",
                        "description": "Branch/tag/commit (default: main)"
                    }
                }),
                &["skill_name"],
            ),
        }
    }

    async fn execute(&self, input: serde_json::Value) -> ToolResult {
        let skill_name = match input.get("skill_name").and_then(|v| v.as_str()) {
            Some(v) if !v.trim().is_empty() => v.trim(),
            _ => return ToolResult::error("Missing required parameter: skill_name".into()),
        };

        let source_repo = input
            .get("source_repo")
            .and_then(|v| v.as_str())
            .filter(|v| !v.trim().is_empty())
            .unwrap_or("vercel-labs/skills")
            .trim();

        let git_ref = input
            .get("git_ref")
            .and_then(|v| v.as_str())
            .filter(|v| !v.trim().is_empty())
            .unwrap_or("main")
            .trim();

        let target_name = input
            .get("target_name")
            .and_then(|v| v.as_str())
            .filter(|v| !v.trim().is_empty())
            .map(|v| v.trim().to_string())
            .unwrap_or_else(|| Self::default_target_name(skill_name));

        let (raw, resolved_ref) =
            match Self::fetch_skill_content(source_repo, skill_name, git_ref).await {
                Ok(v) => v,
                Err(e) => return ToolResult::error(e).with_error_type("sync_fetch_failed"),
            };

        let normalized = Self::normalize_skill_markdown(
            &raw,
            source_repo,
            &resolved_ref,
            skill_name,
            &target_name,
        );

        let out_dir = self.skills_dir.join(&target_name);
        if let Err(e) = std::fs::create_dir_all(&out_dir) {
            return ToolResult::error(format!("Failed to create skill directory: {e}"))
                .with_error_type("sync_write_failed");
        }

        let out_file = out_dir.join("SKILL.md");
        if let Err(e) = std::fs::write(&out_file, normalized) {
            return ToolResult::error(format!("Failed to write SKILL.md: {e}"))
                .with_error_type("sync_write_failed");
        }

        ToolResult::success(format!(
            "Skill synced: {} -> {}\nSource: {}@{}\nPath: {}",
            skill_name,
            target_name,
            source_repo,
            resolved_ref,
            out_file.display()
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_sync_skills_definition() {
        let tool = SyncSkillsTool::new("/tmp/skills");
        assert_eq!(tool.name(), "sync_skills");
        let def = tool.definition();
        assert_eq!(def.name, "sync_skills");
        assert!(def.input_schema["properties"]["skill_name"].is_object());
    }

    #[tokio::test]
    async fn test_sync_skills_missing_name() {
        let tool = SyncSkillsTool::new("/tmp/skills");
        let result = tool.execute(json!({})).await;
        assert!(result.is_error);
        assert!(result.content.contains("skill_name"));
    }

    #[test]
    fn test_normalize_skill_markdown_adds_source_fields() {
        let raw = "# Demo\n\nBody";
        let out = SyncSkillsTool::normalize_skill_markdown(
            raw,
            "vercel-labs/skills",
            "main",
            "demo",
            "demo",
        );
        assert!(out.contains("source: remote:vercel-labs/skills"));
        assert!(out.contains("version: main"));
        assert!(out.contains("updated_at:"));
    }

    #[test]
    fn test_normalize_skill_markdown_quotes_multiline_description() {
        let raw = r#"---
name: demo
description: One line,
continued line
---
Body
"#;
        let out = SyncSkillsTool::normalize_skill_markdown(
            raw,
            "basecamp/fizzy-cli",
            "master",
            "demo",
            "demo",
        );
        assert!(out.contains("description:"));
        assert!(out.contains("continued line"));
        let (fm, body) = SyncSkillsTool::split_frontmatter(&out);
        assert!(fm.is_some(), "normalized frontmatter should stay parseable");
        assert_eq!(body, "Body");
    }

    #[test]
    fn test_default_target_name_normalizes_skill_locator() {
        assert_eq!(SyncSkillsTool::default_target_name("skills/fizzy"), "fizzy");
        assert_eq!(
            SyncSkillsTool::default_target_name("fizzy/SKILL.md"),
            "fizzy"
        );
        assert_eq!(SyncSkillsTool::default_target_name("fizzy.md"), "fizzy");
    }

    #[test]
    fn test_candidate_refs_try_main_and_master() {
        assert_eq!(
            SyncSkillsTool::candidate_refs("main"),
            vec!["main".to_string(), "master".to_string()]
        );
        assert_eq!(
            SyncSkillsTool::candidate_refs("master"),
            vec!["master".to_string(), "main".to_string()]
        );
    }
}
