use std::sync::OnceLock;

use async_trait::async_trait;
use serde_json::json;

use super::web_html::extract_ddg_results;
use super::{schema_object, Tool, ToolResult};
use crate::llm_types::ToolDefinition;

fn http_client() -> &'static reqwest::Client {
    static CLIENT: OnceLock<reqwest::Client> = OnceLock::new();
    CLIENT.get_or_init(|| {
        reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(15))
            .redirect(reqwest::redirect::Policy::limited(5))
            .user_agent("RayClaw/1.0")
            .build()
            .expect("failed to build HTTP client")
    })
}

pub struct WebSearchTool {
    brave_api_key: Option<String>,
    exa_api_key: Option<String>,
}

impl WebSearchTool {
    pub fn new(brave_api_key: Option<String>, exa_api_key: Option<String>) -> Self {
        Self {
            brave_api_key,
            exa_api_key,
        }
    }
}

#[async_trait]
impl Tool for WebSearchTool {
    fn name(&self) -> &str {
        "web_search"
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "web_search".into(),
            description: "Search the web using Exa Search API, Brave Search API, or DuckDuckGo. Returns titles, URLs, and snippets."
                .into(),
            input_schema: schema_object(
                json!({
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default: 8, max: 20)"
                    }
                }),
                &["query"],
            ),
        }
    }

    async fn execute(&self, input: serde_json::Value) -> ToolResult {
        let query = match input.get("query").and_then(|v| v.as_str()) {
            Some(q) => q,
            None => return ToolResult::error("Missing required parameter: query".into()),
        };

        let num_results = input
            .get("num_results")
            .and_then(|v| v.as_u64())
            .map(|n| n as usize)
            .unwrap_or(8)
            .min(20);

        if let Some(ref api_key) = self.exa_api_key {
            match search_exa(query, api_key, num_results).await {
                Ok(results) => {
                    if results.is_empty() {
                        ToolResult::success("No results found.".into())
                    } else {
                        ToolResult::success(results)
                    }
                }
                Err(e) => ToolResult::error(format!("Exa Search failed: {e}")),
            }
        } else if let Some(ref api_key) = self.brave_api_key {
            match search_brave(query, api_key, num_results).await {
                Ok(results) => {
                    if results.is_empty() {
                        ToolResult::success("No results found.".into())
                    } else {
                        ToolResult::success(results)
                    }
                }
                Err(e) => ToolResult::error(format!("Brave Search failed: {e}")),
            }
        } else {
            match search_ddg(query, num_results).await {
                Ok(results) => {
                    if results.is_empty() {
                        ToolResult::success("No results found.".into())
                    } else {
                        ToolResult::success(results)
                    }
                }
                Err(e) => ToolResult::error(format!("Search failed: {e}")),
            }
        }
    }
}

async fn search_exa(query: &str, api_key: &str, num_results: usize) -> Result<String, String> {
    let url = "https://api.exa.ai/search";
    let body = json!({
        "query": query,
        "type": "auto",
        "num_results": num_results,
        "contents": {
            "highlights": true
        }
    });

    let resp = http_client()
        .post(url)
        .header("x-api-key", api_key)
        .header("Content-Type", "application/json")
        .json(&body)
        .send()
        .await
        .map_err(|e| e.to_string())?;

    if !resp.status().is_success() {
        let status = resp.status();
        let err_body = resp.text().await.unwrap_or_default();
        return Err(format!("HTTP {} - {}", status, err_body));
    }

    let val: serde_json::Value = resp.json().await.map_err(|e| e.to_string())?;
    let mut output = String::new();

    if let Some(results) = val.get("results").and_then(|r| r.as_array()) {
        for (i, item) in results.iter().enumerate() {
            let title = item
                .get("title")
                .and_then(|t| t.as_str())
                .unwrap_or_default();
            let url = item.get("url").and_then(|u| u.as_str()).unwrap_or_default();
            let highlights = item
                .get("highlights")
                .and_then(|h| h.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|val| val.as_str())
                        .collect::<Vec<_>>()
                        .join(" ... ")
                })
                .unwrap_or_default();

            output.push_str(&format!(
                "{}. {}\n   {}\n   {}\n\n",
                i + 1,
                title,
                url,
                highlights
            ));
        }
    }

    Ok(output)
}

async fn search_brave(query: &str, api_key: &str, num_results: usize) -> Result<String, String> {
    let encoded = urlencoding::encode(query);
    let url =
        format!("https://api.search.brave.com/res/v1/web/search?q={encoded}&count={num_results}");

    let resp = http_client()
        .get(&url)
        .header("Accept", "application/json")
        .header("X-Subscription-Token", api_key)
        .send()
        .await
        .map_err(|e| e.to_string())?;

    if !resp.status().is_success() {
        return Err(format!("HTTP {}", resp.status()));
    }

    let val: serde_json::Value = resp.json().await.map_err(|e| e.to_string())?;
    let mut output = String::new();

    if let Some(results) = val
        .get("web")
        .and_then(|w| w.get("results"))
        .and_then(|r| r.as_array())
    {
        for (i, item) in results.iter().enumerate() {
            let title = item
                .get("title")
                .and_then(|t| t.as_str())
                .unwrap_or_default();
            let url = item.get("url").and_then(|u| u.as_str()).unwrap_or_default();
            let description = item
                .get("description")
                .and_then(|d| d.as_str())
                .unwrap_or_default();

            output.push_str(&format!(
                "{}. {}\n   {}\n   {}\n\n",
                i + 1,
                title,
                url,
                description
            ));
        }
    }

    Ok(output)
}

async fn search_ddg(query: &str, num_results: usize) -> Result<String, String> {
    let encoded = urlencoding::encode(query);
    let url = format!("https://html.duckduckgo.com/html/?q={encoded}");

    let resp = http_client()
        .get(&url)
        .send()
        .await
        .map_err(|e| e.to_string())?;

    if !resp.status().is_success() {
        return Err(format!("HTTP {}", resp.status()));
    }

    let body = resp.text().await.map_err(|e| e.to_string())?;

    // Robust CAPTCHA and Rate-limiting detection
    if body.contains("anomaly-modal") || body.contains("challenge-form") || body.contains("captcha")
    {
        return Err("Search blocked: CAPTCHA or unusual traffic verification required by DuckDuckGo. Consider configuring a Brave Search API key.".to_string());
    }

    let items = extract_ddg_results(&body, num_results);

    let mut output = String::new();
    for (i, item) in items.iter().enumerate() {
        output.push_str(&format!(
            "{}. {}\n   {}\n   {}\n\n",
            i + 1,
            item.title,
            item.url,
            item.snippet
        ));
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_web_search_definition() {
        let tool = WebSearchTool::new(None, None);
        assert_eq!(tool.name(), "web_search");
        let def = tool.definition();
        assert_eq!(def.name, "web_search");
        assert!(
            def.description.contains("Exa")
                || def.description.contains("Brave Search")
                || def.description.contains("DuckDuckGo")
        );
        assert!(def.input_schema["properties"]["query"].is_object());
        let required = def.input_schema["required"].as_array().unwrap();
        assert!(required.iter().any(|v| v == "query"));
    }

    #[tokio::test]
    async fn test_web_search_missing_query() {
        let tool = WebSearchTool::new(None, None);
        let result = tool.execute(json!({})).await;
        assert!(result.is_error);
        assert!(result.content.contains("Missing required parameter: query"));
    }

    #[tokio::test]
    async fn test_web_search_null_query() {
        let tool = WebSearchTool::new(None, None);
        let result = tool.execute(json!({"query": null})).await;
        assert!(result.is_error);
        assert!(result.content.contains("Missing required parameter: query"));
    }
}
