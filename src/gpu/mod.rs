//! Hardware telemetry and GPU status detection.
//!
//! Queries system utilities like `nvidia-smi` to monitor VRAM usage,
//! utilization, and temperature for local models.

use std::process::Stdio;
use tokio::process::Command;

#[derive(Clone, Debug)]
pub struct GpuStats {
    pub name: String,
    pub util_pct: u8,
    pub vram_used_mb: u64,
    pub vram_total_mb: u64,
    pub temp_c: Option<u8>,
}

/// Probe for GPU stats. Tries NVIDIA, AMD, then Apple Silicon.
/// Returns None if no backend is available.
pub async fn probe() -> Option<GpuStats> {
    if let Some(stats) = probe_nvidia().await {
        return Some(stats);
    }
    if let Some(stats) = probe_amd().await {
        return Some(stats);
    }
    #[cfg(target_os = "macos")]
    if let Some(stats) = probe_apple().await {
        return Some(stats);
    }
    None
}

async fn probe_nvidia() -> Option<GpuStats> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
            "--format=csv,noheader,nounits",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .await
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    parse_nvidia_csv(stdout.trim())
}

async fn probe_amd() -> Option<GpuStats> {
    let output = Command::new("rocm-smi")
        .args(["--showuse", "--showmeminfo", "vram", "--showtemp", "--json"])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .await
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    parse_rocm_json(stdout.trim())
}

/// Apple Silicon GPU stats via `powermetrics`.
/// Requires macOS; silently returns None if powermetrics fails or isn't present.
/// Caller does NOT need sudo — powermetrics succeeds without root for gpu_power
/// on modern macOS; if it doesn't, we just get None.
#[cfg(target_os = "macos")]
async fn probe_apple() -> Option<GpuStats> {
    let output = Command::new("powermetrics")
        .args(["--samplers", "gpu_power", "-n", "1", "--json"])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .await
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    parse_powermetrics_json(stdout.trim())
}

#[cfg(target_os = "macos")]
fn parse_powermetrics_json(s: &str) -> Option<GpuStats> {
    let v: serde_json::Value = serde_json::from_str(s).ok()?;

    // powermetrics JSON: {"gpu_power": {"gpu_id": [{"busy_ratio": 0.34, "memory_used_bytes": 1234, ...}]}}
    let gpu_id = v
        .get("gpu_power")
        .and_then(|gp| gp.get("gpu_id"))
        .and_then(|arr| arr.get(0))?;

    let busy_ratio: f64 = gpu_id.get("busy_ratio").and_then(|r| r.as_f64())?;
    let util_pct = (busy_ratio * 100.0).round().clamp(0.0, 255.0) as u8;

    let memory_used_bytes: u64 = gpu_id.get("memory_used_bytes").and_then(|b| b.as_u64())?;
    let vram_used_mb = memory_used_bytes / 1_048_576;

    Some(GpuStats {
        name: "Apple Silicon GPU".to_string(),
        util_pct,
        vram_used_mb,
        vram_total_mb: 0, // not exposed by powermetrics
        temp_c: None,     // not exposed by powermetrics
    })
}

fn parse_nvidia_csv(s: &str) -> Option<GpuStats> {
    let line = s.lines().next()?;
    let parts: Vec<&str> = line.split(',').collect();
    if parts.len() < 5 {
        return None;
    }

    let name = parts[0].trim().to_string();
    let util_pct: u8 = parts[1].trim().parse().ok()?;
    let vram_used_mb: u64 = parts[2].trim().parse().ok()?;
    let vram_total_mb: u64 = parts[3].trim().parse().ok()?;
    let temp_raw = parts[4].trim();
    let temp_c: Option<u8> = if temp_raw == "[N/A]" {
        None
    } else {
        temp_raw.parse().ok()
    };

    Some(GpuStats {
        name,
        util_pct,
        vram_used_mb,
        vram_total_mb,
        temp_c,
    })
}

fn parse_rocm_json(s: &str) -> Option<GpuStats> {
    let v: serde_json::Value = serde_json::from_str(s).ok()?;
    let obj = v.as_object()?;

    // Find first card key (e.g. "card0")
    let card = obj.values().next()?;

    let util_pct: u8 = card["GPU use (%)"].as_str().and_then(|s| s.parse().ok())?;

    let vram_total_bytes: u64 = card["VRAM Total Memory (B)"]
        .as_str()
        .and_then(|s| s.parse().ok())?;
    let vram_used_bytes: u64 = card["VRAM Total Used Memory (B)"]
        .as_str()
        .and_then(|s| s.parse().ok())?;

    let vram_total_mb = vram_total_bytes / 1_048_576;
    let vram_used_mb = vram_used_bytes / 1_048_576;

    let temp_c: Option<u8> = card["Temperature (Sensor edge) (°C)"]
        .as_str()
        .and_then(|s| s.parse::<f32>().ok())
        .map(|f| f.round().clamp(0.0, 255.0) as u8);

    // ROCm doesn't expose a GPU name easily; use a generic label
    let name = "AMD GPU".to_string();

    Some(GpuStats {
        name,
        util_pct,
        vram_used_mb,
        vram_total_mb,
        temp_c,
    })
}

/// Format VRAM as "X.X/YYgb" (when total >= 1024 MiB) or "XXX/YYYmb".
pub fn format_vram(used_mb: u64, total_mb: u64) -> String {
    if total_mb >= 1024 {
        let used_gb = used_mb as f64 / 1024.0;
        let total_gb = (total_mb as f64 / 1024.0).round() as u64;
        format!("{:.1}/{}gb", used_gb, total_gb)
    } else {
        format!("{}/{}mb", used_mb, total_mb)
    }
}

/// Returns a color level for GPU utilization percentage:
/// 0 = green (< 60), 1 = yellow (60–89), 2 = red (>= 90).
pub fn util_color_level(pct: u8) -> u8 {
    if pct >= 90 {
        2
    } else if pct >= 60 {
        1
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_nvidia_smi_csv() {
        let s = "NVIDIA GeForce RTX 4090, 45, 8400, 24564, 72";
        let stats = parse_nvidia_csv(s).expect("should parse");
        assert_eq!(stats.name, "NVIDIA GeForce RTX 4090");
        assert_eq!(stats.util_pct, 45);
        assert_eq!(stats.vram_used_mb, 8400);
        assert_eq!(stats.vram_total_mb, 24564);
        assert_eq!(stats.temp_c, Some(72));
    }

    #[test]
    fn parse_nvidia_smi_csv_no_temp() {
        let s = "NVIDIA GeForce RTX 4090, 45, 8400, 24564, [N/A]";
        let stats = parse_nvidia_csv(s).expect("should parse");
        assert_eq!(stats.name, "NVIDIA GeForce RTX 4090");
        assert_eq!(stats.util_pct, 45);
        assert_eq!(stats.vram_used_mb, 8400);
        assert_eq!(stats.vram_total_mb, 24564);
        assert_eq!(stats.temp_c, None);
    }

    #[test]
    fn parse_rocm_smi_json() {
        let json = r#"{"card0":{"GPU use (%)":"23","VRAM Total Memory (B)":"17163091968","VRAM Total Used Memory (B)":"4294967296","Temperature (Sensor edge) (°C)":"65.0"}}"#;
        let stats = parse_rocm_json(json).expect("should parse");
        assert_eq!(stats.util_pct, 23);
        // 17163091968 / 1048576 = 16368 MiB
        assert_eq!(stats.vram_total_mb, 17_163_091_968u64 / 1_048_576);
        // 4294967296 / 1048576 = 4096 MiB
        assert_eq!(stats.vram_used_mb, 4_294_967_296u64 / 1_048_576);
        assert_eq!(stats.temp_c, Some(65));
    }

    #[test]
    fn vram_display_gb() {
        let s = format_vram(8400, 16384);
        assert!(s.starts_with("8."), "got: {}", s);
        assert!(s.ends_with("gb"), "got: {}", s);
    }

    #[test]
    fn vram_display_mb() {
        assert_eq!(format_vram(512, 768), "512/768mb");
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn parse_powermetrics_gpu_json() {
        let json =
            r#"{"gpu_power":{"gpu_id":[{"busy_ratio":0.42,"memory_used_bytes":2097152000}]}}"#;
        let stats = parse_powermetrics_json(json).expect("should parse");
        assert_eq!(stats.name, "Apple Silicon GPU");
        assert_eq!(stats.util_pct, 42);
        assert_eq!(stats.vram_used_mb, 2097152000u64 / 1_048_576);
        assert_eq!(stats.vram_total_mb, 0);
        assert_eq!(stats.temp_c, None);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn parse_powermetrics_gpu_json_zero_busy() {
        let json = r#"{"gpu_power":{"gpu_id":[{"busy_ratio":0.0,"memory_used_bytes":0}]}}"#;
        let stats = parse_powermetrics_json(json).expect("should parse");
        assert_eq!(stats.util_pct, 0);
        assert_eq!(stats.vram_used_mb, 0);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn parse_powermetrics_gpu_json_missing_gpu_id_returns_none() {
        let json = r#"{"gpu_power":{}}"#;
        assert!(parse_powermetrics_json(json).is_none());
    }

    #[test]
    fn gpu_util_color_thresholds() {
        assert_eq!(util_color_level(0), 0);
        assert_eq!(util_color_level(59), 0);
        assert_eq!(util_color_level(60), 1);
        assert_eq!(util_color_level(89), 1);
        assert_eq!(util_color_level(90), 2);
    }

    #[test]
    fn gpu_util_color_max() {
        assert_eq!(util_color_level(100), 2);
        assert_eq!(util_color_level(255), 2);
    }

    #[test]
    fn vram_display_zero_values() {
        assert_eq!(format_vram(0, 0), "0/0mb");
    }

    #[test]
    fn vram_display_at_gb_boundary() {
        let s = format_vram(0, 1024);
        assert_eq!(s, "0.0/1gb");
    }

    #[test]
    fn vram_display_just_below_gb_boundary() {
        assert_eq!(format_vram(512, 1023), "512/1023mb");
    }

    #[test]
    fn vram_display_used_exceeds_total() {
        let s = format_vram(20000, 16384);
        assert!(s.contains("gb"), "large values should use gb: {}", s);
    }

    #[test]
    fn parse_nvidia_csv_empty_string() {
        assert!(parse_nvidia_csv("").is_none());
    }

    #[test]
    fn parse_nvidia_csv_too_few_fields() {
        assert!(parse_nvidia_csv("GPU, 50, 1000").is_none());
    }

    #[test]
    fn parse_nvidia_csv_non_numeric_util() {
        assert!(parse_nvidia_csv("GPU, abc, 1000, 2000, 50").is_none());
    }

    #[test]
    fn parse_nvidia_csv_multi_gpu_uses_first_line() {
        let csv = "GPU A, 30, 4000, 8000, 60\nGPU B, 80, 6000, 8000, 75";
        let stats = parse_nvidia_csv(csv).expect("should parse first GPU");
        assert_eq!(stats.name, "GPU A");
        assert_eq!(stats.util_pct, 30);
    }

    #[test]
    fn parse_rocm_json_empty_object() {
        assert!(parse_rocm_json("{}").is_none());
    }

    #[test]
    fn parse_rocm_json_invalid_json() {
        assert!(parse_rocm_json("not json").is_none());
    }

    #[test]
    fn parse_rocm_json_missing_vram_field() {
        let json = r#"{"card0":{"GPU use (%)":"50"}}"#;
        assert!(parse_rocm_json(json).is_none());
    }

    #[test]
    fn parse_rocm_json_missing_temp_still_parses() {
        let json = r#"{"card0":{"GPU use (%)":"50","VRAM Total Memory (B)":"8589934592","VRAM Total Used Memory (B)":"1073741824"}}"#;
        let stats = parse_rocm_json(json).expect("should parse without temp");
        assert_eq!(stats.util_pct, 50);
        assert_eq!(stats.temp_c, None);
    }

    #[test]
    fn parse_rocm_json_extreme_temp_clamps_to_255() {
        let json = r#"{"card0":{"GPU use (%)":"50","VRAM Total Memory (B)":"8589934592","VRAM Total Used Memory (B)":"1073741824","Temperature (Sensor edge) (°C)":"999.0"}}"#;
        let stats = parse_rocm_json(json).expect("should parse");
        assert_eq!(stats.temp_c, Some(255));
    }

    #[test]
    fn parse_rocm_json_negative_temp_clamps_to_zero() {
        let json = r#"{"card0":{"GPU use (%)":"50","VRAM Total Memory (B)":"8589934592","VRAM Total Used Memory (B)":"1073741824","Temperature (Sensor edge) (°C)":"-10.5"}}"#;
        let stats = parse_rocm_json(json).expect("should parse");
        assert_eq!(stats.temp_c, Some(0));
    }
}
