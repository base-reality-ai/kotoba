//! Standardized process exit codes for `dm`.
//!
//! Per project directive Phase 2.4, all user-facing exits must come from
//! this enum. The four values are stable and documented in `dm --help`.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum ExitCode {
    Success = 0,
    AgentError = 1,
    ConfigError = 2,
    ModelUnreachable = 3,
}

impl ExitCode {
    pub const fn as_i32(self) -> i32 {
        self as u8 as i32
    }
}

impl From<ExitCode> for std::process::ExitCode {
    fn from(ec: ExitCode) -> Self {
        std::process::ExitCode::from(ec as u8)
    }
}

/// Map an `anyhow::Error` chain to the most specific `ExitCode` we can
/// identify. Walks `err.chain()` once; returns on first match, else
/// `AgentError`.
///
/// Currently classifies:
/// - `reqwest::Error` where `is_connect()` or `is_timeout()` →
///   `ModelUnreachable`.
///
/// `ConfigError` classification for config-parse anyhow errors is
/// intentionally deferred — call sites that know they hit a config failure
/// should `exit(ConfigError.as_i32())` directly at the source.
pub fn classify(err: &anyhow::Error) -> ExitCode {
    for cause in err.chain() {
        if let Some(re) = cause.downcast_ref::<reqwest::Error>() {
            if re.is_connect() || re.is_timeout() {
                return ExitCode::ModelUnreachable;
            }
        }
    }
    ExitCode::AgentError
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exit_code_numeric_values_are_stable() {
        assert_eq!(ExitCode::Success as u8, 0);
        assert_eq!(ExitCode::AgentError as u8, 1);
        assert_eq!(ExitCode::ConfigError as u8, 2);
        assert_eq!(ExitCode::ModelUnreachable as u8, 3);
    }

    #[test]
    fn as_i32_matches_discriminant() {
        for v in [
            ExitCode::Success,
            ExitCode::AgentError,
            ExitCode::ConfigError,
            ExitCode::ModelUnreachable,
        ] {
            assert_eq!(v.as_i32(), v as u8 as i32);
        }
    }

    #[test]
    fn debug_format_is_stable() {
        assert_eq!(format!("{:?}", ExitCode::Success), "Success");
        assert_eq!(format!("{:?}", ExitCode::AgentError), "AgentError");
        assert_eq!(format!("{:?}", ExitCode::ConfigError), "ConfigError");
        assert_eq!(
            format!("{:?}", ExitCode::ModelUnreachable),
            "ModelUnreachable"
        );
    }

    #[test]
    fn classify_generic_anyhow_error_returns_agent_error() {
        let e = anyhow::anyhow!("some random error");
        assert_eq!(classify(&e), ExitCode::AgentError);
    }

    #[test]
    fn classify_nested_anyhow_chain_returns_agent_error() {
        let base = anyhow::anyhow!("inner");
        let wrapped = base.context("outer context");
        assert_eq!(classify(&wrapped), ExitCode::AgentError);
    }

    #[tokio::test]
    async fn classify_reqwest_connect_error_returns_model_unreachable() {
        // Port 1 is privileged and effectively never serves — the connect
        // fails fast with a real reqwest::Error carrying is_connect() == true,
        // deterministically and offline-friendly.
        let result = reqwest::Client::builder()
            .timeout(std::time::Duration::from_millis(500))
            .build()
            .expect("build client")
            .get("http://127.0.0.1:1/")
            .send()
            .await;
        let err = result.expect_err("connect to port 1 must fail");
        assert!(
            err.is_connect() || err.is_timeout(),
            "expected connect/timeout error, got {:?}",
            err
        );
        let wrapped: anyhow::Error = err.into();
        assert_eq!(classify(&wrapped), ExitCode::ModelUnreachable);
    }
}
