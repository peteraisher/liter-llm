#![cfg(feature = "secrets-aws")]

use std::time::Duration;

use aws_sdk_secretsmanager::config::{BehaviorVersion, Credentials, Region};
use liter_llm_proxy::secrets::{AwsSecretsManagerProvider, SecretError, SecretManager};
use secrecy::ExposeSecret;
use wiremock::matchers::{body_json, header, header_exists, method};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn default_connector_fetches_secret_and_preserves_not_found_error() {
    tokio::time::timeout(Duration::from_secs(15), verify_aws_http_contract())
        .await
        .expect("AWS fixture must finish within its deadline");
}

async fn verify_aws_http_contract() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(header("x-amz-target", "secretsmanager.GetSecretValue"))
        .and(header_exists("authorization"))
        .and(body_json(serde_json::json!({"SecretId": "fixture/present"})))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "Name": "fixture/present", "SecretString": "retained-value", "VersionId": "version-one"
        })))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(header("x-amz-target", "secretsmanager.GetSecretValue"))
        .and(header_exists("authorization"))
        .and(body_json(serde_json::json!({"SecretId": "fixture/missing"})))
        .respond_with(ResponseTemplate::new(400).set_body_json(serde_json::json!({
            "__type": "ResourceNotFoundException", "Message": "fixture is absent"
        })))
        .expect(1)
        .mount(&server)
        .await;
    let config = aws_sdk_secretsmanager::Config::builder()
        .behavior_version(BehaviorVersion::latest())
        .region(Region::new("us-east-1"))
        .credentials_provider(Credentials::new(
            "fixture-access",
            "fixture-secret",
            None,
            None,
            "fixture",
        ))
        .endpoint_url(server.uri())
        .build();
    let provider =
        AwsSecretsManagerProvider::from_client(aws_sdk_secretsmanager::Client::from_conf(config), Duration::ZERO);
    let secret = provider.get("fixture/present").await.expect("local AWS response");
    assert_eq!(secret.value.expose_secret(), "retained-value");
    assert_eq!(secret.metadata.version, "version-one");
    let result = provider.get("fixture/missing").await;
    assert!(matches!(result, Err(SecretError::NotFound(ref name)) if name == "fixture/missing"));
    assert_eq!(server.received_requests().await.expect("recorded requests").len(), 2);
    server.verify().await;
}
