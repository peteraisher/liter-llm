---
id: legacy_java_guides_error_handling
language: java
target: java
level: typecheck
requires: []
side_effect: network
---

<!-- snippet:compile-only -->

```java
import io.xberg.literllm.*;
import java.util.List;

public class ErrorHandling {
    public static void main(String[] args) {
        try (var client = LiterLlm.createClient(System.getenv("OPENAI_API_KEY"))) {
            var response = client.chat(ChatCompletionRequest.builder()
                .withModel("openai/gpt-4o")
                .withMessages(List.of(
                    new Message.User(new UserMessage(UserContent.of("Hello"), null))
                ))
                .build());
            System.out.println(response.choices().get(0).message().content());
        } catch (LiterLlmRsException e) {
            System.err.println("ffi error (" + e.getCode() + "): " + e.getMessage());
        } catch (Exception e) {
            System.err.println("unexpected: " + e.getMessage());
        }
    }
}
```
