# frozen_string_literal: true

require_relative "../lib/liter_llm"

RSpec.describe LiterLlm::CreateFileRequest do
  it "accepts a valid batch purpose" do
    request = described_class.new(file: "data.jsonl", purpose: "batch")
    expect(request.purpose).to(eq(:batch))
  end

  it "rejects an invalid purpose before an HTTP request can be constructed" do
    expect {
      described_class.new(file: "data.jsonl", purpose: "invalid-purpose")
    }
      .to(raise_error(TypeError, /invalid value for `purpose`:.*invalid FilePurpose value: invalid-purpose/))
  end
end
