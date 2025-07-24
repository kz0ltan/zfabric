#!/usr/bin/env python3

from smolagents import CodeAgent, LiteLLMModel

model = LiteLLMModel(
    model_id="ollama_chat/qwen3:30b",
    api_base="http://localhost:11434",
    num_ctx=32768,
)

agent = CodeAgent(tools=[], model=model, add_base_tools=True)

print("starting...")

answer = agent.run(
    "Could you give me the 40th number in the Fibonacci sequence?",
)
print(answer)
