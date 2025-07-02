# ✍️ Prototype an Agent

The first take-home project will be a chance to get some hands-on practice with the concepts of Week 1 for an agent task of your choice.

## Goal

- Choose a task involving multi-turn interaction/tool use
- Implement an agent scaffold either using an API directly, or one of the frameworks we covered
- Create a small set of "test prompts"
- Create a "reward function" to evaluate your agent
- Test your agent setup with multiple models/prompts
- Examine multiple agent outputs, identify a consistent "problem", adjust the setup (prompts/tools) OR adjust your evals to measure/address the problem


## Ideas for agent tasks

- Search agent for your favorite blog/website
- Agent which
- Agent for playing a simple board/card game
- Code agent specialized to only use a specific library, e.g. iterating on a matplotlib plot until it "looks right"
- Terminal-based chat agent with user handoff/confirmation


## Ideas for reward functions

- Format checks using regex
- Deterministic checks (parsing math answers, running code with test cases, solving a puzzle/game)
- Embedding or text overlap similarity to a "ground truth"
- LLM judges which can see the "ground truth"
- LLM judges which evaluate a set of fuzzy criteria + give scores for each



## Tips

- Start simple, get a basic version working, then ramp up complexity
- If your agent "just works" with a fairly powerful model, try it with a weaker model and see what breaks


## Bonus goals

- Try making a "parallel-friendly" version using asyncio + error handling
- Try implementing Best-of-N selection -- can your eval function match your judgment for which outputs are "best"?
- Try testing either a "multi-agent" (parallelized) version of your agent, OR a Client/Server version (e.g. MCP, A2A)


## Deliverable

- A repo, notebook, *or* short writeup detailing your setup + experimentation
- What approaches did you try?
- What roadblocks did you run into?
- Which evaluation methods worked best for your task?
- What's the smallest model that worked decently well?
