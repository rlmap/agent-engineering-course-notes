# Homework Assignment Notes

## 👤 Prompt

Here's a homework assignment I got for a course on agent engineering with a specific focus on GRPO style RLHF. What does the instructor mean here by "reward functions", I get the ideas but not really the definition in relation to GRPO. Need it in layman's terms.

## 🤖 Response

### “Reward function” in plain English

Think of your agent as a student taking a quiz every time it answers a prompt.
A **reward function is just the automatic grader** — a tiny program or checklist that looks at the agent’s output and hands back a numeric score (“+10, great job” … “–3, you forgot the name”).
That single number is the only thing the learning algorithm cares about: the higher the reward, the more the model nudges its weights in that direction.

---

### Why it matters for **GRPO-style RLHF**

Classical RLHF (PPO, DPO, etc.) keeps two extra neural networks around:

| piece              | job                                          |
| ------------------ | -------------------------------------------- |
| **Reward model**   | guesses what humans would think of an answer |
| **Value / critic** | estimates future reward for partial answers  |

GRPO (Group-/General/Guided Relative Policy Optimization) throws both away to save memory and compute. Instead it **calls your *programmable reward function* directly on each finished answer and compares scores across samples**. The policy is updated so that answers with higher scores become more likely. Because of that, the quality of the hand-written reward function (not a learned model) is the main driver of learning ([deeplearning.ai][1], [docs.unsloth.ai][2]).

---

### What counts as a good reward function for the homework?

1. **Deterministic pass/fail tests**
   *Run the answer, run unit-tests, return +1 if all pass, 0 otherwise.*

2. **Regex / structure checks**
   *Does the output start with “SUMMARY:”?  +1; if length > 200 tokens, –1.*

3. **Ground-truth comparison**
   *Compare to gold answer with string or embedding similarity → reward = similarity score.*

4. **LLM-as-a-judge**
   *Feed both the candidate answer and the reference into a judge model; map 1-to-5 stars onto –2…+2 reward.*

These are the exact kinds of reward functions GRPO tutorials encourage ([docs.unsloth.ai][2]).

---

### How to frame it for a few sample agent tasks

| Task idea                   | Simple reward rubric (example)                                                                                    |
| --------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| **Blog-search agent**       | -1 if the URL 404s; +2 if page title contains query; additional +1 if summary under 60 words                      |
| **Matplotlib “plot fixer”** | Run the script; +3 if it executes without error; +1 for each required label present; –2 if more than two warnings |
| **Board-game bot**          | +1 for a win, 0 for draw, –1 for loss; bonus +0.2 for winning quickly                                             |
| **Terminal chat agent**     | Regex check for `sudo`, `rm -rf /`; –5 if found (safety); otherwise judge answer clarity 1-5 and scale to –1…+1   |

Keep the scale small (e.g., –5 … +5) so gradients don’t explode.

---

### Take-aways for your project write-up

* **Define the success rubric first** – write the reward code before touching the model loop.
* **Start shallow** – one or two rules, see if the agent moves the score upward, then refine.
* **Watch for reward-hacking** – if the agent finds a loophole (e.g., always prints the regex token), patch the grader or add a penalty term.
* **Document the mistakes** you saw and how tweaking the reward fixed them – that’s exactly what the last bullet in the assignment is asking for.

If you keep the mindset “my reward function is an automated grader that hands the agent a score every round,” the rest of GRPO will click into place.

[1]: https://www.deeplearning.ai/short-courses/reinforcement-fine-tuning-llms-grpo/ "Reinforcement Fine-Tuning LLMs with GRPO - DeepLearning.AI"
[2]: https://docs.unsloth.ai/basics/reinforcement-learning-guide "Reinforcement Learning Guide | Unsloth Documentation"
