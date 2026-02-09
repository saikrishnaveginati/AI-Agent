# AI Agent (LangGraph)

This repository contains experiments and prototypes for building AI agents using **LangGraph**.  
The focus is on agent orchestration, state management, and tool-driven workflows for real-world tasks.

## What’s in this repo

- Agent workflows built using LangGraph state graphs
- Explicit state handling across multi-step reasoning
- Tool invocation and control flow between agent nodes
- Python-based implementation designed for extensibility

## Current Example

### Venue Decoration AI Agent
A prototype AI agent that:
- Maintains structured state across steps
- Breaks down a user request into planning and execution phases
- Uses LangGraph to manage transitions between agent nodes
- Produces structured, actionable outputs instead of free-form text
## What This Agent Generates

The agent does not just return free-form text. It generates **structured, decision-oriented outputs** derived through multiple reasoning steps.

Specifically, the agent produces:

- A clear understanding of the user’s intent (event type, theme, constraints)
- A structured decoration plan broken down into:
  - Theme and aesthetic direction
  - Color palette and mood
  - Decoration elements (lighting, props, layout ideas)
  - Budget-aware recommendations
- Step-by-step reasoning across agent nodes, with state passed explicitly between steps
- A final consolidated response synthesized from intermediate planning outputs

Each step updates the shared agent state, allowing the final output to be:
- Consistent
- Traceable
- Easy to extend with tools or persistence later

This design mirrors how real-world agent systems move from **interpretation → planning → execution**, rather than responding in a single opaque LLM call.


## Tech Stack

- Python
- LangGraph
- Large Language Models (LLMs)

## Why LangGraph

LangGraph enables explicit control over agent state and execution flow, making agent behavior:
- More predictable
- Easier to debug
- Safer to extend with tools and memory

This approach avoids opaque agent loops and encourages production-ready design patterns.

## How to Run

```bash
python venue_deco_langgraph.py
