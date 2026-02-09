# AI Agent Experiments (LangGraph)

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
