"""
AI-Powered Venue Decoration Agent using LangGraph
==================================================

This is a fully autonomous AI agent that:
1. Analyzes venue images using a vision model
2. Generates decoration prompts based on analysis
3. Creates decorated images using AI models
4. Evaluates results using an LLM judge
5. Automatically retries until quality threshold is met

Requirements:
    pip install langgraph langchain langchain-anthropic replicate pillow requests
"""

import os
import json
import base64
import requests
from pathlib import Path
from typing import TypedDict, Literal, Optional, Annotated
from dataclasses import dataclass
import operator

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# For API calls
import replicate
from anthropic import Anthropic


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration settings for the agent."""
    
    # API Tokens (set these!)
    REPLICATE_TOKEN: str = "YOUR_REPLICATE_TOKEN"
    ANTHROPIC_TOKEN: str = "YOUR_ANTHROPIC_TOKEN"
    
    # Agent settings
    MAX_ATTEMPTS: int = 3
    MIN_PASSING_SCORE: float = 7.0
    
    # Image generation settings
    DEFAULT_MODEL: str = "sdxl"
    FALLBACK_MODEL: str = "flux_kontext"
    DEFAULT_STRENGTH: float = 0.65
    
    # Available models
    MODELS = {
        "sdxl": "stability-ai/sdxl:7762fd07cf82c948538e41f63f77d685e02b063e37e496e96eefd46c929f9bdc",
        "flux_kontext": "black-forest-labs/flux-kontext-pro",
        "flux_pro": "black-forest-labs/flux-1.1-pro",
        "stable_diffusion": "stability-ai/stable-diffusion:ac732df83cea7fff18b8472768c88ad041fa750ff7682a21affe81863cbe77e4",
    }


# ============================================================================
# STATE DEFINITION
# ============================================================================

class AgentState(TypedDict):
    """
    The state that flows through the agent graph.
    LangGraph automatically manages this state between nodes.
    """
    
    # Inputs
    venue_image_path: str
    color_scheme: dict
    event_type: str
    style: str
    intensity: str
    
    # Analysis results
    venue_analysis: str
    decoration_areas: list[str]
    
    # Generation
    current_prompt: str
    current_model: str
    generated_image_path: str
    generated_image_url: str
    
    # Evaluation
    evaluation_score: float
    evaluation_feedback: str
    evaluation_issues: list[str]
    
    # Tracking
    attempt_number: int
    attempt_history: Annotated[list[dict], operator.add]  # Appends to list
    
    # Final output
    final_image_path: str
    final_score: float
    status: str  # "in_progress", "success", "failed"
    error_message: str


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def image_to_base64(image_path: str) -> str:
    """Convert image to base64 data URI."""
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode()
    
    suffix = Path(image_path).suffix.lower()
    mime_type = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp"
    }.get(suffix, "image/jpeg")
    
    return f"data:{mime_type};base64,{image_data}"


def get_aspect_ratio(image_path: str) -> str:
    """Detect aspect ratio from image."""
    from PIL import Image
    with Image.open(image_path) as img:
        width, height = img.size
        ratio = width / height
        
        if ratio > 1.7:
            return "16:9"
        elif ratio > 1.4:
            return "3:2"
        elif ratio > 1.1:
            return "4:3"
        elif ratio > 0.9:
            return "1:1"
        elif ratio > 0.7:
            return "3:4"
        elif ratio > 0.6:
            return "2:3"
        else:
            return "9:16"


def extract_hex_color(color_str: str) -> str:
    """Extract hex code from color string like 'Soft Gold (#F8E231)'."""
    if not color_str:
        return ""
    if "(" in color_str and "#" in color_str:
        start = color_str.find("#")
        end = color_str.find(")", start)
        if end == -1:
            end = len(color_str)
        return color_str[start:end].strip()
    elif color_str.startswith("#"):
        return color_str.split()[0]
    return color_str


def hex_to_color_name(hex_code: str) -> str:
    """Convert hex code to descriptive color name."""
    color_map = {
        "#FF69B4": "vibrant hot pink",
        "#FFC0CB": "soft blush pink",
        "#FFD700": "rich gold",
        "#FFC499": "warm peach",
        "#8B9467": "elegant sage green",
        "#C9C3E6": "soft lavender",
        "#C7B8EA": "muted lavender",
        "#1A1D23": "deep navy blue",
        "#333333": "charcoal grey",
        "#F8E231": "bright gold",
        "#3E8E41": "rich green",
        "#FFF0F0": "soft cream",
    }
    
    hex_upper = hex_code.upper()
    if hex_upper in color_map:
        return color_map[hex_upper]
    
    # Basic color detection
    hex_clean = hex_code.lstrip('#')
    if len(hex_clean) != 6:
        return "neutral tone"
    
    try:
        r, g, b = int(hex_clean[0:2], 16), int(hex_clean[2:4], 16), int(hex_clean[4:6], 16)
    except ValueError:
        return "neutral tone"
    
    if r > 200 and g > 200 and b > 200:
        return "soft white"
    elif r < 50 and g < 50 and b < 50:
        return "deep black"
    elif r > g and r > b:
        return "warm red tone"
    elif g > r and g > b:
        return "natural green"
    elif b > r and b > g:
        return "cool blue"
    else:
        return "neutral tone"


def parse_color_scheme(color_json: dict) -> dict:
    """Parse color scheme from JSON and extract relevant colors."""
    
    # Handle nested "raw_response" structure
    if "raw_response" in color_json:
        color_json = color_json["raw_response"]
    
    couple_coord = color_json.get("couple_coordination", {})
    complementary_schemes = couple_coord.get("complementary_color_schemes", [])
    
    colors = {
        "primary": [],
        "accent": [],
        "color_names": [],
        "theme": ""
    }
    
    if complementary_schemes:
        # Use first scheme
        first_scheme = complementary_schemes[0]
        
        venue_color = extract_hex_color(first_scheme.get("venue_decoration_color", ""))
        bride_color = extract_hex_color(first_scheme.get("bride_color", ""))
        groom_color = extract_hex_color(first_scheme.get("groom_color", ""))
        
        if venue_color:
            colors["primary"].append(venue_color)
        if bride_color:
            colors["primary"].append(bride_color)
        if groom_color:
            colors["accent"].append(groom_color)
        
        colors["theme"] = first_scheme.get("description", "elegant celebration")
    
    # Convert to color names
    all_colors = colors["primary"] + colors["accent"]
    colors["color_names"] = [hex_to_color_name(c) for c in all_colors if c]
    
    return colors


# ============================================================================
# AGENT NODES (Each node is a step in the workflow)
# ============================================================================

def analyze_venue_node(state: AgentState) -> dict:
    """
    NODE 1: Analyze the venue image using a vision model.
    
    This node:
    - Sends the venue image to Claude Vision
    - Identifies decoration areas
    - Notes architectural features
    - Returns analysis for prompt generation
    """
    print("\n" + "="*60)
    print("🔍 NODE: ANALYZE VENUE")
    print("="*60)
    
    try:
        client = Anthropic(api_key=Config.ANTHROPIC_TOKEN)
        
        # Read and encode image
        image_data = image_to_base64(state["venue_image_path"])
        
        # Analysis prompt
        analysis_prompt = """Analyze this venue image for event decoration planning.

Identify and describe:

1. ROOM TYPE: What kind of space is this? (ballroom, outdoor, barn, tent, etc.)

2. ARCHITECTURAL FEATURES:
   - Ceiling type and height (high, low, vaulted, etc.)
   - Wall characteristics
   - Windows and natural light
   - Pillars or columns
   - Floor type

3. DECORATION AREAS (list all areas that can be decorated):
   - Ceiling opportunities
   - Wall spaces
   - Entrance/doorways
   - Aisle or walkway
   - Stage or focal point area
   - Table areas

4. CHALLENGES: Any limitations or obstacles for decoration?

5. RECOMMENDATIONS: What decoration styles would work best here?

Be specific and detailed. This analysis will be used to create decoration prompts."""

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1500,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_data.split(",")[1] if "," in image_data else image_data
                            }
                        },
                        {
                            "type": "text",
                            "text": analysis_prompt
                        }
                    ]
                }
            ]
        )
        
        analysis = response.content[0].text
        
        # Extract decoration areas (simple parsing)
        decoration_areas = []
        area_keywords = ["ceiling", "wall", "entrance", "aisle", "stage", "focal", "table", "window", "arch"]
        for keyword in area_keywords:
            if keyword in analysis.lower():
                decoration_areas.append(keyword)
        
        print(f"✅ Analysis complete!")
        print(f"   Identified areas: {decoration_areas}")
        
        return {
            "venue_analysis": analysis,
            "decoration_areas": decoration_areas,
            "status": "in_progress"
        }
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return {
            "venue_analysis": "Analysis failed - using default assumptions",
            "decoration_areas": ["ceiling", "walls", "entrance", "focal point"],
            "error_message": str(e),
            "status": "in_progress"
        }


def generate_prompt_node(state: AgentState) -> dict:
    """
    NODE 2: Generate a detailed decoration prompt based on analysis.
    
    This node:
    - Uses venue analysis to create targeted prompt
    - Incorporates color scheme from JSON
    - Considers event type, style, and intensity
    - Adjusts based on previous attempts if retrying
    """
    print("\n" + "="*60)
    print("📝 NODE: GENERATE PROMPT")
    print("="*60)
    
    event_type = state["event_type"]
    style = state["style"]
    intensity = state["intensity"]
    analysis = state["venue_analysis"]
    colors = parse_color_scheme(state["color_scheme"])
    attempt = state.get("attempt_number", 1)
    
    # Color information
    color_names = colors.get("color_names", ["warm neutral tones"])
    primary_colors = ", ".join(color_names[:2]) if color_names else "elegant neutral"
    accent_colors = ", ".join(color_names[2:4]) if len(color_names) > 2 else "complementary accents"
    
    # Intensity descriptions
    intensity_desc = {
        "subtle": "tastefully minimal with carefully chosen accents — less is more",
        "moderate": "beautifully balanced with thoughtful arrangements throughout",
        "elaborate": "lavishly abundant with stunning, layered decorations everywhere"
    }
    
    # Style descriptions
    style_desc = {
        "elegant": "luxurious, refined, and sophisticated with high-end finishes",
        "rustic": "natural, warm, and charming with organic textures",
        "modern": "sleek, minimalist, and contemporary with clean lines",
        "romantic": "soft, dreamy, and intimate with delicate touches",
        "classic": "timeless, traditional, and formal with symmetrical arrangements",
        "bohemian": "eclectic, free-spirited, and artistic with layered textiles"
    }
    
    # Build prompt
    prompt = f"""Transform this exact venue for a {event_type} celebration — 
DO NOT change the venue's architecture, walls, floor, ceiling structure, windows, doors, or perspective. 
Keep the EXACT same room layout and camera angle.

VENUE ANALYSIS:
{analysis[:500]}...

DECORATIONS TO ADD:

1. FOCAL POINT: Create a stunning central decoration appropriate for a {event_type} — 
   such as a decorated arch, mandap, stage backdrop, or floral installation.

2. CEILING: Add elegant ceiling decorations — chandeliers, draped fabrics, hanging florals, 
   fairy lights, or paper lanterns depending on the style.

3. WALLS: Add tasteful wall treatments — fabric draping, greenery walls, uplighting, 
   or decorative panels.

4. AISLE/WALKWAY: If visible, decorate with flower petals, lanterns, floral stands, 
   or lined arrangements.

5. TABLES: If visible, add beautiful centerpieces, candles, and table settings.

6. ENTRANCE: Add welcoming decorations near any visible doorways or entrances.

STYLE: Make everything {style_desc.get(style, 'elegant and beautiful')}.

COLOR PALETTE: Incorporate subtle hints of {primary_colors} as primary colors, 
with touches of {accent_colors} as accents. Colors should appear naturally in 
flowers, fabrics, lighting, and decorative elements.

INTENSITY: {intensity_desc.get(intensity, 'moderate decorations')}.

CRITICAL RULES:
- Keep the EXACT same venue structure and architecture
- Only ADD decorations — never remove or change existing elements
- Maintain the same camera angle and perspective
- Make decorations look realistic and professionally done
- Ensure cohesive, harmonious design

Output: Photorealistic professional event photography, high resolution, stunning transformation."""

    # If retrying, add adjustment notes
    if attempt > 1 and state.get("evaluation_issues"):
        issues = state["evaluation_issues"]
        prompt += f"""

IMPORTANT ADJUSTMENTS (based on previous attempt feedback):
- Issues to fix: {', '.join(issues[:3])}
- Please pay special attention to these problems in this attempt."""

    print(f"✅ Prompt generated!")
    print(f"   Event: {event_type}, Style: {style}, Intensity: {intensity}")
    print(f"   Colors: {primary_colors}")
    print(f"   Attempt: {attempt}")
    
    return {
        "current_prompt": prompt
    }


def create_image_node(state: AgentState) -> dict:
    """
    NODE 3: Generate the decorated image using Replicate API.
    
    This node:
    - Calls the image generation model
    - Handles different model configurations
    - Saves the result locally
    """
    print("\n" + "="*60)
    print("🎨 NODE: CREATE IMAGE")
    print("="*60)
    
    os.environ["REPLICATE_API_TOKEN"] = Config.REPLICATE_TOKEN
    
    model = state.get("current_model", Config.DEFAULT_MODEL)
    prompt = state["current_prompt"]
    image_path = state["venue_image_path"]
    attempt = state.get("attempt_number", 1)
    
    print(f"   Model: {model}")
    print(f"   Attempt: {attempt}")
    
    try:
        image_uri = image_to_base64(image_path)
        
        # Negative prompt
        negative = "blurry, low quality, distorted architecture, different room, cartoon, anime, unrealistic"
        
        if model == "sdxl":
            output = replicate.run(
                Config.MODELS["sdxl"],
                input={
                    "image": image_uri,
                    "prompt": prompt,
                    "negative_prompt": negative,
                    "prompt_strength": Config.DEFAULT_STRENGTH + (attempt * 0.05),  # Increase strength on retries
                    "num_inference_steps": 35,
                    "guidance_scale": 8.0,
                    "scheduler": "K_EULER_ANCESTRAL"
                }
            )
        elif model == "flux_kontext":
            output = replicate.run(
                Config.MODELS["flux_kontext"],
                input={
                    "image": image_uri,
                    "prompt": prompt,
                    "guidance_scale": 3.5,
                    "output_format": "jpg",
                    "safety_tolerance": 5,
                    "aspect_ratio": "match_input_image"
                }
            )
        else:
            # Default to SDXL
            output = replicate.run(
                Config.MODELS["sdxl"],
                input={
                    "image": image_uri,
                    "prompt": prompt,
                    "negative_prompt": negative,
                    "prompt_strength": Config.DEFAULT_STRENGTH,
                    "num_inference_steps": 30,
                    "guidance_scale": 7.5,
                    "scheduler": "K_EULER"
                }
            )
        
        # Get result URL
        if isinstance(output, list):
            result_url = output[0] if output else None
        elif hasattr(output, 'url'):
            result_url = output.url
        else:
            result_url = str(output)
        
        if not result_url:
            raise Exception("No output URL received")
        
        # Download and save image
        output_path = f"decorated_attempt_{attempt}.jpg"
        response = requests.get(result_url)
        with open(output_path, 'wb') as f:
            f.write(response.content)
        
        print(f"✅ Image generated!")
        print(f"   Saved to: {output_path}")
        
        return {
            "generated_image_url": result_url,
            "generated_image_path": output_path
        }
        
    except Exception as e:
        print(f"❌ Image generation failed: {e}")
        return {
            "generated_image_url": "",
            "generated_image_path": "",
            "error_message": str(e)
        }


def evaluate_node(state: AgentState) -> dict:
    """
    NODE 4: Evaluate the generated image using an LLM judge.
    
    This node:
    - Compares original and decorated images
    - Scores on multiple criteria
    - Provides specific feedback for improvement
    """
    print("\n" + "="*60)
    print("⚖️ NODE: EVALUATE RESULT")
    print("="*60)
    
    if not state.get("generated_image_path") or not os.path.exists(state["generated_image_path"]):
        print("❌ No image to evaluate")
        return {
            "evaluation_score": 0.0,
            "evaluation_feedback": "No image was generated",
            "evaluation_issues": ["Image generation failed"]
        }
    
    try:
        client = Anthropic(api_key=Config.ANTHROPIC_TOKEN)
        
        # Encode both images
        original_b64 = image_to_base64(state["venue_image_path"])
        decorated_b64 = image_to_base64(state["generated_image_path"])
        
        evaluation_prompt = f"""You are an expert event decoration evaluator. Compare these two images:

IMAGE 1: Original venue (before decoration)
IMAGE 2: Decorated venue (after AI decoration)

The decoration requirements were:
- Event type: {state['event_type']}
- Style: {state['style']}
- Intensity: {state['intensity']}

Evaluate on these criteria (score 1-10 for each):

1. STRUCTURE PRESERVATION: Is the venue architecture EXACTLY the same? (walls, floor, ceiling, windows)
2. DECORATION QUALITY: Are decorations appropriate, realistic, and well-placed?
3. STYLE MATCH: Does it match the requested {state['style']} style?
4. COLOR SCHEME: Are the requested colors subtly incorporated?
5. REALISM: Do decorations look photorealistic and professional?
6. COMPLETENESS: Are key areas decorated (focal point, ceiling, etc.)?
7. COHERENCE: Do all elements work together harmoniously?

Respond in this exact JSON format:
{{
    "scores": {{
        "structure_preservation": <1-10>,
        "decoration_quality": <1-10>,
        "style_match": <1-10>,
        "color_scheme": <1-10>,
        "realism": <1-10>,
        "completeness": <1-10>,
        "coherence": <1-10>
    }},
    "overall_score": <1-10 weighted average>,
    "feedback": "<2-3 sentence overall assessment>",
    "issues": ["<specific issue 1>", "<specific issue 2>"],
    "suggestions": ["<improvement suggestion 1>", "<improvement suggestion 2>"]
}}

Be critical but fair. Structure preservation is most important."""

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": original_b64.split(",")[1] if "," in original_b64 else original_b64
                            }
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": decorated_b64.split(",")[1] if "," in decorated_b64 else decorated_b64
                            }
                        },
                        {
                            "type": "text",
                            "text": evaluation_prompt
                        }
                    ]
                }
            ]
        )
        
        # Parse response
        response_text = response.content[0].text
        
        # Try to extract JSON from response
        try:
            # Find JSON in response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                evaluation = json.loads(json_str)
            else:
                raise ValueError("No JSON found")
        except:
            # Fallback parsing
            evaluation = {
                "overall_score": 5.0,
                "feedback": response_text[:200],
                "issues": ["Could not parse detailed evaluation"],
                "suggestions": ["Please retry"]
            }
        
        score = float(evaluation.get("overall_score", 5.0))
        feedback = evaluation.get("feedback", "Evaluation complete")
        issues = evaluation.get("issues", [])
        
        print(f"✅ Evaluation complete!")
        print(f"   Score: {score}/10")
        print(f"   Feedback: {feedback[:100]}...")
        if issues:
            print(f"   Issues: {issues[:2]}")
        
        # Record this attempt in history
        attempt_record = {
            "attempt": state.get("attempt_number", 1),
            "model": state.get("current_model", Config.DEFAULT_MODEL),
            "score": score,
            "issues": issues
        }
        
        return {
            "evaluation_score": score,
            "evaluation_feedback": feedback,
            "evaluation_issues": issues,
            "attempt_history": [attempt_record]  # Will be appended to list
        }
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return {
            "evaluation_score": 0.0,
            "evaluation_feedback": f"Evaluation error: {str(e)}",
            "evaluation_issues": ["Evaluation failed"],
            "attempt_history": [{
                "attempt": state.get("attempt_number", 1),
                "error": str(e)
            }]
        }


def route_decision_node(state: AgentState) -> dict:
    """
    NODE 5: Decide what to do next based on evaluation.
    
    This node doesn't change state much, but prepares for routing.
    """
    print("\n" + "="*60)
    print("🔀 NODE: ROUTE DECISION")
    print("="*60)
    
    score = state.get("evaluation_score", 0)
    attempt = state.get("attempt_number", 1)
    
    print(f"   Current score: {score}/10")
    print(f"   Attempt: {attempt}/{Config.MAX_ATTEMPTS}")
    print(f"   Min passing: {Config.MIN_PASSING_SCORE}")
    
    if score >= Config.MIN_PASSING_SCORE:
        print("   Decision: ✅ PASS - Quality threshold met!")
    elif attempt >= Config.MAX_ATTEMPTS:
        print("   Decision: ⚠️ MAX ATTEMPTS - Returning best result")
    elif score < 5.0 and attempt < Config.MAX_ATTEMPTS:
        print("   Decision: 🔄 RETRY with different model")
    else:
        print("   Decision: 🔄 RETRY with same model")
    
    return {}


def finalize_success_node(state: AgentState) -> dict:
    """
    NODE 6A: Finalize successful result.
    """
    print("\n" + "="*60)
    print("🎉 NODE: FINALIZE SUCCESS")
    print("="*60)
    
    # Copy to final output
    final_path = "final_decorated_venue.jpg"
    if state.get("generated_image_path") and os.path.exists(state["generated_image_path"]):
        import shutil
        shutil.copy(state["generated_image_path"], final_path)
    
    print(f"✅ Success! Final image: {final_path}")
    print(f"   Score: {state.get('evaluation_score', 0)}/10")
    print(f"   Attempts: {state.get('attempt_number', 1)}")
    
    return {
        "final_image_path": final_path,
        "final_score": state.get("evaluation_score", 0),
        "status": "success"
    }


def retry_same_model_node(state: AgentState) -> dict:
    """
    NODE 6B: Prepare for retry with same model.
    """
    print("\n" + "="*60)
    print("🔄 NODE: RETRY (Same Model)")
    print("="*60)
    
    new_attempt = state.get("attempt_number", 1) + 1
    print(f"   Incrementing to attempt {new_attempt}")
    
    return {
        "attempt_number": new_attempt
    }


def retry_switch_model_node(state: AgentState) -> dict:
    """
    NODE 6C: Prepare for retry with different model.
    """
    print("\n" + "="*60)
    print("🔄 NODE: RETRY (Switch Model)")
    print("="*60)
    
    current_model = state.get("current_model", Config.DEFAULT_MODEL)
    new_model = Config.FALLBACK_MODEL if current_model == Config.DEFAULT_MODEL else Config.DEFAULT_MODEL
    new_attempt = state.get("attempt_number", 1) + 1
    
    print(f"   Switching from {current_model} to {new_model}")
    print(f"   Incrementing to attempt {new_attempt}")
    
    return {
        "current_model": new_model,
        "attempt_number": new_attempt
    }


def finalize_max_attempts_node(state: AgentState) -> dict:
    """
    NODE 6D: Finalize when max attempts reached.
    """
    print("\n" + "="*60)
    print("⚠️ NODE: FINALIZE (Max Attempts)")
    print("="*60)
    
    # Find best attempt from history
    history = state.get("attempt_history", [])
    best_score = state.get("evaluation_score", 0)
    
    final_path = "final_decorated_venue.jpg"
    if state.get("generated_image_path") and os.path.exists(state["generated_image_path"]):
        import shutil
        shutil.copy(state["generated_image_path"], final_path)
    
    print(f"⚠️ Max attempts reached. Best score: {best_score}/10")
    print(f"   Final image: {final_path}")
    
    return {
        "final_image_path": final_path,
        "final_score": best_score,
        "status": "completed_max_attempts"
    }


# ============================================================================
# ROUTING FUNCTION
# ============================================================================

def route_after_evaluation(state: AgentState) -> str:
    """
    Conditional edge: Decide which node to go to after evaluation.
    
    Returns the name of the next node.
    """
    score = state.get("evaluation_score", 0)
    attempt = state.get("attempt_number", 1)
    
    # Success - quality threshold met
    if score >= Config.MIN_PASSING_SCORE:
        return "finalize_success"
    
    # Max attempts reached
    if attempt >= Config.MAX_ATTEMPTS:
        return "finalize_max_attempts"
    
    # Low score - try different model
    if score < 5.0:
        return "retry_switch_model"
    
    # Medium score - retry same model with adjustments
    return "retry_same_model"


# ============================================================================
# BUILD THE GRAPH
# ============================================================================

def build_venue_decorator_agent():
    """
    Build the LangGraph agent for venue decoration.
    
    Graph structure:
    
    START → analyze_venue → generate_prompt → create_image → evaluate → route_decision
                                                                              │
                    ┌──────────────────────┬──────────────────┬───────────────┤
                    │                      │                  │               │
                    ▼                      ▼                  ▼               ▼
            finalize_success    retry_same_model    retry_switch_model   finalize_max
                    │                      │                  │               │
                    ▼                      └────────┬─────────┘               ▼
                   END                              │                        END
                                                    ▼
                                            generate_prompt (loop back)
    """
    
    # Create the graph with our state type
    workflow = StateGraph(AgentState)
    
    # Add all nodes
    workflow.add_node("analyze_venue", analyze_venue_node)
    workflow.add_node("generate_prompt", generate_prompt_node)
    workflow.add_node("create_image", create_image_node)
    workflow.add_node("evaluate", evaluate_node)
    workflow.add_node("route_decision", route_decision_node)
    workflow.add_node("finalize_success", finalize_success_node)
    workflow.add_node("retry_same_model", retry_same_model_node)
    workflow.add_node("retry_switch_model", retry_switch_model_node)
    workflow.add_node("finalize_max_attempts", finalize_max_attempts_node)
    
    # Set entry point
    workflow.set_entry_point("analyze_venue")
    
    # Add edges (linear flow)
    workflow.add_edge("analyze_venue", "generate_prompt")
    workflow.add_edge("generate_prompt", "create_image")
    workflow.add_edge("create_image", "evaluate")
    workflow.add_edge("evaluate", "route_decision")
    
    # Add conditional edges from route_decision
    workflow.add_conditional_edges(
        "route_decision",
        route_after_evaluation,
        {
            "finalize_success": "finalize_success",
            "finalize_max_attempts": "finalize_max_attempts",
            "retry_same_model": "retry_same_model",
            "retry_switch_model": "retry_switch_model"
        }
    )
    
    # Retry nodes loop back to generate_prompt
    workflow.add_edge("retry_same_model", "generate_prompt")
    workflow.add_edge("retry_switch_model", "generate_prompt")
    
    # Final nodes go to END
    workflow.add_edge("finalize_success", END)
    workflow.add_edge("finalize_max_attempts", END)
    
    # Compile the graph with memory (for state persistence)
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    return app


# ============================================================================
# MAIN AGENT CLASS
# ============================================================================

class VenueDecoratorAgent:
    """
    The main agent class that wraps the LangGraph workflow.
    
    Usage:
        agent = VenueDecoratorAgent(
            replicate_token="your_token",
            anthropic_token="your_token"
        )
        
        result = agent.decorate(
            image_path="venue.jpg",
            color_json={"..."},
            event_type="wedding",
            style="elegant",
            intensity="moderate"
        )
    """
    
    def __init__(
        self,
        replicate_token: str,
        anthropic_token: str,
        max_attempts: int = 3,
        min_passing_score: float = 7.0
    ):
        """Initialize the agent with API tokens."""
        
        # Set configuration
        Config.REPLICATE_TOKEN = replicate_token
        Config.ANTHROPIC_TOKEN = anthropic_token
        Config.MAX_ATTEMPTS = max_attempts
        Config.MIN_PASSING_SCORE = min_passing_score
        
        # Build the graph
        self.app = build_venue_decorator_agent()
        
        print("="*60)
        print("🤖 VENUE DECORATOR AGENT INITIALIZED")
        print("="*60)
        print(f"   Max attempts: {max_attempts}")
        print(f"   Min passing score: {min_passing_score}")
        print(f"   Default model: {Config.DEFAULT_MODEL}")
        print(f"   Fallback model: {Config.FALLBACK_MODEL}")
        print("="*60)
    
    def decorate(
        self,
        image_path: str,
        color_json: dict,
        event_type: str = "wedding",
        style: str = "elegant",
        intensity: str = "moderate"
    ) -> dict:
        """
        Run the agent to decorate a venue.
        
        Args:
            image_path: Path to venue image
            color_json: Color scheme dictionary
            event_type: Type of event
            style: Decoration style
            intensity: Decoration intensity
            
        Returns:
            Final state with results
        """
        
        # Validate inputs
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Initial state
        initial_state = {
            "venue_image_path": image_path,
            "color_scheme": color_json,
            "event_type": event_type,
            "style": style,
            "intensity": intensity,
            "attempt_number": 1,
            "current_model": Config.DEFAULT_MODEL,
            "attempt_history": [],
            "status": "in_progress"
        }
        
        print("\n" + "="*60)
        print("🚀 STARTING VENUE DECORATION AGENT")
        print("="*60)
        print(f"   Image: {image_path}")
        print(f"   Event: {event_type}")
        print(f"   Style: {style}")
        print(f"   Intensity: {intensity}")
        print("="*60)
        
        # Run the graph
        config = {"configurable": {"thread_id": "venue_decoration_1"}}
        
        # Stream through the graph to see progress
        final_state = None
        for event in self.app.stream(initial_state, config):
            # Each event is a dict with node name as key
            for node_name, node_output in event.items():
                if node_output:
                    # Update tracking of final state
                    if final_state is None:
                        final_state = initial_state.copy()
                    final_state.update(node_output)
        
        # Print final summary
        print("\n" + "="*60)
        print("📋 AGENT EXECUTION COMPLETE")
        print("="*60)
        print(f"   Status: {final_state.get('status', 'unknown')}")
        print(f"   Final Score: {final_state.get('final_score', 'N/A')}/10")
        print(f"   Total Attempts: {final_state.get('attempt_number', 1)}")
        print(f"   Output: {final_state.get('final_image_path', 'N/A')}")
        print("="*60)
        
        return final_state


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    """Command line interface."""
    
    # ==========================================================================
    # ⚠️ CONFIGURATION - CHANGE THESE VALUES BEFORE RUNNING
    # ==========================================================================
    
    REPLICATE_TOKEN = "YOUR_REPLICATE_TOKEN"  # ← Replace
    ANTHROPIC_TOKEN = "YOUR_ANTHROPIC_TOKEN"  # ← Replace
    
    IMAGE_PATH = "/path/to/your/venue.jpg"  # ← Replace
    
    COLOR_JSON = {
        "raw_response": {
            "couple_coordination": {
                "complementary_color_schemes": [
                    {
                        "bride_color": "Dusty Rose (#C9C3E6)",
                        "groom_color": "Navy Blue (#1A1D23)",
                        "venue_decoration_color": "Soft Gold (#F8E231)",
                        "description": "Elegant dusty rose and navy combination"
                    }
                ]
            }
        }
    }
    
    EVENT_TYPE = "wedding"
    STYLE = "elegant"
    INTENSITY = "elaborate"
    
    # ==========================================================================
    # END CONFIGURATION
    # ==========================================================================
    
    # Validate
    if REPLICATE_TOKEN == "YOUR_REPLICATE_TOKEN":
        print("❌ Please set your REPLICATE_TOKEN")
        return
    
    if ANTHROPIC_TOKEN == "YOUR_ANTHROPIC_TOKEN":
        print("❌ Please set your ANTHROPIC_TOKEN")
        return
    
    if not os.path.exists(IMAGE_PATH):
        print(f"❌ Image not found: {IMAGE_PATH}")
        return
    
    # Create and run agent
    agent = VenueDecoratorAgent(
        replicate_token=REPLICATE_TOKEN,
        anthropic_token=ANTHROPIC_TOKEN,
        max_attempts=3,
        min_passing_score=7.0
    )
    
    result = agent.decorate(
        image_path=IMAGE_PATH,
        color_json=COLOR_JSON,
        event_type=EVENT_TYPE,
        style=STYLE,
        intensity=INTENSITY
    )
    
    # Print result
    if result.get("status") == "success":
        print(f"\n🎉 SUCCESS! Decorated venue saved to: {result.get('final_image_path')}")
    else:
        print(f"\n⚠️ Completed with status: {result.get('status')}")
        print(f"   Best result saved to: {result.get('final_image_path')}")


if __name__ == "__main__":
    main()