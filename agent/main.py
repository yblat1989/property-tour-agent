"""
Property Tour AI Agent — LiveKit + Gemini Live
No external data dependencies — uses listing data passed via room metadata.

Environment variables (set in Railway dashboard):
  LIVEKIT_URL         wss://your-app.livekit.cloud
  LIVEKIT_API_KEY     your livekit api key
  LIVEKIT_API_SECRET  your livekit api secret
  GOOGLE_API_KEY      your gemini api key
"""

import json
import logging
import os

from dotenv import load_dotenv
from livekit.agents import Agent, AgentSession, JobContext, RoomInputOptions, WorkerOptions, cli
from livekit.plugins import google, silero

load_dotenv()
os.environ.setdefault("LIVEKIT_URL", "wss://realaitour-8egrtmu5.livekit.cloud")
os.environ.setdefault("LIVEKIT_API_KEY", "APIGNNqKxoa8TCd")
os.environ.setdefault("LIVEKIT_API_SECRET", "6Ly652vPoJJfeAniXAqnhP0QWWvhsNofqVPadPw67BCB")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_system_prompt(metadata: dict) -> str:
    address  = metadata.get("address", "this property")
    price    = metadata.get("price")
    beds     = metadata.get("beds")
    baths    = metadata.get("baths")
    sqft     = metadata.get("sqft")
    year     = metadata.get("yearBuilt")
    mls      = metadata.get("mlsNumber", "N/A")
    desc     = metadata.get("description", "")
    features = metadata.get("features", [])

    return f"""You are an expert AI real estate tour guide. A buyer is self-touring this property RIGHT NOW using their phone camera and microphone. You can see what they see and hear what they say.

PROPERTY:
- Address: {address}
- List Price: {"${:,.0f}".format(price) if price else "See listing"}
- Beds/Baths: {beds or "?"}bd / {baths or "?"}ba
- Square Feet: {"{:,.0f}".format(sqft) if sqft else "See listing"}
- Year Built: {year or "Unknown"}
- MLS #: {mls}
{f"- Description: {desc}" if desc else ""}
{f"- Features: {', '.join(features)}" if features else ""}

YOUR RULES:
1. Keep every response to 1-3 sentences. The buyer is walking and listening.
2. Speak naturally — this is a live voice conversation.
3. When you see a room via camera, name it and call out 1-2 notable features immediately.
4. For appliances: try to identify brand and estimate age from visible labels or wear.
5. FLAG THESE OUT LOUD the moment you see them:
   - Water stains or discoloration on ceilings or walls
   - Cracks in walls, floors, or visible foundation
   - Mold, peeling paint, or efflorescence (white powder on concrete)
   - Outdated electrical panels (Federal Pacific, Zinsco, fuse boxes)
   - Floors that look uneven or doors that appear off-square
   - HVAC equipment that looks very old or poorly maintained
6. Answer questions about the property factually using the listing data above.
7. If you don't know something, say so in one sentence — never guess.
8. Never recommend buying or not buying. Observations and facts only.

Start with a single friendly sentence welcoming the buyer and letting them know you can see through their camera."""


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    metadata = {}
    try:
        metadata = json.loads(ctx.room.metadata or "{}")
    except json.JSONDecodeError:
        logger.warning("Could not parse room metadata")

    logger.info(f"Tour starting for: {metadata.get('address', 'unknown address')}")

    session = AgentSession(
        vad=silero.VAD.load(),
        llm=google.realtime.RealtimeModel(
            model="gemini-2.5-flash-native-audio-preview-12-2025",
            voice="Puck",
            temperature=0.7,
            instructions=build_system_prompt(metadata),
        ),
    )

    await session.start(
        agent=Agent(
            instructions=build_system_prompt(metadata),
        ),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            video_enabled=True,
        ),
    )

    logger.info("Agent session started successfully")


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="property-tour-agent",
        )
    )
