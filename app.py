from dotenv import load_dotenv
import chainlit as cl
from agents.base_agent import Agent
import base64

load_dotenv()

# Note: If switching to LangSmith, uncomment the following, and replace @observe with @traceable
# from langsmith.wrappers import wrap_openai
# from langsmith import traceable
# client = wrap_openai(openai.AsyncClient())

from langfuse.decorators import observe
from langfuse.openai import AsyncOpenAI
 
client = AsyncOpenAI()

gen_kwargs = {
    "model": "gpt-4o",
    "temperature": 0.2
}

SYSTEM_PROMPT = """\
You are a pirate.
"""

PLANNING_PROMPT = """\
You are a software architect, preparing to build the web page in the image that the user sends. 
Once they send an image, generate a plan, described below, in markdown format.

If the user or reviewer confirms the plan is good, available tools to save it as an artifact \
called `plan.md`. If the user has feedback on the plan, revise the plan, and save it using \
the tool again. A tool is available to update the artifact. Your role is only to plan the \
project. You will not implement the plan, and will not write any code. 

If the user requests to implement a milestone, there is an available tool to call an agent. \
You should delegate to the implementation agent. 

If the plan has already been saved, no need to save it again unless there is feedback. Do not \
use the tool again if there are no changes.

For the contents of the markdown-formatted plan, create two sections, "Overview" and "Milestones".

In a section labeled "Overview", analyze the image, and describe the elements on the page, \
their positions, and the layout of the major sections.

Using vanilla HTML and CSS, discuss anything about the layout that might have different \
options for implementation. Review pros/cons, and recommend a course of action.

In a section labeled "Milestones", describe an ordered set of milestones for methodically \
building the web page, so that errors can be detected and corrected early. Pay close attention \
to the aligment of elements, and describe clear expectations in each milestone. Do not include \
testing milestones, just implementation.

Milestones should be formatted like this:

 - [ ] 1. This is the first milestone
 - [ ] 2. This is the second milestone
 - [ ] 3. This is the third milestone
"""

IMPLEMENTATION_PROMPT = """\
You are the Implementation Agent. Your role is to read the `plan.md` file and implement ONE milestone at a time from the plan. To maximize the chances of success, you should take small, incremental steps that help achieve the goal efficiently and effectively.

Alternatively, if feedback is provided, incorporate that feedback to improve or fix a milestone.

For every milestone you tackle:
1. **Identify and Implement**: Choose one milestone from `plan.md` that has not yet been completed, and focus solely on implementing it. Alternatively, if feedback is provided, use it to enhance the existing implementation.
2. **Generate Artifacts**: Write or update the `index.html` and `styles.css` files in the `artifacts` folder to reflect the work done for the selected milestone.
3. **Guide Implementation**: Clearly articulate and document the changes being made to achieve the milestone.
4. **Update the Plan**: After completing a milestone, mark it as completed in `plan.md`. Ensure that it is clear that this milestone is now addressed.

Ensure that all steps are small, manageable, and clearly documented to reduce complexity and improve success rates.

Your goal is to consistently take incremental steps, achieve milestones, generate the appropriate artifacts (`index.html` and `styles.css`), and reflect your progress in `plan.md` as you move through the milestones.
"""

# Create an instance of the Agent class
planning_agent = Agent(name="Planning Agent", client=client, prompt=PLANNING_PROMPT)
implementation_agent = Agent(name="implementation", client=client, prompt=IMPLEMENTATION_PROMPT)

@observe
@cl.on_chat_start
def on_chat_start():    
    message_history = [{"role": "system", "content": SYSTEM_PROMPT}]
    cl.user_session.set("message_history", message_history)

@observe
async def generate_response(client, message_history, gen_kwargs):
    response_message = cl.Message(content="")
    await response_message.send()

    stream = await client.chat.completions.create(messages=message_history, stream=True, **gen_kwargs)
    async for part in stream:
        if token := part.choices[0].delta.content or "":
            await response_message.stream_token(token)
    
    await response_message.update()

    return response_message

@cl.on_message
@observe
async def on_message(message: cl.Message):
    message_history = cl.user_session.get("message_history", [])

    # Processing images exclusively
    images = [file for file in message.elements if "image" in file.mime] if message.elements else []

    if images:
        # Read the first image and encode it to base64
        with open(images[0].path, "rb") as f:
            base64_image = base64.b64encode(f.read()).decode('utf-8')
        message_history.append({
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": message.content
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        })
    else:
        message_history.append({"role": "user", "content": message.content})
    
    response_message = await planning_agent.execute(message_history)

    message_history.append({"role": "assistant", "content": response_message})
    cl.user_session.set("message_history", message_history)

if __name__ == "__main__":
    cl.main()
