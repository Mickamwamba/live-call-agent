""" Simplified Agent UI """
import gradio as gr
from agent import build_graph
from transcriber import AudioTranscriber
from langchain_core.messages import HumanMessage

class BasicAgent:
    def __init__(self):
        self.graph = build_graph()

    def __call__(self, question: str) -> str:
        messages = [HumanMessage(content=question)]
        messages = self.graph.invoke({"messages": messages})
        answer = messages['messages'][-1].content
        return answer[14:] if len(answer) > 14 else answer

def run_agent_analysis(progress=gr.Progress()):
    try:
     
        agent = BasicAgent()
        result = agent("Give the suggestions based on the transcribed audio")
        return (
            "✅ Analysis completed successfully!",
            "🔄 Listening...",
            result
        )

    except Exception as e:
        return (
            f"❌ Error during analysis: {str(e)}",
            "Error occurred",
            "No recommendations available due to error"
        )

with gr.Blocks(css="""
body { background-color: #ffffff; font-family: 'Helvetica Neue', sans-serif; }
h1, h2, h3 { color: #1e3a8a; }
button { background-color: #1e3a8a; color: white; font-weight: bold; border: none; border-radius: 5px; padding: 12px 20px; }
button:hover { background-color: #3b82f6; }
textarea, .gr-box { border: 1px solid #d1d5db; border-radius: 5px; padding: 10px; font-family: monospace; background-color: #f9fafb; color: #111827; }
""") as demo:
    
    gr.Markdown("# 🎧 Live Call Assistant\nReal-time transcription and actionable insights")
    
    start_button = gr.Button("Start Live Call Analysis")

    with gr.Row():
        transcript_output = gr.Textbox(label="Live Transcript", lines=15, interactive=False)
        insights_output = gr.Textbox(label="Real-Time Suggestions", lines=15, interactive=False)
    
    status_output = gr.Textbox(label="Status", interactive=False)
    
    start_button.click(fn=run_agent_analysis, inputs=[], outputs=[status_output, transcript_output, insights_output])


    run_agent_analysis

if __name__ == "__main__":
    demo.launch(debug=True)
