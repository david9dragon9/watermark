from text_v2.vllm_watermark import VLLMWatermarkGenerator
import gradio as gr

model_name = "meta-llama/Llama-3.2-1B-Instruct"  # only Llama-3.2-1B-Instruct is supported because I need to squeeze it in Colab!


def main():
    generator = VLLMWatermarkGenerator(model_name=model_name)
    # Two tabs: Watermark and Detect
    with gr.Blocks() as demo:
        gr.Markdown(
            """## Watermark Generation!!!\nTo generate watermarked text, simply input a prompt along with a password and hash token window, which controls how many of the last few tokens in context are used to determine the watermark.
        Then, click generate to output the generated text. To detect the watermark, simply input the text along with a hash token window and password to get a p-value. If this p-value is below 0.05, then it is very likely that the text is watermarked. Note that the p-value is expressed in scientific notation (e.g. 2e-3 = 0.002).
        How it works: When generating the watermark, before generating each token, the LLM will randomly select around half of the tokens to be red list tokens that are randomly excluded. Then, the LLM will choose among the remaining tokens. At detection time, we check to see how many tokens would have been on the red list. If there is suspiciously low number of watermarked tokens, then the text is likely watermarked."""
        )
        with gr.Tab("Watermark Generation"):
            with gr.Row():
                prompt = gr.Textbox(
                    label="Prompt", placeholder="Enter your prompt here..."
                )
                password = gr.Textbox(
                    label="Password", placeholder="Enter your password here..."
                )
                hash_token_window = gr.Slider(
                    label="Hash Token Window",
                    minimum=1,
                    maximum=10,
                    step=1,
                    value=4,
                    interactive=True,
                )
                generate_button = gr.Button("Generate Watermarked Text")
                output = gr.Textbox(label="Watermarked Text")

            generate_button.click(
                lambda p, pw, hash_token_window: generator.watermark(
                    p, pw, hash_token_window
                ),
                inputs=[prompt, password, hash_token_window],
                outputs=output,
            )

        gr.Markdown("## Watermark Detection")
        with gr.Tab("Watermark Detection"):
            with gr.Row():
                response = gr.Textbox(
                    label="Response", placeholder="Enter the response to check..."
                )
                detect_button = gr.Button("Detect Watermark")
                password = gr.Textbox(
                    label="Password", placeholder="Enter your password here..."
                )
                hash_token_window = gr.Slider(
                    label="Hash Token Window",
                    minimum=1,
                    maximum=10,
                    step=1,
                    value=4,
                    interactive=True,
                )
                detection_output = gr.Textbox(label="Detection Result")

            detect_button.click(
                lambda r, pw, hash_token_window: generator.detect(
                    r, pw, hash_token_window
                ),
                inputs=[response, password, hash_token_window],
                outputs=detection_output,
            )
    demo.launch(share=True)  # 🚀


if __name__ == "__main__":
    main()
