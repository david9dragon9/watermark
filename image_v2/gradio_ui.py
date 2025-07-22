import gradio as gr
from image_v2.td_watermark import watermark_image

# image watermarking


def main():
    # two tabs: watermark and detect. Watermark takes in image and password and outputs image. Detect takes in image and password and outputs detection p-value.
    with gr.Blocks() as demo:
        gr.Markdown(
            """## Image Watermark Generation!!!\nTo generate a watermarked image, simply input the original image along with a password and click the generate button. To detect whether an image has a watermark, input the image along with the password and click detect. The resulting number under Detection Result is a p-value that indicates how likely the image is to have certain features if it were not watermarked. A low p-value (generally under 0.05) indicates that the image is watermarked.

How the method works: The method works by using back propagation and gradient descent to find small changes in pixel values of the image that result in a transformed version of the image all satisfying certain constraints. An un-watermarked image is unlikely to satisfy all of the constraints, so the fact that an image satisfies all of the constraints indicates that it must be watermarked."""
        )
        with gr.Tab("Watermark Generation"):
            with gr.Row():
                input_image = gr.Image(label="Input Image", type="pil")
                password = gr.Textbox(
                    label="Password", placeholder="Enter your password here..."
                )
                generate_button = gr.Button("Generate Watermarked Image")
                output_image = gr.Image(label="Watermarked Image", format="png")

            generate_button.click(
                lambda img, pw: watermark_image(
                    img, None, pw, "watermark"
                ),  # Placeholder for watermark generation function
                inputs=[input_image, password],
                outputs=output_image,
            )

        gr.Markdown("## Watermark Detection")
        with gr.Tab("Watermark Detection"):
            with gr.Row():
                detection_image = gr.Image(label="Detection Image", type="pil")
                password = gr.Textbox(
                    label="Password", placeholder="Enter your password here..."
                )
                detect_button = gr.Button("Detect Watermark")
                detection_output = gr.Textbox(label="Detection Result")

            detect_button.click(
                lambda img, pw: watermark_image(
                    None, img, pw, "detect"
                ),  # Placeholder for watermark detection function
                inputs=[detection_image, password],
                outputs=detection_output,
            )
    demo.launch(share=True)


if __name__ == "__main__":
    main()
