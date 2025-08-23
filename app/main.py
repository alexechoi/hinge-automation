# app/main.py

import asyncio
import time
import os
import cv2
import uuid
from dotenv import load_dotenv
from multiprocessing import Process, freeze_support, set_start_method
from ppadb.client import Client as AdbClient

# Import your prompt engine weight updater
from prompt_engine import update_template_weights
from config import GEMINI_API_KEY

# Import your existing helper functions
from helper_functions import (
    connect_device,
    connect_device_remote,
    get_screen_resolution,
    open_hinge,
    swipe,
    capture_screenshot,
    do_comparision,
    find_icon,
    generate_comment,  # If you're using the advanced prompt_engine, you can rename or unify
    tap,
    input_text,
)

# Import Gemini-based image analyzer
from gemini_analyzer import extract_text_from_image_gemini, analyze_profile_with_gemini

# Import data store logic for success-rate tracking
from data_store import (
    store_generated_comment,
    store_feedback,
    calculate_template_success_rates,
)

# Using Gemini API instead of OpenAI


# async def main():
def main():
    # device = connect_device_remote(os.getenv("DEVICE_IP", "127.0.0.1"))
    device = connect_device("127.0.0.1")
    if not device:
        return

    width, height = get_screen_resolution(device)

    # Approximate coordinates based on experimentation
    x_select_like_button_approx = int(width * 0.90)
    # y_select_like_button_approx = int(height * 0.67 * 0.75)
    y_select_like_button_approx = int(height * 0.67)

    x_select_comment_button_approx = 540
    y_select_comment_button_approx = 1755

    x_select_done_button_approx = int(width * 0.85)
    y_select_done_button_approx = int(height * 0.50)

    x_send_like_button = int(width * 0.75)
    y_send_like_button = int(height * 0.80)

    x_dislike_button_approx = int(width * 0.15)
    y_dislike_button_approx = int(height * 0.85)

    x1_swipe = int(width * 0.15)
    x2_swipe = x1_swipe

    y1_swipe = int(height * 0.5)
    y2_swipe = int(y1_swipe * 0.75)

    # Load sample images for matching criteria (like/dislike)
    like_images = [
        cv2.imread(path) for path in ["images/like2.jpeg"] if os.path.exists(path)
    ]
    dislike_images = [
        cv2.imread(path) for path in ["images/dislike.jpeg"] if os.path.exists(path)
    ]

    open_hinge(device=device)
    time.sleep(5)

    previous_profile_text = ""

    # Optionally, run once at the start: recalc success rates & update template weights
    success_rates = calculate_template_success_rates()
    update_template_weights(success_rates)

    for _ in range(10):
        # Swipe to next profile
        swipe(device, x1_swipe, y1_swipe, x2_swipe, y2_swipe)
        screenshot_path = capture_screenshot(device, "screen")

        # Use Gemini API for text extraction from profile images
        current_profile_text = extract_text_from_image_gemini(screenshot_path, GEMINI_API_KEY).strip()
        if not current_profile_text:
            print("Warning: OCR returned empty text.")

        profile_image = cv2.imread(screenshot_path)

        # Compare with sample images
        match_like = do_comparision(profile_image, like_images)
        match_dislike = do_comparision(profile_image, dislike_images)

        print("Calculated scores => Like:", match_like, "Dislike:", match_dislike)

        # Find the Like button
        x_select_like_button, y_select_like_button = find_icon(
            "images/screen.png",
            "images/heart1.png",
            threshold=0.75,
            min_matches=10,
            approx_x=x_select_like_button_approx,
            approx_y=y_select_like_button_approx,
        )

        # Decision-making logic
        if (
            match_like * 0 < match_dislike
            and x_select_like_button is not None
            and y_select_like_button is not None
        ):
            # Generate a comment using your advanced logic or the existing generate_comment
            # For demonstration, let's assume your 'generate_comment' calls GPT-4, etc.
            comment = (
                generate_comment(current_profile_text) or "Hey, I'd love to meet up!"
            )
            print(f"Generated Comment: {comment}")

            # Create a comment_id to track feedback
            comment_id = str(uuid.uuid4())

            # Optionally store the generated comment for analytics
            # If you used a comedic template, "style_used" might be "comedic", etc.
            store_generated_comment(
                comment_id=comment_id,
                profile_text=current_profile_text,
                generated_comment=comment,
                style_used="unknown",  # Could be comedic/flirty if you parse from the template
            )

            # Tap Like
            tap(device, x_select_like_button, y_select_like_button)
            print("Like tapped at:", x_select_like_button, y_select_like_button)

            # Tap to open comment field
            # tap(device, x_select_comment_button_approx, y_select_comment_button_approx)

            # Type the comment (working somewhat)
            # input_text(device, comment)
            # capture_screenshot(device, "screen_after_message")
            # time.sleep(100)
            # swipe(device, width * 0.65, height * 0.82, width * 0.75, height * 0.82)

            # while input_text(device, comment):
            #     capture_screenshot(device, "screen_after_message")
            #     time.sleep(0.5)
            #     tap(
            #         device,
            #         x_select_comment_button_approx,
            #         y_select_comment_button_approx,
            #     )

            # After some period, you could store feedback (maybe you get a callback or check the app)
            # For demonstration, let's just store "match" or "no match" randomly
            # store_feedback(comment_id=comment_id, outcome="match")

        else:
            # If same profile text as previous, might be stuck
            if (
                previous_profile_text == current_profile_text
                and current_profile_text != ""
            ):
                print("Dislike (same profile encountered again)")
            else:
                print("Dislike (new profile or no like match)")

            print(
                "Dislike tapped at:", x_dislike_button_approx, y_dislike_button_approx
            )
            tap(device, x_dislike_button_approx, y_dislike_button_approx)

        previous_profile_text = current_profile_text
        time.sleep(2)

    # After processing 10 profiles, re-check success rates, update template weights
    success_rates = calculate_template_success_rates()
    update_template_weights(success_rates)
    print("Final success rates:", success_rates)
    print("Main loop finished.")


def test():
    height = 1080
    width = 2340
    device = connect_device()
    comment = "Hi"

    x_select_comment_button_approx = 540
    y_select_comment_button_approx = 1755

    swipe(device, width * 0.50, height * 0.70, width * 0.55, height * 0.70)
    tap(device, x_select_comment_button_approx, y_select_comment_button_approx)
    # Type the comment
    input_text(device, comment)


if __name__ == "__main__":
    # async run of main
    # Windows fix for multiprocessing
    # freeze_support()
    # set_start_method("spawn", force=True)
    # asyncio.run(main())

    # Test for checking only input text
    # test()

    main()
