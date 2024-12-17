import discord
import cv2
import numpy as np
import requests
from io import BytesIO

# Discord bot token
TOKEN = "--"  # Replace with your bot's token

# Define the mask URLs
map_mask_url = "https://i.ibb.co/0n1QfDg/mapmask.png"
mountain_mask_url = "https://i.ibb.co/NCQcsjQ/BM-M.png"  
#mountian mask map images below:
# Broken Moon:  "https://i.ibb.co/NCQcsjQ/BM-M.png"  
# Worlds Edge: "https://i.ibb.co/7JgyZVY/WE-M.png"

# Intents for the bot
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
intents.reactions = True

client = discord.Client(intents=intents)

# Track previous predictions to avoid duplicates
previous_predictions = set()

@client.event
async def on_ready():
    print(f"We have logged in as {client.user}")

async def generate_prediction(screenshot, map_mask, mountain_mask):
    """
    Generate a valid Ring 3 prediction with correct scaling and constraints.
    """
    # Ensure masks and screenshot are the same size
    if screenshot.shape[:2] != map_mask.shape[:2]:
        map_mask = cv2.resize(map_mask, (screenshot.shape[1], screenshot.shape[0]))
    if screenshot.shape[:2] != mountain_mask.shape[:2]:
        mountain_mask = cv2.resize(mountain_mask, (screenshot.shape[1], screenshot.shape[0]))

    # Apply the map mask to the screenshot
    alpha_channel = map_mask[:, :, 3]
    binary_map_mask = (alpha_channel > 0).astype(np.uint8)
    isolated_map = cv2.bitwise_and(screenshot, screenshot, mask=binary_map_mask)

    # Convert to grayscale and blur for circle detection
    gray_image = cv2.cvtColor(isolated_map, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (9, 9), 2)

    # Detect zones
    small_circles = cv2.HoughCircles(
        blurred_image, cv2.HOUGH_GRADIENT, dp=1.0, minDist=200,
        param1=100, param2=30, minRadius=50, maxRadius=300
    )
    large_circles = cv2.HoughCircles(
        blurred_image, cv2.HOUGH_GRADIENT, dp=1.0, minDist=200,
        param1=100, param2=40, minRadius=300, maxRadius=600
    )

    # Prepare output image
    output_image = isolated_map.copy()

    if small_circles is not None and large_circles is not None:
        # Extract circles
        small_circle = np.round(small_circles[0, 0]).astype("int")
        large_circle = np.round(large_circles[0, 0]).astype("int")
        x_ring2, y_ring2, r_ring2 = small_circle
        x_ring1, y_ring1, r_ring1 = large_circle

        # Calculate Ring 3 radius dynamically
        r_ring3 = int(r_ring2 * 0.615)  # 61.5% of Ring 2's radius

        # Predict Ring 3 location
        for _ in range(100):  # Retry up to 100 times
            offset_x = np.random.randint(-r_ring2 // 2, r_ring2 // 2)
            offset_y = np.random.randint(-r_ring2 // 2, r_ring2 // 2)
            x_ring3 = x_ring2 + offset_x
            y_ring3 = y_ring2 + offset_y

            # Ensure Ring 3 is valid
            distance_to_ring2 = np.linalg.norm([x_ring3 - x_ring2, y_ring3 - y_ring2])
            if distance_to_ring2 + r_ring3 <= r_ring2 and (x_ring3, y_ring3) not in previous_predictions:
                previous_predictions.add((x_ring3, y_ring3))
                break

        # Draw detected rings and prediction
        cv2.circle(output_image, (x_ring1, y_ring1), r_ring1, (0, 255, 0), 2)  # Green: Ring 1
        cv2.circle(output_image, (x_ring2, y_ring2), r_ring2, (255, 0, 0), 2)  # Blue: Ring 2
        cv2.circle(output_image, (x_ring3, y_ring3), r_ring3, (0, 0, 255), 2)  # Red: Ring 3

        # Save to buffer
        _, buffer = cv2.imencode(".png", output_image)
        return BytesIO(buffer), (x_ring3, y_ring3)
    return None, None

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if message.content.lower() == "!predict":
        await message.channel.send("Please upload a screenshot to process.")

        def check(m):
            return m.author == message.author and m.attachments

        try:
            response = await client.wait_for("message", check=check, timeout=60.0)
            attachment = response.attachments[0]
            screenshot_response = requests.get(attachment.url)
            screenshot_array = np.frombuffer(screenshot_response.content, np.uint8)
            screenshot = cv2.imdecode(screenshot_array, cv2.IMREAD_UNCHANGED)

            # Download masks
            map_mask_response = requests.get(map_mask_url)
            map_mask_array = np.frombuffer(map_mask_response.content, np.uint8)
            map_mask = cv2.imdecode(map_mask_array, cv2.IMREAD_UNCHANGED)

            mountain_mask_response = requests.get(mountain_mask_url)
            mountain_mask_array = np.frombuffer(mountain_mask_response.content, np.uint8)
            mountain_mask = cv2.imdecode(mountain_mask_array, cv2.IMREAD_UNCHANGED)

            # Generate prediction
            output_image_bytes, ring3_coords = await generate_prediction(screenshot, map_mask, mountain_mask)
            if output_image_bytes:
                sent_message = await message.channel.send(
                    "Prediction complete. React with 🔄 to retry.",
                    file=discord.File(output_image_bytes, "prediction.png")
                )

                # Add reaction for retry
                await sent_message.add_reaction("🔄")

                # Store data for retry
                client.ring3_coords = ring3_coords
                client.screenshot = screenshot
                client.map_mask = map_mask
                client.mountain_mask = mountain_mask
            else:
                await message.channel.send("Could not detect the zones. Please try a clearer screenshot.")
        except Exception as e:
            await message.channel.send(f"An error occurred: {str(e)}")

@client.event
async def on_reaction_add(reaction, user):
    if user.bot:
        return

    if reaction.emoji == "🔄" and hasattr(client, "screenshot"):
        # Retry with a new prediction
        output_image_bytes, _ = await generate_prediction(
            client.screenshot, client.map_mask, client.mountain_mask
        )
        if output_image_bytes:
            await reaction.message.channel.send(
                "New prediction:",
                file=discord.File(output_image_bytes, "new_prediction.png")
            )

# Run the bot
client.run(TOKEN)
