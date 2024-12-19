import discord
import cv2
import numpy as np
import requests
from io import BytesIO
import psutil
import asyncio

# Discord bot token
TOKEN = "--"  # Replace with your bot's token

# Define the mask URLs
map_mask_url = "https://i.ibb.co/0n1QfDg/mapmask.png"
mountain_mask_url = "https://i.ibb.co/YX5nmft/SP-M.png"
# Mountain mask map images below:
# Broken Moon:  "https://i.ibb.co/NCQcsjQ/BM-M.png"
# Worlds Edge: "https://i.ibb.co/7JgyZVY/WE-M.png"
# Storm Point: "https://i.ibb.co/YX5nmft/SP-M.png"

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
    client.loop.create_task(update_status())  # Start the status update loop

def is_valid_ring3(x_ring3, y_ring3, r_ring3, x_ring2, y_ring2, r_ring2, mountain_mask):
    """
    Validates if the generated Ring 3 is playable:
    - At least 50% of its area must be in a playable zone.
    - Ring 3 must be entirely within the bounds of Ring 2.
    """
    # Ensure Ring 3 is entirely within Ring 2
    distance_to_ring2_center = np.linalg.norm([x_ring3 - x_ring2, y_ring3 - y_ring2])
    if distance_to_ring2_center + r_ring3 > r_ring2:
        return False

    # Check playable area using the mountain mask
    mask = np.zeros(mountain_mask.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (x_ring3, y_ring3), r_ring3, 255, -1)
    total_area = np.sum(mask > 0)
    unplayable_area = np.sum((mask > 0) & (mountain_mask[:, :, 0] > 0))

    playable_ratio = (total_area - unplayable_area) / total_area
    return playable_ratio >= 0.5

async def generate_prediction(screenshot, map_mask, mountain_mask):
    """
    Generate a valid Ring 3 prediction with correct scaling and constraints.
    """
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
            offset_angle = np.random.uniform(0, 2 * np.pi)
            max_offset = r_ring2 - r_ring3
            offset_distance = np.random.uniform(0, max_offset)
            offset_x = int(offset_distance * np.cos(offset_angle))
            offset_y = int(offset_distance * np.sin(offset_angle))
            x_ring3 = x_ring2 + offset_x
            y_ring3 = y_ring2 + offset_y

            if is_valid_ring3(x_ring3, y_ring3, r_ring3, x_ring2, y_ring2, r_ring2, mountain_mask):
                previous_predictions.add((x_ring3, y_ring3))
                break

        # Create an overlay for the circles
        overlay = output_image.copy()
        circle_opacity = 0.4  # Adjust the opacity here (0 is fully transparent, 1 is fully opaque)

        # Draw filled circles with opacity
        cv2.circle(overlay, (x_ring1, y_ring1), r_ring1, (0, 255, 0), -1)  # Green: Ring 1
        cv2.circle(overlay, (x_ring2, y_ring2), r_ring2, (255, 0, 0), -1)  # Blue: Ring 2
        cv2.circle(overlay, (x_ring3, y_ring3), r_ring3, (0, 0, 255), -1)  # Red: Ring 3

        # Blend the overlay onto the output image
        cv2.addWeighted(overlay, circle_opacity, output_image, 1 - circle_opacity, 0, output_image)

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
            map_mask = cv2.imdecode(np.frombuffer(map_mask_response.content, np.uint8), cv2.IMREAD_UNCHANGED)
            mountain_mask_response = requests.get(mountain_mask_url)
            mountain_mask = cv2.imdecode(np.frombuffer(mountain_mask_response.content, np.uint8), cv2.IMREAD_UNCHANGED)

            # Generate prediction
            output_image_bytes, ring3_coords = await generate_prediction(screenshot, map_mask, mountain_mask)
            if output_image_bytes:
                sent_message = await message.channel.send(
                    "Prediction complete. React with 🔄 to retry.",
                    file=discord.File(output_image_bytes, "prediction.png")
                )
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
        output_image_bytes, _ = await generate_prediction(
            client.screenshot, client.map_mask, client.mountain_mask
        )
        if output_image_bytes:
            await reaction.message.channel.send(
                "New prediction:",
                file=discord.File(output_image_bytes, "new_prediction.png")
            )

async def update_status():
    """
    Periodically updates the bot's status based on RAM usage every 15 seconds.
    """
    while True:
        ram_usage = psutil.Process().memory_info().rss / (1024 ** 2)  # RAM usage in MB
        if ram_usage < 500:
            emoji = "🟢"  # Green
        elif ram_usage < 750:
            emoji = "🟡"  # Yellow
        elif ram_usage < 1000:
            emoji = "🟠"  # Orange
        else:
            emoji = "🔴"  # Red

        await client.change_presence(activity=discord.Activity(type=discord.ActivityType.watching, name=f"{emoji} {ram_usage:.1f} MB used"))
        await asyncio.sleep(15)

# Run the bot
client.run(TOKEN)