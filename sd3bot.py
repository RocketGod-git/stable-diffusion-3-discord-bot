# __________                  __             __     ________             .___ 
# \______   \  ____    ____  |  | __  ____ _/  |_  /  _____/   ____    __| _/ 
#  |       _/ /  _ \ _/ ___\ |  |/ /_/ __ \\   __\/   \  ___  /  _ \  / __ |  
#  |    |   \(  <_> )\  \___ |    < \  ___/ |  |  \    \_\  \(  <_> )/ /_/ |  
#  |____|_  / \____/  \___  >|__|_ \ \___  >|__|   \______  / \____/ \____ |  
#         \/              \/      \/     \/               \/              \/  
#
# Stable Diffusion 3 Discord Bot by RocketGod
#
# https://github.com/RocketGod-git/stable-diffusion-3-discord-bot


import discord
from discord.ext import commands
import logging
import json
from diffusers import StableDiffusion3Pipeline
import torch
from huggingface_hub import login
import os


with open("config.json", "r") as config_file:
    config = json.load(config_file)


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


access_token = config.get("huggingface_token", "")
login(access_token)


pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16)
pipe = pipe.to("cuda")


class SDBot(commands.Bot):
    def __init__(self, command_prefix, intents):
        super().__init__(command_prefix=command_prefix, intents=intents)

    async def setup_hook(self):
        await self.tree.sync()
        print(f'Synced {len(self.tree.get_commands())} command(s)')

    async def on_ready(self):
        logging.info(f'{self.user} is ready!')
        await self.update_activity()

    async def update_activity(self):
        guild_count = len(self.guilds)
        activity = discord.Activity(
            type=discord.ActivityType.watching,
            name=f"/sd3 on {guild_count} servers"
        )
        await self.change_presence(activity=activity)


intents = discord.Intents.all()
bot = SDBot(command_prefix="!", intents=intents)


@bot.tree.command(name="sd3", description="Stable Diffusion 3")
async def sd3(interaction: discord.Interaction, prompt: str):
    try:
        await interaction.response.defer()

        username = interaction.user.name
        filename = f"{username}_{interaction.id}.png"

        result = pipe(
            prompt,
            negative_prompt="",
            num_inference_steps=28,
            guidance_scale=7.0
        )
        image = result.images[0]
        image.save(filename)
        image_file = discord.File(fp=filename, filename=filename)

        await interaction.followup.send(f"`{prompt}`", file=image_file)

        logging.info(f"{username}: {prompt}")

        os.remove(filename)

    except Exception as e:
        await interaction.followup.send(f"An error occurred: {str(e)}")
        logging.error(f"Error occurred: {str(e)}")


TOKEN = config["discord_bot_token"]
bot.run(TOKEN)