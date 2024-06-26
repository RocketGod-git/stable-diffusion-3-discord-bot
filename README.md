# Stable Diffusion 3 - Discord Bot

A Discord bot for generating images using Stable Diffusion 3.

![image](https://github.com/RocketGod-git/stable-diffusion-3-discord-bot/assets/57732082/3ab6c2cc-48c7-4bad-808b-e6966552b950)

## Features
- Generate images from text prompts using Stable Diffusion 3.
- Easy to set up and run.
- Real-time interaction on Discord.

## Prerequisites
- Python
- GPU with CUDA installed
- Discord Developer account

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/RocketGod-git/stable-diffusion-3-discord-bot.git
   cd stable-diffusion-3-discord-bot
   ```

2. Configure your tokens:
   - Open `config.json` and add your Discord bot token and Huggingface token:
     ```json
     {
       "discord_bot_token": "YOUR-DISCORD-BOT-TOKEN-GOES-HERE",
       "huggingface_token": "YOUR-HUGGINGFACE-TOKEN-GOES-HERE"
     }
     ```

3. Run the bot using the batch script:
   ```bash
   sd3bot.bat
   ```

4. Setup the bot in Discord Developer Portal. I can't help with this so you'll have to figure it out.

## Usage

- Use the `/sd3` command in Discord followed by your prompt to generate an image.

## Contributing

Help make this better! Submit a PR if you have an improvement. 

## License

[LICENSE](LICENSE)

## Acknowledgements

- [Discord.py](https://github.com/Rapptz/discord.py)
- [Huggingface](https://huggingface.co/)
- [Stable Diffusion](https://github.com/CompVis/stable-diffusion)

![RocketGod](https://github.com/RocketGod-git/Flipper_Zero/assets/57732082/f5d67cfd-585d-4b23-905f-37151e3d6a7d)
