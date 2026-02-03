#!/bin/bash
# ORB + Gooey Model Bot - Launch Script
# Runs daily at 9:25 AM ET before market open

cd ~/projects/orb-gooey-trader
source venv/bin/activate

# Set timezone
export TZ=America/New_York

# Log file with date
LOG_FILE="logs/gooey_$(date +%Y%m%d).log"

echo "=== ORB + Gooey Model Bot Starting ===" >> "$LOG_FILE"
echo "Date: $(date)" >> "$LOG_FILE"
echo "==========================================" >> "$LOG_FILE"

# Run the bot
python -m src.main >> "$LOG_FILE" 2>&1

echo "=== Bot Finished ===" >> "$LOG_FILE"
echo "Exit time: $(date)" >> "$LOG_FILE"
