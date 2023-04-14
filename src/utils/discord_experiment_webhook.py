import requests
import json


def send_discord_message(webhook_url, results):
    content = {
        "username": "ML Model Results",
        "embeds": [
            {
                "title": title,
                "color": color,
                "fields": [
                    {"name": key, "value": str(value), "inline": True} for key, value in data.items()
                ],
            } for title, color, data in results
        ],
    }

    response = requests.post(webhook_url, data=json.dumps(content), headers={"Content-Type": "application/json"})

    if response.status_code != 204:
        raise ValueError(f"Request to Discord returned an error {response.status_code}, the response is:\n{response.text}")


# Example usage
webhook_url = "WEBHOOK_URL"
results = [
    ("Training Parameters", 3447003, {
        "learning_rate": 0.01,
        "batch_size": 64,
        "epochs": 50,
    }),
    ("Train Results", 16711680, {
        "train_accuracy": 0.95,
        "train_loss": 0.12,
    }),
    ("Test Results", 255, {
        "test_accuracy": 0.90,
        "test_loss": 0.25,
    }),
]

send_discord_message(webhook_url, results)
