import os
import smtplib
from email.message import EmailMessage


def send_email(top5, digest_path):
    sender = os.environ["EMAIL_ADDRESS"]
    password = os.environ["EMAIL_APP_PASSWORD"]
    recipients = [
        email.strip()
        for email in os.environ["RECIPIENT_EMAIL"].split(",")
        if email.strip()
    ]

    if not recipients:
        raise ValueError("No recipient email addresses configured.")

    message = EmailMessage()
    message["Subject"] = "Daily AI News Digest"
    message["From"] = sender
    message["To"] = ", ".join(recipients)

    lines = ["Daily AI News Digest", ""]

    for i, item in enumerate(top5, start=1):
        lines.append(f"{i}. {item['title']}")
        lines.append(item["description"])
        lines.append(f"Why it made the top 5: {item['reasoning']}")
        lines.append(f"Read more: {item['link']}")
        lines.append("")

    message.set_content("\n".join(lines))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(sender, password.replace(" ", ""))
        smtp.send_message(message)

    print(
        f"Email sent successfully to {len(recipients)} recipient(s)."
    )