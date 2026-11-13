"""Send an experiment-completion email over SMTP SSL."""

from __future__ import annotations

import argparse
import getpass
import html
import os
import shlex
import smtplib
import socket
import ssl
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

FROM_NAME = os.environ.get("FROM_NAME", "mlkit")

REQUIRED_ENV_VARS = (
    "FROM_EMAIL",
    "FROM_SMTP_HOST",
    "FROM_SMTP_PORT",
    "FROM_SMTP_PASSWORD",
    "TO_EMAIL",
)


@dataclass(frozen=True)
class SmtpConfig:
    host: str
    port: int
    from_email: str
    from_name: str
    password: str
    to_email: str


@dataclass(frozen=True)
class ExperimentReport:
    experiment_name: str
    exit_code: int
    finished_at: str
    working_dir: str
    hostname: str
    user_name: str
    machine_ip: str
    extra_args: tuple[str, ...]


def load_smtp_config() -> SmtpConfig:
    missing = [name for name in REQUIRED_ENV_VARS if not os.environ.get(name)]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"missing environment variables: {joined}")

    port_text = os.environ["FROM_SMTP_PORT"]
    try:
        port = int(port_text)
    except ValueError as exc:
        raise ValueError(f"FROM_SMTP_PORT must be an integer, got {port_text!r}") from exc

    return SmtpConfig(
        host=os.environ["FROM_SMTP_HOST"],
        port=port,
        from_email=os.environ["FROM_EMAIL"],
        from_name=FROM_NAME,
        password=os.environ["FROM_SMTP_PASSWORD"],
        to_email=os.environ["TO_EMAIL"],
    )


def machine_ip() -> str:
    """Return the first address from `hostname -i`, or an empty string."""
    try:
        completed = subprocess.run(
            ["hostname", "-i"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return ""
    if completed.returncode != 0:
        return ""
    parts = completed.stdout.split()
    return parts[0] if parts else ""


def collect_report(exit_code: int, extra_args: list[str]) -> ExperimentReport:
    working_dir = Path.cwd()
    return ExperimentReport(
        experiment_name=working_dir.name,
        exit_code=exit_code,
        finished_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        working_dir=str(working_dir),
        hostname=socket.gethostname(),
        user_name=getpass.getuser(),
        machine_ip=machine_ip(),
        extra_args=tuple(extra_args),
    )


def render_subject(report: ExperimentReport) -> str:
    return f"{report.hostname} experiment done: {report.experiment_name}"


def render_extra_args_section(report: ExperimentReport) -> str:
    if not report.extra_args:
        return ""
    extra_args = html.escape(" ".join(shlex.quote(arg) for arg in report.extra_args))
    return f"""\
        <div style="background: white; padding: 15px; border-radius: 5px; margin-bottom: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <h3 style="margin-top: 0; color: #2c3e50;">Extra Info</h3>
            <pre style="margin: 0; white-space: pre-wrap; overflow-wrap: anywhere;">{extra_args}</pre>
        </div>
"""


def render_html(report: ExperimentReport) -> str:
    ok = report.exit_code == 0
    status = "Success" if ok else "Failed"
    status_color = "#27ae60" if ok else "#e74c3c"
    status_label = "OK" if ok else "FAIL"
    experiment_name = html.escape(report.experiment_name)
    finished_at = html.escape(report.finished_at)
    working_dir = html.escape(report.working_dir)
    hostname = html.escape(report.hostname)
    user_name = html.escape(report.user_name)
    machine_ip = html.escape(report.machine_ip)
    return f"""\
<html>
<head>
<meta http-equiv="Content-Type" content="text/html; charset=utf-8">
</head>
<body style="font-family: Arial, sans-serif; line-height: 1.4; color: #333; max-width: 720px; margin: 0 auto;">
    <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 8px 8px 0 0;">
        <h1 style="margin: 0; font-size: 24px;">{status_label} Experiment finished</h1>
    </div>
    <div style="padding: 20px; background: #f8f9fa; border-radius: 0 0 8px 8px;">
        {render_extra_args_section(report)}
        <div style="background: white; padding: 15px; border-radius: 5px; margin-bottom: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <h3 style="margin-top: 0; color: #2c3e50;">Experiment</h3>
            <p><strong>Name:</strong> {experiment_name}</p>
            <p><strong>Status:</strong> <span style="color: {status_color}; font-weight: bold;">{status}</span></p>
            <p><strong>Finished at:</strong> {finished_at}</p>
            <p><strong>Exit code:</strong> {report.exit_code}</p>
        </div>
        <div style="background: white; padding: 15px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <h3 style="margin-top: 0; color: #2c3e50;">Host</h3>
            <p><strong>Working dir:</strong> {working_dir}</p>
            <p><strong>Hostname:</strong> {hostname}</p>
            <p><strong>User:</strong> {user_name}</p>
            <p><strong>IP:</strong> {machine_ip}</p>
        </div>
    </div>
</body>
</html>
"""


def build_message(config: SmtpConfig, report: ExperimentReport) -> MIMEMultipart:
    message = MIMEMultipart("alternative")
    message["Subject"] = render_subject(report)
    message["From"] = f"{config.from_name} <{config.from_email}>"
    message["To"] = config.to_email
    message.attach(MIMEText(render_html(report), "html", "utf-8"))
    return message


def send_message(config: SmtpConfig, message: MIMEMultipart) -> None:
    # Corporate SMTP CAs are often missing on HPC images; require TLS but skip CA verify.
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    with smtplib.SMTP_SSL(config.host, config.port, context=context) as server:
        server.login(config.from_email, config.password)
        server.sendmail(config.from_email, [config.to_email], message.as_string())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send an experiment-completion email.")
    parser.add_argument("exit_code", nargs="?", type=int, default=0)
    parser.add_argument("extra_args", nargs="*")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = load_smtp_config()
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1
    report = collect_report(args.exit_code, args.extra_args)
    send_message(config, build_message(config, report))
    print("Notification sent successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
