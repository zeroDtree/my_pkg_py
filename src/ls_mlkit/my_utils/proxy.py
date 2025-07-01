import os


def set_proxy(proxy_url: str = "http://127.0.0.1:7890"):
    r"""Set all common proxy environment variables."""
    os.environ.update(
        {
            "http_proxy": proxy_url,
            "https_proxy": proxy_url,
            "all_proxy": proxy_url,
            "HTTP_PROXY": proxy_url,
            "HTTPS_PROXY": proxy_url,
            "ALL_PROXY": proxy_url,
        }
    )
