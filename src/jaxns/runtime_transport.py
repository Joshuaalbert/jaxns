"""ZeroMQ transport setup shared by the coordinator, nodes, and workers."""

from __future__ import annotations

from pathlib import Path

from jaxns.runtime_config import NetworkConfig


def configure_curve_server(socket, network: NetworkConfig) -> None:
    """Load the coordinator identity before binding its TCP router."""
    import zmq.auth

    if network.server_public_key is None or network.server_secret_key is None:
        raise ValueError("The coordinator has incomplete CurveZMQ key paths.")
    declared_public, _ = zmq.auth.load_certificate(
        str(network.server_public_key)
    )
    public, secret = zmq.auth.load_certificate(str(network.server_secret_key))
    if declared_public is None or public is None or secret is None:
        raise ValueError("The coordinator certificate must contain both keys.")
    if declared_public != public:
        raise ValueError("The coordinator public and secret certificates disagree.")
    socket.curve_publickey = public
    socket.curve_secretkey = secret
    socket.curve_server = True
    socket.zap_domain = b"jaxns"


def configure_curve_client(socket, network: NetworkConfig) -> None:
    """Authenticate and encrypt one node-to-coordinator connection."""
    import zmq.auth

    paths = (
        network.server_public_key,
        network.client_public_key,
        network.client_secret_key,
    )
    if any(path is None for path in paths):
        raise ValueError("The worker node has incomplete CurveZMQ key paths.")
    server, _ = zmq.auth.load_certificate(str(network.server_public_key))
    declared_public, _ = zmq.auth.load_certificate(str(network.client_public_key))
    public, secret = zmq.auth.load_certificate(str(network.client_secret_key))
    if server is None or declared_public is None or public is None or secret is None:
        raise ValueError("The worker node certificates are incomplete.")
    if declared_public != public:
        raise ValueError("The worker node public and secret certificates disagree.")
    socket.curve_serverkey = server
    socket.curve_publickey = public
    socket.curve_secretkey = secret


def start_curve_authenticator(context, network: NetworkConfig):
    """Allow only public certificates explicitly installed by the operator."""
    from zmq.auth.thread import ThreadAuthenticator

    directory = network.authorized_clients
    if directory is None:
        raise ValueError("The coordinator has no authorized-client directory.")
    if not directory.is_dir():
        raise ValueError(
            f"Authorized-client directory does not exist: {directory}."
        )
    authenticator = ThreadAuthenticator(context)
    authenticator.start()
    authenticator.configure_curve(domain="jaxns", location=str(directory))
    return authenticator


def create_curve_certificate(directory: str | Path, name: str) -> tuple[Path, Path]:
    """Create one CurveZMQ public/secret certificate pair with safe modes."""
    import zmq.auth

    if not name or not all(
        character.isalnum() or character in "._-" for character in name
    ):
        raise ValueError(
            "Certificate name may contain only letters, digits, '.', '_', or '-'."
        )
    target = Path(directory).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True, mode=0o700)
    target.chmod(0o700)
    public, secret = zmq.auth.create_certificates(str(target), name)
    public_path = Path(public)
    secret_path = Path(secret)
    public_path.chmod(0o644)
    secret_path.chmod(0o600)
    return public_path, secret_path
